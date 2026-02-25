import os
import h5py
import torch
import json
import numpy as np
import torch_geometric as pyg
import math
import torch_scatter
import matplotlib.pyplot as plt
from matplotlib import animation
import glob
import re
from metrics import ot_loss, mmd_loss


# ============================================================================
# Configuration
# ============================================================================

DATASET_NAME = "WaterDrop"
PATH = os.path.join("Processed", DATASET_NAME)
MODEL_DIR = os.path.join("VsCode", "GNN","temp_gnn", "models", DATASET_NAME)
ROLLOUT_SAVE_DIR = os.path.join("temp_gnn", "rollouts", DATASET_NAME)
os.makedirs(ROLLOUT_SAVE_DIR, exist_ok=True)

# ============================================================================
# Copy necessary functions and classes from training code
# ============================================================================

def generate_noise(position_seq, noise_std):
    """Generate noise for a trajectory"""
    velocity_seq = position_seq[:, 1:] - position_seq[:, :-1]
    time_steps = velocity_seq.size(1)
    velocity_noise = torch.randn_like(velocity_seq) * (noise_std / time_steps ** 0.5)
    velocity_noise = velocity_noise.cumsum(dim=1)
    position_noise = velocity_noise.cumsum(dim=1)
    position_noise = torch.cat((torch.zeros_like(position_noise)[:, 0:1], position_noise), dim=1)
    return position_noise


def preprocess(particle_type, position_seq, target_position, metadata, noise_std):
    """Preprocess a trajectory and construct the graph"""
    position_noise = generate_noise(position_seq, noise_std)
    position_seq = position_seq + position_noise

    recent_position = position_seq[:, -1]
    velocity_seq = position_seq[:, 1:] - position_seq[:, :-1]

    n_particle = recent_position.size(0)
    edge_index = pyg.nn.radius_graph(recent_position, metadata["default_connectivity_radius"], loop=True, max_num_neighbors=n_particle)
    
    normal_velocity_seq = (velocity_seq - torch.tensor(metadata["vel_mean"])) / torch.sqrt(torch.tensor(metadata["vel_std"]) ** 2 + noise_std ** 2)
    boundary = torch.tensor(metadata["bounds"])
    distance_to_lower_boundary = recent_position - boundary[:, 0]
    distance_to_upper_boundary = boundary[:, 1] - recent_position
    distance_to_boundary = torch.cat((distance_to_lower_boundary, distance_to_upper_boundary), dim=-1)
    distance_to_boundary = torch.clip(distance_to_boundary / metadata["default_connectivity_radius"], -1.0, 1.0)

    dim = recent_position.size(-1)
    edge_displacement = (torch.gather(recent_position, dim=0, index=edge_index[0].unsqueeze(-1).expand(-1, dim)) -
                   torch.gather(recent_position, dim=0, index=edge_index[1].unsqueeze(-1).expand(-1, dim)))
    edge_displacement /= metadata["default_connectivity_radius"]
    edge_distance = torch.norm(edge_displacement, dim=-1, keepdim=True)

    if target_position is not None:
        last_velocity = velocity_seq[:, -1]
        next_velocity = target_position + position_noise[:, -1] - recent_position
        acceleration = next_velocity - last_velocity
        acceleration = (acceleration - torch.tensor(metadata["acc_mean"])) / torch.sqrt(torch.tensor(metadata["acc_std"]) ** 2 + noise_std ** 2)
    else:
        acceleration = None

    graph = pyg.data.Data(
        x=particle_type,
        edge_index=edge_index,
        edge_attr=torch.cat((edge_displacement, edge_distance), dim=-1),
        y=acceleration,
        pos=torch.cat((velocity_seq.reshape(velocity_seq.size(0), -1), distance_to_boundary), dim=-1)
    )
    return graph


class RolloutDataset(pyg.data.Dataset):
    def __init__(self, path, split, trajectory=None):
        super().__init__()
        with open(os.path.join(path, "metadata.json")) as f:
            self.metadata = json.load(f)

        self.db = h5py.File(os.path.join(path, split + ".h5"), "r")
        self.keys = list(self.db.keys())

        if trajectory is not None:
            self.keys = [self.keys[trajectory]]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        pos = torch.tensor(self.db[f"{key}/position"][:], dtype=torch.float)
        ptype = torch.tensor(
            self.db[f"{key}/particle_type"][:] if f"{key}/particle_type" in self.db else
            self.db["particle_type"][:] if "particle_type" in self.db else
            torch.zeros(pos.size(1)),
            dtype=torch.long
        )

        return {"position": pos, "particle_type": ptype, "metadata": self.metadata}


class MLP(torch.nn.Module):
    """Multi-Layer perceptron"""
    def __init__(self, input_size, hidden_size, output_size, layers, layernorm=True):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(layers):
            self.layers.append(torch.nn.Linear(
                input_size if i == 0 else hidden_size,
                output_size if i == layers - 1 else hidden_size,
            ))
            if i != layers - 1:
                self.layers.append(torch.nn.ReLU())
        if layernorm:
            self.layers.append(torch.nn.LayerNorm(output_size))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(0, 1 / math.sqrt(layer.in_features))
                layer.bias.data.fill_(0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class InteractionNetwork(pyg.nn.MessagePassing):
    """Interaction Network"""
    def __init__(self, hidden_size, layers):
        super().__init__()
        self.lin_edge = MLP(hidden_size * 3, hidden_size, hidden_size, layers)
        self.lin_node = MLP(hidden_size * 2, hidden_size, hidden_size, layers)

    def forward(self, x, edge_index, edge_feature):
        edge_out, aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature)
        node_out = self.lin_node(torch.cat((x, aggr), dim=-1))
        edge_out = edge_feature + edge_out
        node_out = x + node_out
        return node_out, edge_out

    def message(self, x_i, x_j, edge_feature):
        x = torch.cat((x_i, x_j, edge_feature), dim=-1)
        x = self.lin_edge(x)
        return x

    def aggregate(self, inputs, index, dim_size=None):
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        return (inputs, out)


class LearnedSimulator(torch.nn.Module):
    """Graph Network-based Simulators(GNS)"""
    def __init__(
        self,
        hidden_size=128,
        n_mp_layers=10,
        num_particle_types=9,
        particle_type_dim=16,
        dim=2,
        window_size=5,
    ):
        super().__init__()
        self.window_size = window_size
        self.embed_type = torch.nn.Embedding(num_particle_types, particle_type_dim)
        self.node_in = MLP(particle_type_dim + dim * (window_size + 2), hidden_size, hidden_size, 3)
        self.edge_in = MLP(dim + 1, hidden_size, hidden_size, 3)
        self.node_out = MLP(hidden_size, hidden_size, dim, 3, layernorm=False)
        self.n_mp_layers = n_mp_layers
        self.layers = torch.nn.ModuleList([InteractionNetwork(
            hidden_size, 3
        ) for _ in range(n_mp_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embed_type.weight)

    def forward(self, data):
        node_feature = torch.cat((self.embed_type(data.x), data.pos), dim=-1)
        node_feature = self.node_in(node_feature)
        edge_feature = self.edge_in(data.edge_attr)
        for i in range(self.n_mp_layers):
            node_feature, edge_feature = self.layers[i](node_feature, data.edge_index, edge_feature=edge_feature)
        out = self.node_out(node_feature)
        return out

# ============================================================================
# Rollout Function
# ============================================================================

def rollout(model, data, metadata, noise_std):
    device = next(model.parameters()).device
    model.eval()
    window_size = model.window_size + 1
    total_time = data["position"].size(0)
    traj = data["position"][:window_size]
    traj = traj.permute(1, 0, 2)
    particle_type = data["particle_type"]

    for time in range(total_time - window_size):
        with torch.no_grad():
            graph = preprocess(particle_type, traj[:, -window_size:], None, metadata, 0.0)
            graph = graph.to(device)
            acceleration = model(graph).cpu()
            acceleration = acceleration * torch.sqrt(torch.tensor(metadata["acc_std"]) ** 2 + noise_std ** 2) + torch.tensor(metadata["acc_mean"])

            recent_position = traj[:, -1]
            recent_velocity = recent_position - traj[:, -2]
            new_velocity = recent_velocity + acceleration
            new_position = recent_position + new_velocity
            traj = torch.cat((traj, new_position.unsqueeze(1)), dim=1)

    return traj

# ============================================================================
# Visualization Functions
# ============================================================================

TYPE_TO_COLOR = {
    3: "black",
    0: "green",
    7: "magenta",
    6: "gold",
    5: "blue",
}


def visualize_prepare(ax, particle_type, position, metadata):
    bounds = metadata["bounds"]
    ax.set_xlim(bounds[0][0], bounds[0][1])
    ax.set_ylim(bounds[1][0], bounds[1][1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1.0)
    points = {type_: ax.plot([], [], "o", ms=2, color=color)[0] for type_, color in TYPE_TO_COLOR.items()}
    return ax, position, points


def visualize_pair(particle_type, position_pred, position_gt, metadata, title_suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_info = [
        visualize_prepare(axes[0], particle_type, position_gt, metadata),
        visualize_prepare(axes[1], particle_type, position_pred, metadata),
    ]
    axes[0].set_title("Ground truth")
    axes[1].set_title(f"Prediction {title_suffix}")

    plt.close()

    def update(step_i):
        outputs = []
        for _, position, points in plot_info:
            for type_, line in points.items():
                mask = particle_type == type_
                line.set_data(position[step_i, mask, 0], position[step_i, mask, 1])
            outputs.append(line)
        return outputs

    return animation.FuncAnimation(fig, update, frames=np.arange(0, position_gt.size(0)), interval=10, blit=True)

# ============================================================================
# Find all checkpoints
# ============================================================================

def find_checkpoints(model_dir):
    """Find all checkpoint files and sort them by step number"""
    checkpoint_files = glob.glob(os.path.join(model_dir, "checkpoint_*.pt"))
    
    # Extract step numbers and sort
    checkpoints = []
    for file in checkpoint_files:
        basename = os.path.basename(file)
        # Match checkpoint_XXXX.pt or checkpoint_final.pt
        match = re.search(r'checkpoint_(\d+|final)\.pt', basename)
        if match:
            step = match.group(1)
            if step == "final":
                step_num = float('inf')  # Put final at the end
            else:
                step_num = int(step)
            checkpoints.append((step_num, file, step))
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

# ============================================================================
# Main Visualization
# ============================================================================

if __name__ == "__main__":
    # Load only checkpoint_850000
    checkpoint_path = os.path.join(MODEL_DIR, "checkpoint_850000.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        exit(1)
    
    print(f"Processing checkpoint: checkpoint_850000.pt")
    
    # Load rollout dataset
    print("\nLoading rollout dataset...")
    rollout_dataset = RolloutDataset(PATH, "valid")
    rollout_data = rollout_dataset[0]
    noise_std = 3e-4
    
    # Load model
    print("Loading model...")
    simulator = LearnedSimulator()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    simulator.load_state_dict(checkpoint["model"])
    simulator.eval()
    
    # Run rollout
    print("Running rollout simulation...")
    rollout_out = rollout(simulator, rollout_data, rollout_dataset.metadata, noise_std)
    rollout_out = rollout_out.permute(1, 0, 2)
    
    # Calculate MSE
    mse = ((rollout_out - rollout_data["position"]) ** 2).mean().item()
    print(f"Rollout MSE: {mse:.6f}")
    

    # Additional evaluation metrics - loop over all trajectories
    print("\nEvaluating OT/MMD over all trajectories...")
    ot_losses = []
    mmd_losses_list = []

    for i in range(len(rollout_dataset)):
        traj_data = rollout_dataset[i]
        traj_out = rollout(simulator, traj_data, rollout_dataset.metadata, noise_std)
        traj_out = traj_out.permute(1, 0, 2)

        with torch.no_grad():
            gt = traj_data["position"]
            ot_val = ot_loss(traj_out, gt).item()
            mmd_val = mmd_loss(traj_out, gt).item()

        ot_losses.append(ot_val)
        mmd_losses_list.append(mmd_val)
        print(f"  Trajectory {i}: OT={ot_val:.6f}, MMD={mmd_val:.6f}")

    mean_ot = np.mean(ot_losses)
    mean_mmd = np.mean(mmd_losses_list)
    print(f"\nMean OT distance across all trajectories:  {mean_ot:.6f}")
    print(f"Mean MMD distance across all trajectories: {mean_mmd:.6f}")

    # Create animation
    print("Creating animation...")
    anim = visualize_pair(
        rollout_data["particle_type"], 
        rollout_out, 
        rollout_data["position"], 
        rollout_dataset.metadata,
        title_suffix=f"(Step 850000 | MSE:{mse:.4f} OT:{mean_ot:.4f} MMD:{mean_mmd:.4f})"
    )
    
    # Save animation
    gif_path = os.path.join(ROLLOUT_SAVE_DIR, "rollout_step_850000_otmean_mmdmean.gif")
    print(f"Saving animation to {gif_path}...")
    anim.save(gif_path, writer="pillow", fps=30)
    print("Done!")
