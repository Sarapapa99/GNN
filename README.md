## Learned Physical Simulator (GNS) 
for Particle DynamicsThis repository contains a PyTorch Geometric implementation of a Graph Network-based Simulator (GNS). The model is designed to learn complex physical interactions by representing particles as nodes in a graph and predicting their future states (accelerations) based on historical trajectories.## OverviewThe simulator follows the Encode-Process-Decode architecture:Encoder: Transforms node (particle) and edge (relational) features into a latent representation. It accounts for particle types, velocities, and proximity to boundaries.Processor: A stack of InteractionNetwork layers (Message Passing GNNs) that compute multi-step interactions between particles.Decoder: A Multi-Layer Perceptron (MLP) that extracts the predicted acceleration for each particle from the latent graph representation.## Key FeaturesDynamic Graph Construction: Uses radius_graph to establish connections between particles within a specific physical range.Noise Robustness: Implements a correlated noise walk during training to prevent error accumulation during long-term rollouts.Flexible Data Handling: Supports HDF5 datasets for efficient storage and loading of large-scale particle trajectories.Boundary Awareness: Normalizes and includes distances to simulation boundaries as a node feature.## Project StructurePlaintext├── Processed/
│   └── WaterDrop/
│       ├── metadata.json      # Simulation constants (radius, bounds, normalization stats)
│       ├── train.h5           # Training trajectory data
│       └── valid.h5           # Validation trajectory data
├── temp_gnn/
│   ├── models/                # Saved model checkpoints
│   └── rollouts/              # Generated simulation results
├── main.py                    # Training and evaluation script
└── README.md
## InstallationEnsure you have the following dependencies installed:Bashpip install torch torch-geometric torch-scatter h5py tqdm numpy
Note: Match your torch and torch-geometric versions with your CUDA toolkit version.## Usage1. Data PreparationPlace your metadata.json, train.h5, and valid.h5 files in the Processed/WaterDrop/ directory. The HDF5 files should contain position and optionally particle_type datasets.2. TrainingRun the script to begin training. The model will periodically save checkpoints and evaluate performance on the validation set.Python# The training loop includes:
# - One-step MSE (next-frame prediction)
# - Rollout MSE (long-term trajectory stability)
# - Learning rate scheduling
python main.py
3. HyperparametersYou can adjust the simulation parameters in the params dictionary within the script:| Parameter | Description | Default || :--- | :--- | :--- || epoch | Number of training passes | 5 || batch_size | Number of trajectories per batch | 4 || lr | Initial learning rate | $1e-4$ || noise | Standard deviation of training noise | $3e-4$ || n_mp_layers | Number of Message Passing layers | 10 |## Implementation DetailsInteraction NetworkThe core of the processor is the InteractionNetwork, which updates edge features based on sender/receiver nodes and then updates node features based on aggregated edge information:$$e_{i,j}' = \phi_e(e_{i,j}, v_i, v_j)$$$$v_i' = \phi_v(v_i, \sum_{j \in \mathcal{N}_i} e_{i,j}')$$IntegrationThe model predicts acceleration ($\ddot{p}$). The updated position is calculated using semi-implicit Euler integration:$v_{t+1} = v_t + \ddot{p} \cdot \Delta t$$p_{t+1} = p_t + v_{t+1} \cdot \Delta t$
