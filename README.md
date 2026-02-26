## Graph Network-based Simulator
This repository contains a PyTorch Geometric implementation of a Graph Network-based Simulator (GNS), as popularized by DeepMind's research into particle-based physics. This model is capable of learning complex fluid dynamics (like the WaterDrop dataset) by treating particles as nodes in a dynamic graph.
## Project Overview
The simulator treats physical systems as graphs $\mathcal{G} = (V, E)$. It learns to predict the next state of the system by processing the spatial relationships between particles.Encoder: Embeds particle types and historical trajectories into a latent space.Processor: Uses 10 layers of Message Passing (Interaction Networks) to simulate internal forces and particle-to-particle constraints.Decoder: Predicts the acceleration of each particle, which is then integrated to update positions.
## Features
Adaptive Connectivity: Uses a radius_graph to dynamically update neighbors as particles move.Noise Injection: Implements a random-walk noise strategy during training to prevent "drift" during long-term rollouts.Multi-step History: Looks back at the previous 6 frames to capture velocity and acceleration trends.Boundary Handling: Includes normalized distance-to-wall features to help the model learn collision physics.
## Installation
Ensure you have PyTorch and PyTorch Geometric installed.Bashpip install torch torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
pip install h5py tqdm numpy
## Dataset Structure
The model expects data in HDF5 format. Place your files in the following directory structure:
PlaintextProcessed/
└── WaterDrop/
    ├── metadata.json      # Contains "bounds", "vel_mean", "acc_std", etc.
    ├── train.h5           # Training trajectories
    └── valid.h5           # Validation trajectories
**Explanation of files:**

*   **`metadata.json`**: This file should contain key metadata fields such as "bounds", "vel_mean", and "acc_std".
*   **`train.h5`**: This file contains the data for training trajectories.
*   **`valid.h5`**: This file contains the data for validation trajectories.
## Training & Configuration
The training logic includes an exponential learning rate scheduler and periodic "Rollout" evaluations to check long-term stability.Parameter Value Description:hidden_size is 128, Latent dimension of MLPs and GNN layers n_mp_layers is 10, Number of message-passing steps lr $1e-4$ Initial learning rate noise $3e-4$ Noise scale for training stabilitybatch_size 4 Number of trajectory windows per batchTo start training
## Evaluation Metrics
One-Step MSE: Measures the error of predicting the very next frame.Rollout MSE: Measures the accumulated error over a full sequence (e.g., 100+ steps). This is the "true" test of a physical simulator.
## Model Architecture Detail
The core interaction is defined by the InteractionNetwork:Edge Update: $e_{i,j}' = \phi_e([v_i, v_j, e_{i,j}])$ Node Update: $v_i' = \phi_v([v_i, \sum e_{i,j}'])$ Where $\phi_e$ and $\phi_v$ are Multi-Layer Perceptrons (MLPs).
