# GNS: Graph Network-based Simulator (PyTorch Geometric)

This repository provides a PyTorch implementation of a **Graph Network-based Simulator (GNS)**, a powerful framework for learning particle-based physical simulations (e.g., fluid dynamics, sand, or cloth). This implementation is optimized for the `WaterDrop` dataset.



##  Overview

The simulator treats physical systems as dynamic graphs where particles are **nodes** and spatial interactions are **edges**. It utilizes an **Encode-Process-Decode** architecture:

1.  **Encoder**: Constructs a graph from particle positions using a radius-based search. It embeds particle types and captures historical velocity sequences.
2.  **Processor**: Executes 10 steps of message passing using **Interaction Networks** to propagate "forces" and constraints through the system.
3.  **Decoder**: Transforms the processed latent features into predicted accelerations.



---

##  Features

* **Dynamic Graph Construction**: Automatically builds edges using `radius_graph` based on a connectivity radius defined in metadata.
* **Noise Injection**: Implements a random-walk noise strategy during training to improve the stability of long-term rollouts.
* **Boundary Awareness**: Incorporates normalized distance-to-boundary features, allowing the model to learn collision physics.
* **Accumulated History**: Utilizes a window of previous frames (default: 6) to provide the model with momentum and acceleration context.

---

## Installation

This project requires **PyTorch** and **PyTorch Geometric**. Ensure you have a compatible CUDA version installed.

```bash
# Install core dependencies
pip install torch torch-scatter torch-sparse torch-cluster torch-geometric
pip install h5py tqdm numpy

Processed/
└── WaterDrop/
    ├── metadata.json      # Simulation constants (radius, bounds, normalization stats)
    ├── train.h5           # Training trajectory data
    └── valid.h5           # Validation trajectory data

Parameter,Default,Description
hidden_size,128,Latent dimension of MLPs and GNN layers
n_mp_layers,10,Number of message-passing steps (Processor depth)
lr,1e-4,Learning rate with exponential decay
noise,3e-4,Standard deviation of training noise
window_size,5,Number of historical frames provided to the model
