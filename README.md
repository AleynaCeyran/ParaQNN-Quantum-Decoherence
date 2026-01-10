# Equation-Free Discovery of Open Quantum Systems via Paraconsistent Neural Networks

[![License: Dual](https://img.shields.io/badge/License-Dual_Academic%2FCommercial-blue.svg)](#license)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Submission_Ready-success)]()

> **Supplementary Material for Scientific Reports Submission**
> *Official PyTorch implementation of ParaQNN.*

---

## Abstract

This repository presents the official implementation of **ParaQNN**, a novel neuro-symbolic architecture that integrates Paraconsistent Logic ($\tau$-Lattice) with deep learning to solve inverse problems in open quantum dynamics.

Standard Physics-Informed Neural Networks (PINNs) typically struggle when the underlying governing equations are unknown or when data is heavily corrupted by non-Gaussian noise (e.g., SPAM errors, telegraph noise). ParaQNN addresses this limitation by treating environmental noise and decoherence not merely as statistical outliers, but as **"contradictory evidence"** within a non-classical logical framework.

By assigning a "Degree of Contradiction" ($\lambda$) to the loss landscape, the model autonomously learns to disentangle coherent unitary evolution (truth) from dissipative noise channels (falsity) directly from raw time-series data. This approach enables equation-free discovery of physical laws without relying on differential equation solvers or prior Hamiltonian assumptions.

---
## Project Structure

This repository is structured to ensure full reproducibility of the results presented in the manuscript.

```text
ParaQNN-Project/
├── src/
│   ├── models/            # ParaQNN architecture & PANN Logic
│   ├── simulations/       # Quantum Master Equation solvers (Ground Truth)
│   ├── training/          # Training loops with dynamic loss weighting
│   └── benchmarks/        # Comparison baselines (PINN, RF, XGB, GAN)
├── data/                  # Synthetic datasets (Rabi, Lindblad, Mixed)
├── notebooks/             # Step-by-step reproduction walkthrough
├── configs/               # Hyperparameter configurations (.yaml)
└── results/               # Model checkpoints and generated figures
```


## Installation

We recommend using a virtual environment.

```bash
# 1. Clone the repository
git clone https://github.com/AleynaCeyran/ParaQNN-Quantum-Decoherence.git
cd ParaQNN-Quantum-Decoherence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the package in editable mode (Critical for imports)
pip install -e .
```

## Reproduction of Results
```bash
Data Generation
python src/simulations/mixed_regime_simulation.py --config configs/mixed_regime.yaml
#Repeat for rabi.yaml and lindblad.yaml

Model Training
python src/training/trainer.py --config configs/mixed_regime.yaml

Benchmarking
python src/benchmarks/mixed_regime.py

Jupyter Notebook
jupyter notebook notebooks/ParaQNN_Paper_Walkthrough.ipynb
```

## Citation
@article{ceyran2026paraqnn,
  title={Equation-Free Discovery of Open Quantum Systems via Paraconsistent Neural Networks},
  author={Ceyran, Aleyna and Abe, Jair Minoro},
}

## License
This project is released under a Dual License:

Academic/Non-Commercial: Free to use under MIT-like terms for research.

Commercial: Requires a separate license for proprietary use.

