# Learning Quantum Decoherence via Paraconsistent Logic: An Equation-Free Neural Network Framework

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Submission_Ready-success)]()

> **Supplementary Material for Manuscript Submission**
> *Official PyTorch implementation of ParaQNN.*

---

## Abstract

This repository provides the official implementation of ParaQNN, a neuro-symbolic learning framework that integrates Paraconsistent Logic ($\tau$-lattice) with deep neural networks for inverse modeling in open quantum dynamics.

Conventional physics-informed learning approaches rely on explicit governing equations and often degrade in regimes where such equations are incomplete, mismatched, or obscured by strongly non-Gaussian noise, including SPAM errors and telegraph processes. ParaQNN adopts a different epistemic stance by treating decoherence and environmental disturbances as structured, informative evidence rather than residual statistical error, formalized within a non-classical logical framework.

Central to the architecture is a learnable Degree of Contradiction ($\lambda$) embedded in the loss formulation, which enables the model to separate coherent unitary evolution from dissipative noise contributions directly from raw time-series measurements. This design supports equation-free reconstruction and characterization of quantum dynamics in settings where predefined differential equation solvers or explicit Hamiltonian models are unreliable or unavailable.

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
## Evaluation Protocol

All models (RF, XGBoost, PINN, GAN, ParaQNN) are trained and evaluated on identical 80/20 extrapolation splits.
Each benchmark is repeated over 5 independent random seeds (42–46) and reported as mean ± standard deviation.

ParaQNN is retrained independently for each seed and evaluated using seed-matched checkpoints.

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
python src/simulations/rabi_simulation.py --config configs/rabi.yaml
python src/simulations/lindblad_simulation.py --config configs/lindblad.yaml
python src/simulations/mixed_regime_simulation.py --config configs/mixed_regime.yaml

Model Training
python src/training/trainer.py --config configs/rabi.yaml --seed 42
python src/training/trainer.py --config configs/rabi.yaml --seed 43
python src/training/trainer.py --config configs/rabi.yaml --seed 44
python src/training/trainer.py --config configs/rabi.yaml --seed 45
python src/training/trainer.py --config configs/rabi.yaml --seed 46

python src/training/trainer.py --config configs/lindblad.yaml --seed 42
python src/training/trainer.py --config configs/lindblad.yaml --seed 43
python src/training/trainer.py --config configs/lindblad.yaml --seed 44
python src/training/trainer.py --config configs/lindblad.yaml --seed 45
python src/training/trainer.py --config configs/lindblad.yaml --seed 46

python src/training/trainer.py --config configs/mixed_regime.yaml --seed 42
python src/training/trainer.py --config configs/mixed_regime.yaml --seed 43
python src/training/trainer.py --config configs/mixed_regime.yaml --seed 44
python src/training/trainer.py --config configs/mixed_regime.yaml --seed 45
python src/training/trainer.py --config configs/mixed_regime.yaml --seed 46

Benchmarking
python src/benchmarks/rabi.py --config configs/rabi.yaml
python src/benchmarks/lindblad.py --config configs/lindblad.yaml
python src/benchmarks/mixed_regime.py --config configs/mixed_regime.yaml

Figure generation
python src/visualization/rabi_plotting.py
python src/visualization/lindblad_plotting.py
python src/visualization/mixed_regime_plotting.py

Jupyter Notebook
jupyter notebook notebooks/ParaQNN_Paper_Walkthrough.ipynb
```
## Reproducibility Snapshot (Paper Version)

The results reported in the manuscript were generated using:

- Git commit: `v1.0`
- Python: 3.10
- PyTorch: 2.5+
- CUDA: 12.8
- Random seeds: 42–46
- Config files:
  - `configs/rabi.yaml`
  - `configs/lindblad.yaml`
  - `configs/mixed_regime.yaml`

The `_quick.yaml` configurations are provided for functional verification only and do not reproduce the paper figures.

## Quick Start for Reviewers

To verify the functionality of the framework without running the full-scale experiments (which require 1500+ epochs and multiple random seeds), use the provided "quick" configurations. These runs complete in minutes on a standard CPU. **For expediency, these quick configurations are restricted to a single random seed (seed 42).**

“*_quick.yaml files are provided for fast smoke-testing and do not reproduce the reported numbers.”

> **Note:** Full-scale experiments (`rabi.yaml`, `lindblad.yaml`, `mixed_regime.yaml`) are configured to reproduce the paper figures exactly (high epoch counts, multiple seeds). Please be aware that these runs are computationally intensive and **may take a significant amount of time** (several hours) to complete, especially on CPU-only environments. For functional verification, we strongly recommend using the `*_quick.yaml` configurations.

**1. Rabi Regime Demo:**
```bash
# Train ParaQNN (100 epochs, single seed)
python src/training/trainer.py --config configs/rabi_quick.yaml
# Run Benchmarks (ParaQNN vs. Baselines) using the quick model
python src/benchmarks/rabi.py --config configs/rabi_quick.yaml
```
**1. Lindblad Regime Demo:**
```bash
python src/training/trainer.py --config configs/lindblad_quick.yaml
python src/benchmarks/lindblad.py --config configs/lindblad_quick.yaml
```
**1. Mixed Regime Demo:**
```bash
python src/training/trainer.py --config configs/mixed_quick.yaml
python src/benchmarks/mixed_regime.py --config configs/mixed_quick.yaml
```
## Data Availability

All datasets used in this work are generated deterministically by the simulation scripts in this repository.
No external or proprietary datasets are required to reproduce the results.

The exact datasets used in the manuscript are archived in the corresponding Zenodo release.

## Citation

@article{ceyran2026paraqnn,
  title={Equation-Free Discovery of Open Quantum Systems via Paraconsistent Neural Networks},
  author={Ceyran, Aleyna and Abe, Jair Minoro},
}

## License

This project is licensed under the **Apache License 2.0**.  
You are free to use, modify, and distribute this software for academic or commercial purposes, provided that proper attribution is given. See the [LICENSE](LICENSE) file for details.
