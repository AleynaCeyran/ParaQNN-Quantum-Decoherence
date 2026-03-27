<div align="center">

# ParaQNN: Equation-Free Discovery of Open Quantum Systems

### via Paraconsistent Neural Networks

[![arXiv](https://img.shields.io/badge/arXiv-2601.12635-b31b1b.svg)](https://arxiv.org/abs/2601.12635)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: Dual](https://img.shields.io/badge/License-Dual_Academic%2FCommercial-blue.svg)](#license)
[![Status](https://img.shields.io/badge/Status-Submission_Ready-success.svg)]()
[![Code Style: Black](https://img.shields.io/badge/Code_Style-Black-000000.svg)](https://github.com/psf/black)

<p align="center">
  <img src="results/figures/mixed_regime_reconstruction.png" width="780" alt="ParaQNN Mixed Regime Reconstruction"/>
</p>

**ParaQNN** is a neuro-symbolic architecture that integrates Paraconsistent Logic (τ-Lattice) with deep learning to solve inverse problems in open quantum dynamics — **without any governing equations**.

[Paper](#citation) · [Quick Start](#quick-start) · [Reproduce Results](#reproducing-paper-results) · [Report Issue](https://github.com/AleynaCeyran/ParaQNN-Quantum-Decoherence/issues)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Reproducibility Snapshot](#reproducibility-snapshot)
- [Data Availability](#data-availability)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

Standard Physics-Informed Neural Networks (PINNs) fail when governing equations are unknown or data is corrupted by non-Gaussian noise (telegraph noise, 1/f pink noise, SPAM errors). **ParaQNN** addresses this by treating environmental noise and decoherence not as statistical outliers, but as *contradictory evidence* within a non-classical logical framework.

**Core idea:** Each neuron maintains two evidential channels — a *truth* channel (coherent signal) and a *falsity* channel (decoherent noise). A learnable contradiction coefficient `α` dynamically regulates their interaction via the Paraconsistent Interaction Activation Function (PIAF):

```
t_out = σ(k(z_t − α·z_f))
f_out = σ(k·z_f)
```

This allows the network to **discover physical structure** from noisy time-series data without a predefined Hamiltonian or master equation.

**Three benchmark regimes:**

| Regime | Physics | Noise Profile |
|---|---|---|
| Rabi | Damped unitary oscillations | Gaussian + telegraph |
| Lindblad | Dissipative open-system decay | Gaussian + telegraph |
| Mixed | Time-dependent switching dynamics | 1/f pink + SPAM |

---

## Key Results

Test-set MSE (mean ± std, 5 independent seeds):

| Model | Rabi | Lindblad | Mixed |
|---|---|---|---|
| Random Forest | 1.5×10⁻² | 2.8×10⁻² | 4.4×10⁻³ |
| XGBoost | 4.9×10⁻² | 3.0×10⁻² | 4.4×10⁻³ |
| PINN (Incomplete) | 1.5×10⁻² | 9.4×10⁻² | 1.5×10⁻² |
| PINN (Known) | 2.6×10⁻¹ | 3.0×10⁻¹ | 2.5×10⁻¹ |
| GAN | 4.2×10⁻² | 2.9×10⁻² | 8.7×10⁻² |
| **ParaQNN (ours)** | **1.9×10⁻⁴** | **4.9×10⁻⁷** | **7.7×10⁻⁶** |

ParaQNN outperforms the strongest baselines by **2–5 orders of magnitude** across all regimes.

---

## Installation

**Requirements:** Python 3.8+, pip, (optional) CUDA 11.8+

```bash
# 1. Clone the repository
git clone https://github.com/AleynaCeyran/ParaQNN-Quantum-Decoherence.git
cd ParaQNN-Quantum-Decoherence

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package in editable mode
pip install -e .
```

<details>
<summary><b>Dependency overview</b></summary>

| Package | Version | Purpose |
|---|---|---|
| torch | ≥ 2.0 | Core deep learning framework |
| numpy | ≥ 1.23 | Numerical computation |
| scipy | ≥ 1.9 | ODE integration for data generation |
| scikit-learn | ≥ 1.1 | Random Forest baseline |
| xgboost | ≥ 1.7 | XGBoost baseline |
| matplotlib | ≥ 3.6 | Figure generation |
| pyyaml | ≥ 6.0 | Config file parsing |
| jupyter | ≥ 1.0 | Notebook walkthrough |

</details>

---

## Quick Start

For functional verification without running full-scale experiments, use the `*_quick.yaml` configurations. These complete in **minutes on CPU** and do not reproduce the paper figures exactly.

```bash
# Rabi regime
python src/training/trainer.py --config configs/rabi_quick.yaml
python src/benchmarks/rabi.py  --config configs/rabi_quick.yaml

# Lindblad regime
python src/training/trainer.py --config configs/lindblad_quick.yaml
python src/benchmarks/lindblad.py --config configs/lindblad_quick.yaml

# Mixed regime
python src/training/trainer.py --config configs/mixed_quick.yaml
python src/benchmarks/mixed_regime.py --config configs/mixed_quick.yaml
```

> **Note:** `*_quick.yaml` files use 100 epochs and a single seed (42). They are intended for smoke-testing only. Full reproduction requires the standard config files — see [Reproducing Paper Results](#reproducing-paper-results).

Alternatively, follow the step-by-step notebook:

```bash
jupyter notebook notebooks/ParaQNN_Paper_Walkthrough.ipynb
```

---

## Repository Structure

```
ParaQNN-Quantum-Decoherence/
│
├── src/
│   ├── models/                  # ParaQNN architecture & PIAF activation
│   │   ├── paraqnn.py
│   │   └── paraconsistent_layer.py
│   ├── simulations/             # Quantum master equation solvers (ground truth)
│   │   ├── rabi_simulation.py
│   │   ├── lindblad_simulation.py
│   │   └── mixed_regime_simulation.py
│   ├── training/                # Training loops with dynamic loss weighting
│   │   └── trainer.py
│   ├── benchmarks/              # Baseline models (PINN, RF, XGBoost, GAN)
│   │   ├── rabi.py
│   │   ├── lindblad.py
│   │   └── mixed_regime.py
│   └── visualization/           # Figure generation scripts
│       ├── rabi_plotting.py
│       ├── lindblad_plotting.py
│       └── mixed_regime_plotting.py
│
├── data/                        # Synthetic datasets (generated deterministically)
├── notebooks/                   # Step-by-step reproduction walkthrough
├── configs/                     # Hyperparameter configurations (.yaml)
│   ├── rabi.yaml
│   ├── lindblad.yaml
│   ├── mixed_regime.yaml
│   └── *_quick.yaml             # Fast smoke-test configs
├── results/                     # Model checkpoints and generated figures
└── requirements.txt
```

---

## Reproducing Paper Results

Full reproduction uses the standard configs with 5 independent seeds and high epoch counts. **Expected runtime: several hours on GPU; significantly longer on CPU-only environments.**

### Step 1 — Data Generation

```bash
python src/simulations/rabi_simulation.py        --config configs/rabi.yaml
python src/simulations/lindblad_simulation.py    --config configs/lindblad.yaml
python src/simulations/mixed_regime_simulation.py --config configs/mixed_regime.yaml
```

### Step 2 — Model Training (5 seeds per regime)

```bash
for seed in 42 43 44 45 46; do
    python src/training/trainer.py --config configs/rabi.yaml         --seed $seed
    python src/training/trainer.py --config configs/lindblad.yaml     --seed $seed
    python src/training/trainer.py --config configs/mixed_regime.yaml --seed $seed
done
```

### Step 3 — Benchmarking

```bash
python src/benchmarks/rabi.py         --config configs/rabi.yaml
python src/benchmarks/lindblad.py     --config configs/lindblad.yaml
python src/benchmarks/mixed_regime.py --config configs/mixed_regime.yaml
```

### Step 4 — Figure Generation

```bash
python src/visualization/rabi_plotting.py
python src/visualization/lindblad_plotting.py
python src/visualization/mixed_regime_plotting.py
```

---

## Reproducibility Snapshot

The exact environment used to generate the manuscript figures:

| Component | Version |
|---|---|
| Git tag | `v1.0` |
| Python | 3.10 |
| PyTorch | 2.5+ |
| CUDA | 12.8 |
| Random seeds | 42, 43, 44, 45, 46 |
| OS | Ubuntu 22.04 |

Config files: `configs/rabi.yaml` · `configs/lindblad.yaml` · `configs/mixed_regime.yaml`

---

## Data Availability

All datasets are generated **deterministically** by the simulation scripts in this repository. No external or proprietary datasets are required.

The exact datasets used in the manuscript are archived at Zenodo (DOI provided in the paper).

To regenerate datasets from scratch, run the data generation scripts in Step 1 above.

---

## Citation

If you use this code or find this work helpful, please cite:

```bibtex
@article{ceyran2026paraqnn,
  title   = {Learning quantum decoherence via paraconsistent logic:
             an equation-free neural network framework},
  author  = {Ceyran, Aleyna and Abe, Jair Minoro},
  journal = {arXiv preprint arXiv:2601.12635},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.12635}
}
```

---

## License

This project is released under a **Dual License**:

- **Academic / Non-Commercial:** Free to use under MIT-like terms for research and educational purposes.
- **Commercial:** Requires a separate commercial license for proprietary or for-profit use.

See [`LICENSE.md`](LICENSE.md) for full legal terms.

---

## Acknowledgements

- **Paraconsistent logic foundations:** Da Costa (1974), Da Silva Filho et al. (2011–2016)
- **Paraconsistent Artificial Neural Network lineage:** Abe et al. (2004–2015)
- This work was conducted at the Department of Physics, Sakarya University, in collaboration with the Graduate Program in Production Engineering, Paulista University (UNIP), São Paulo, Brazil.

---

<div align="center">
<sub>Correspondence: <a href="mailto:aleyna.ceyran@ogr.sakarya.edu.tr">aleyna.ceyran@ogr.sakarya.edu.tr</a></sub>
</div>