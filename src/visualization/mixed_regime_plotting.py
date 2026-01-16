"""
Mixed Regime Visualization Suite 

Description:
    Generates publication-quality figures for the Time-Dependent Mixed Regime experiment.
    This scenario involves switching dynamics: Strong Drive -> Free Decay -> Weak Probe.
"""

import sys
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from pathlib import Path
from typing import Optional, Dict

# Path Setup 
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.paraqnn import ParaQNN

# PUBLICATION STYLE CONFIGURATION 
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.fontsize': 9,
    'figure.dpi': 600,
    'lines.linewidth': 1.5,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.grid': True,
    'grid.alpha': 0.15,
    'grid.linewidth': 0.5,
    'grid.linestyle': '-',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'mathtext.fontset': 'stixsans',
})

COLORS = {
    'data': '#7F8C8D',
    'ideal': '#000000',
    'model': '#C0392B',
    'train': '#2874A6',
    'val': '#D35400',
    'alpha': '#2874A6',
    'phase_line': '#2C3E50'
}

DEFAULT_SEED = 42

def get_save_dir() -> Path:
    save_dir = project_root / 'results' / 'figures' / 'mixed_regime_analysis'
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir

def load_history() -> Optional[Dict]:
    possible_paths = [
        project_root / 'checkpoints' / 'mixed' / f'seed_{DEFAULT_SEED}' / 'training_history.npy',
        project_root / 'checkpoints' / 'mixed_quick' / f'seed_{DEFAULT_SEED}' / 'training_history.npy',
        project_root / 'checkpoints' / 'mixed' / 'training_history.npy'
    ]
    for p in possible_paths:
        if p.exists():
            return np.load(p, allow_pickle=True).item()
    return None

def add_panel_label(ax, label: str):
    ax.text(0.04, 0.94, label, transform=ax.transAxes, 
            fontsize=14, fontweight='normal', va='top', ha='left')

def plot_fig5a_simulation():
    print(">>> Generating Fig5a (Simulation)...")
    data_path = project_root / 'data' / 'synthetic' / 'mixed_regime_data.npz'
    
    if not data_path.exists():
        print(f"Dataset not found at {data_path}")
        return

    data = np.load(data_path)
    t_np = data['t'] if 't' in data else data['time']
    y_noisy = data['signal'] if 'signal' in data else data['signal_noisy']
    y_ideal = data['ideal'] if 'ideal' in data else data['signal_ideal']
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    idx = np.arange(0, len(t_np), 5) if len(t_np) > 5000 else np.arange(len(t_np))
    
    ax.scatter(t_np[idx], y_noisy[idx], color=COLORS['data'], s=8, alpha=0.3, label='Measurement', rasterized=True)
    ax.plot(t_np, y_ideal, color=COLORS['ideal'], linewidth=1.5, label='Hamiltonian (Hidden)')
    
    ax.axvline(x=4.0, color=COLORS['phase_line'], linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axvline(x=7.0, color=COLORS['phase_line'], linestyle='--', linewidth=0.8, alpha=0.6)
    
    ax.text(2.0, 0.90, "Drive", ha='center', fontsize=9, color=COLORS['phase_line'])
    ax.text(5.5, 0.90, "Decay", ha='center', fontsize=9, color=COLORS['phase_line'])
    ax.text(8.5, 0.90, "Probe", ha='center', fontsize=9, color=COLORS['phase_line'])

    ax.set_xlabel(r"Time ($\mu s$)")
    ax.set_ylabel(r"Population $P(|1\rangle)$")
    ax.set_ylim(-0.2, 1.2)
    ax.legend(loc='upper right', frameon=False)
    add_panel_label(ax, '(a)')

    plt.savefig(get_save_dir() / 'Fig5a_Mixed_Simulation.pdf', format='pdf', bbox_inches='tight')
    plt.close()

def plot_fig5b_loss():
    print(">>> Generating Fig5b (Loss)...")
    history = load_history()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    if history:
        ax.plot(history['train_loss'], color=COLORS['train'], label='Train', linewidth=1.5)
        ax.plot(history['val_loss'], color=COLORS['val'], linestyle='--', label='Validation', linewidth=1.5)
        ax.set_yscale('log')
        ax.set_xlabel("Epochs")
        ax.set_ylabel(r"Loss $\mathcal{L}$")
        ax.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.2)
        ax.legend(loc='upper right', frameon=False)
    else:
        ax.text(0.5, 0.5, "Data Missing", ha='center', color='red')
        
    add_panel_label(ax, '(b)')
    plt.savefig(get_save_dir() / 'Fig5b_Mixed_Loss.pdf', format='pdf', bbox_inches='tight')
    plt.close()

def plot_fig5c_alpha():
    print(">>> Generating Fig5c (Alpha)...")
    history = load_history()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    alpha_data = None
    if history:
        for k in ("alpha_first", "alpha_history", "alpha_mean_history", "alpha_mean", "alpha_values", "alpha", "alphas"):
            if k in history and history[k] is not None:
                alpha_data = history[k]
                break

    if alpha_data is not None:
        alpha_arr = np.asarray(alpha_data)
        if alpha_arr.ndim == 0:
            n = len(history.get("train_loss", [])) if history else 2
            alpha_arr = np.full(max(int(n), 2), float(alpha_arr), dtype=float)
        else:
            alpha_arr = alpha_arr.reshape(-1).astype(float)

        ax.plot(alpha_arr, color=COLORS['alpha'], linewidth=2.0, label=r'$\alpha$')
        ax.set_xlabel("Epochs")
        ax.set_ylabel(r"Parameter $\alpha$")
        ax.legend(loc='lower right', frameon=False)
    else:
        ax.text(0.5, 0.5, "Alpha Missing", ha='center', color='red')

    add_panel_label(ax, '(c)')
    plt.savefig(get_save_dir() / 'Fig5c_Mixed_Alpha.pdf', format='pdf', bbox_inches='tight')
    plt.close()

def plot_fig5d_discovery():
    print(">>> Generating Fig5d (Discovery)...")
    
    data_path = project_root / 'data' / 'synthetic' / 'mixed_regime_data.npz'
    ckpt_path = project_root / 'checkpoints' / 'mixed' / f'seed_{DEFAULT_SEED}' / 'best_model.pth'
    
    if not ckpt_path.exists():
         ckpt_path = project_root / 'checkpoints' / 'mixed_quick' / f'seed_{DEFAULT_SEED}' / 'best_model.pth'

    if not data_path.exists(): return
    data = np.load(data_path)
    t_np = data['t'] if 't' in data else data['time']
    y_ideal = data['ideal'] if 'ideal' in data else data['signal_ideal']
    y_noisy = data['signal'] if 'signal' in data else data['signal_noisy']
    
    y_pred = None
    
    if not ckpt_path.exists():
        print(f"CRITICAL: Model checkpoint not found at {ckpt_path}. Skipping plot.")
        return

    try:
        # Try full configuration first
        model = ParaQNN(input_dim=1, hidden_dim=128, output_dim=1, num_layers=3, initial_alpha=5.0)
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        except RuntimeError:
            print("Size mismatch in plotting, trying hidden=64 (Quick Config)...")
            model = ParaQNN(input_dim=1, hidden_dim=64, output_dim=1, num_layers=3, initial_alpha=5.0)
            model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

        model.eval()
        with torch.no_grad():
            t_tensor = torch.tensor(t_np, dtype=torch.float32).unsqueeze(1)
            y_pred = model(t_tensor)[0].numpy().flatten()
    except Exception as e:
        raise RuntimeError(f"Model failed to load: {e}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    idx = np.arange(0, len(t_np), 5) if len(t_np) > 5000 else np.arange(len(t_np))
    
    ax.scatter(t_np[idx], y_noisy[idx], color=COLORS['data'], s=10, alpha=0.4, label='Measurement', rasterized=True, zorder=1)
    ax.plot(t_np, y_ideal, color='k', linestyle=':', linewidth=1.5, label='Ideal Physics', alpha=0.8, zorder=2)
    ax.plot(t_np, y_pred, color=COLORS['model'], linewidth=2.0, label='ParaQNN', zorder=3)
    
    ax.axvline(x=4.0, color=COLORS['phase_line'], linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axvline(x=7.0, color=COLORS['phase_line'], linestyle='--', linewidth=0.8, alpha=0.6)

    ax.set_xlabel(r"Time ($\mu s$)")
    ax.set_ylabel(r"Population $P(|1\rangle)$") 
    ax.set_ylim(-0.2, 1.2)
    ax.legend(loc='upper right', frameon=False, ncol=1)
    add_panel_label(ax, '(d)')

    # Inset Zoom (Decay Phase)
    axins = inset_axes(ax, width="40%", height="35%", loc='lower center', borderpad=2)
    mask = (t_np > 4.5) & (t_np < 6.5)
    
    if np.any(mask):
        axins.scatter(t_np[mask], y_noisy[mask], color=COLORS['data'], s=15, alpha=0.4, rasterized=True, zorder=1)
        axins.plot(t_np[mask], y_ideal[mask], color='k', linestyle=':', linewidth=1.5, zorder=2)
        axins.plot(t_np[mask], y_pred[mask], color=COLORS['model'], linewidth=2.0, zorder=3)
        axins.set_xlim(4.5, 6.5)
        y_section = y_ideal[mask]
        axins.set_ylim(y_section.min()-0.15, y_section.max()+0.15)
        axins.set_xticks([])
        axins.set_yticks([])
        mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5", alpha=0.5)

    plt.savefig(get_save_dir() / 'Fig5d_Mixed_Discovery.pdf', format='pdf', bbox_inches='tight')
    plt.close()

def plot_fig6_benchmark():
    print(">>> Generating Fig6 (Benchmark)...")

    metrics_path = project_root / "results" / "benchmarks" / "mixed_metrics.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Benchmark results not found at {metrics_path}. Please run 'python src/benchmarks/mixed_regime.py' first.")

    with open(metrics_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    metrics = loaded.get("metrics", loaded)
    if "benchmark_mse" in loaded: metrics = loaded["benchmark_mse"]

    key_map = {
        "RF": "RF",
        "XGB": "XGB",
        "PINN (Inc)": "PINN_Incomplete",
        "PINN (Known)": "PINN_Known",
        "GAN": "GAN",
        "ParaQNN": "ParaQNN",
    }

    data = {}
    for label, key in key_map.items():
        val = metrics.get(key)
        # Try fallback keys
        if val is None and key == "PINN_Incomplete": val = metrics.get("PINN (Inc)")
        if val is None and key == "PINN_Known": val = metrics.get("PINN (Known)")
        
        data[label] = float(val) if val is not None else 0.0

    models = ["RF", "XGB", "PINN (Inc)", "PINN (Known)", "GAN", "ParaQNN"]
    mse_values = [data[m] for m in models]

    fig, ax = plt.subplots(figsize=(7, 5))
    bar_colors = ['#BDC3C7', '#95A5A6', '#7F8C8D', '#566573', '#2C3E50', COLORS['model']]
    bars = ax.bar(models, mse_values, color=bar_colors, edgecolor='black', linewidth=0.5, width=0.6)
     
    ax.set_yscale("log")
    
    ax.set_ylabel("Mean-Squared Error (MSE)")
    ax.set_ylim(top=max(mse_values) * 50 if max(mse_values) > 0 else 1.0)
    ax.grid(True, which="both", axis="y", linestyle="-", linewidth=0.3, alpha=0.3)

    for bar in bars:
        h = float(bar.get_height())
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h * 1.2,
                f"{h:.1e}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#222222",
            )

    out_path = get_save_dir() / "Fig6_Mixed_Benchmark.pdf"
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    plot_fig5a_simulation()
    plot_fig5b_loss()
    plot_fig5c_alpha()
    plot_fig5d_discovery()
    plot_fig6_benchmark()
    print(f"Mixed Regime figures generated successfully at: {get_save_dir()}")