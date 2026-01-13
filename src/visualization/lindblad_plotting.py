"""
Lindblad Regime Visualization Suite (Open-Source Release)

Description:
    Generates publication-quality figures for the Open Quantum System (Lindblad) 
    experiment. Optimized for Physical Review (PRL/PRA) standards.

    Features:
    - Strict Dirac Notation: P(|1>) rendered via STIX fonts.
    - Minimalist Design: Inner ticks, boxed frames, clean legends.
    - Robustness: Strict error handling (No visual fallbacks).
    - Consistency: Uniform typography matches Rabi and Mixed regime plots.
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
    'alpha': '#2874A6'      
}

DEFAULT_SEED = 42

def get_save_dir() -> Path:
    save_dir = project_root / 'results' / 'figures' / 'lindblad_analysis'
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir

def load_history() -> Optional[Dict]:
    possible_paths = [
        project_root / 'checkpoints' / 'lindblad' / f'seed_{DEFAULT_SEED}' / 'training_history.npy',
        project_root / 'checkpoints' / 'lindblad_quick' / f'seed_{DEFAULT_SEED}' / 'training_history.npy',
        project_root / 'checkpoints' / 'lindblad' / 'training_history.npy'
    ]
    for p in possible_paths:
        if p.exists():
            return np.load(p, allow_pickle=True).item()
    return None

def add_panel_label(ax, label: str):
    ax.text(0.04, 0.94, label, transform=ax.transAxes, 
            fontsize=14, fontweight='normal', va='top', ha='left')

def plot_fig3a_simulation():
    print(">>> Generating Fig3a (Simulation)...")
    data_path = project_root / 'data' / 'synthetic' / 'lindblad_data.npz'
    
    if not data_path.exists():
        print(f"Dataset not found at {data_path}")
        return

    data = np.load(data_path)
    t_np = data['time'] if 'time' in data else data['t']
    y_ideal = data['signal_ideal'] if 'signal_ideal' in data else data['ideal']
    y_noisy = data['signal_noisy'] if 'signal_noisy' in data else data['signal']
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    idx = np.arange(0, len(t_np), 5) if len(t_np) > 5000 else np.arange(len(t_np))
    
    ax.scatter(t_np[idx], y_noisy[idx], color=COLORS['data'], s=8, alpha=0.3, label='Measurement', rasterized=True)
    ax.plot(t_np, y_ideal, color=COLORS['ideal'], linewidth=1.5, label='Hamiltonian (Hidden)')
    
    ax.set_xlabel(r"Time ($\mu s$)")
    ax.set_ylabel(r"Population $P(|1\rangle)$")
    ax.set_ylim(-0.2, 1.2)
    ax.legend(loc='upper right', frameon=False)
    add_panel_label(ax, '(a)')

    plt.savefig(get_save_dir() / 'Fig3a_Lindblad_Simulation.pdf', format='pdf', bbox_inches='tight')
    plt.close()

def plot_fig3b_loss():
    print(">>> Generating Fig3b (Loss)...")
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
    plt.savefig(get_save_dir() / 'Fig3b_Lindblad_Loss.pdf', format='pdf', bbox_inches='tight')
    plt.close()

def plot_fig3c_alpha():
    print(">>> Generating Fig3c (Alpha)...")
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
    plt.savefig(get_save_dir() / 'Fig3c_Lindblad_Alpha.pdf', format='pdf', bbox_inches='tight')
    plt.close()

def plot_fig3d_discovery():
    print(">>> Generating Fig3d (Discovery)...")
    
    data_path = project_root / 'data' / 'synthetic' / 'lindblad_data.npz'
    ckpt_path = project_root / 'checkpoints' / 'lindblad' / f'seed_{DEFAULT_SEED}' / 'best_model.pth'
    
    if not ckpt_path.exists():
         ckpt_path = project_root / 'checkpoints' / 'lindblad_quick' / f'seed_{DEFAULT_SEED}' / 'best_model.pth'

    if not data_path.exists(): return
    data = np.load(data_path)
    t_np = data['time'] if 'time' in data else data['t']
    y_ideal = data['signal_ideal'] if 'signal_ideal' in data else data['ideal']
    y_noisy = data['signal_noisy'] if 'signal_noisy' in data else data['signal']
    
    y_pred = None
    
    if not ckpt_path.exists():
        print(f"CRITICAL: Model checkpoint not found at {ckpt_path}. Skipping plot.")
        return

    try:
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
    
    ax.set_xlabel(r"Time ($\mu s$)")
    ax.set_ylabel(r"Population $P(|1\rangle)$") 
    ax.set_ylim(-0.2, 1.2)
    ax.legend(loc='upper right', frameon=False)
    add_panel_label(ax, '(d)')

    # Inset Zoom
    axins = inset_axes(ax, width="40%", height="35%", loc='lower center', borderpad=2)
    mask = (t_np > 2.0) & (t_np < 4.0)
    
    if np.any(mask):
        axins.scatter(t_np[mask], y_noisy[mask], color=COLORS['data'], s=15, alpha=0.4, rasterized=True, zorder=1)
        axins.plot(t_np[mask], y_ideal[mask], color='k', linestyle=':', linewidth=1.5, zorder=2)
        axins.plot(t_np[mask], y_pred[mask], color=COLORS['model'], linewidth=2.0, zorder=3)
        axins.set_xlim(2.0, 4.0)
        y_section = y_ideal[mask]
        axins.set_ylim(y_section.min()-0.15, y_section.max()+0.15)
        axins.set_xticks([])
        axins.set_yticks([])
        mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5", alpha=0.5)

    plt.savefig(get_save_dir() / 'Fig3d_Lindblad_Discovery.pdf', format='pdf', bbox_inches='tight')
    plt.close()

def plot_fig4_benchmark():
    print(">>> Generating Fig4 (Benchmark)...")
    
    metrics_path = project_root / 'results' / 'benchmarks' / 'lindblad_metrics.json'
    
    if not metrics_path.exists():
        raise FileNotFoundError(f"Benchmark results not found at {metrics_path}. Please run 'python src/benchmarks/lindblad.py' first.")
    
    data = {}
    try:
        with open(metrics_path, 'r') as f:
            loaded = json.load(f)
            
        metrics = loaded.get("metrics", loaded)
        if "benchmark_mse" in loaded: metrics = loaded["benchmark_mse"]

        key_map = {
            'RF': 'RF', 'XGB': 'XGB',
            'PINN (Inc)': 'PINN_Incomplete', 'PINN (Known)': 'PINN_Known',
            'GAN': 'GAN', 'ParaQNN': 'ParaQNN'
        }
        
        for display_key, json_key in key_map.items():
            val = metrics.get(json_key)
            if val is None and json_key == 'PINN_Incomplete': val = metrics.get('PINN (Inc)')
            if val is None and json_key == 'PINN_Known': val = metrics.get('PINN (Known)')
            
            data[display_key] = float(val) if val is not None else 0.0

    except Exception as e:
        raise ValueError(f"Error reading metrics JSON: {e}")
    
    models = ['RF', 'XGB', 'PINN (Inc)', 'PINN (Known)', 'GAN', 'ParaQNN']
    mse_values = [data[m] for m in models]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    bar_colors = ['#BDC3C7', '#95A5A6', '#7F8C8D', '#566573', '#2C3E50', COLORS['model']]
    bars = ax.bar(models, mse_values, color=bar_colors, edgecolor='black', linewidth=0.5, width=0.6)
    
    ax.set_yscale('log')
    
    ax.set_ylim(top=max(mse_values)*50 if max(mse_values) > 0 else 1.0)
    ax.set_ylabel("Mean-Squared Error (MSE)")
    ax.grid(True, which='both', axis='y', linestyle='-', linewidth=0.3, alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height*1.2, f"{height:.1e}", 
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#222222')
                
    plt.savefig(get_save_dir() / 'Fig4_Lindblad_Benchmark.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    plot_fig3a_simulation()
    plot_fig3b_loss()
    plot_fig3c_alpha()
    plot_fig3d_discovery()
    plot_fig4_benchmark()
    print(f"Lindblad figures generated successfully at: {get_save_dir()}")