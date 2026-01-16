"""
Lindblad Regime Benchmarking Script

Compares ParaQNN against baselines (RF, XGB, PINN, GAN) on the Lindblad open-system dataset.
All models are trained to predict the ideal signal y_ideal(t) and evaluated by MSE on
the extrapolation split (last 20% of the trajectory).
"""

import sys
import json
import logging
import math
import numpy as np
import torch
import yaml
from pathlib import Path
from sklearn.metrics import mean_squared_error
from collections import defaultdict

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.paraqnn import ParaQNN
from src.benchmarks.baselines import BaselineFactory, train_pinn, train_gan
from src.utils.common import set_reproducibility, load_yaml as load_config
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def lindblad_pinn_loss_incomplete(u, t):
    """
    Incomplete physics knowledge: Assumes simple first-order decay without oscillation/driving details.
    model: du/dt = -gamma * u (with generic gamma=0.1)
    """
    du_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    residual = du_dt + 0.1 * u
    return torch.mean(residual**2)


def run_benchmark() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/lindblad.yaml")
    args, _ = parser.parse_known_args()

    config_path = project_root / args.config
    if not config_path.exists():
        config_path = Path(args.config)

    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        return

    cfg = load_config(config_path)
    seeds = cfg.get("experiment", {}).get("seeds", [42])
    if isinstance(seeds, int):
        seeds = [seeds]
        
    logger.info("Starting benchmark across %d seeds: %s", len(seeds), seeds)

    # Data Loading 
    data_path = project_root / cfg.get("data", {}).get("path", "data/synthetic/lindblad_data.npz")
    if not data_path.exists():
        data_path = project_root / "data" / "synthetic" / "lindblad_data.npz"
    
    if not data_path.exists():
        logger.error("Dataset not found: %s", data_path)
        return

    logger.info("Loading dataset: %s", data_path)
    data = np.load(data_path, allow_pickle=True)
    
    # Extract metadata safely
    if "metadata" in data:
        meta = data["metadata"].item()
    elif "meta" in data:
        meta = data["meta"].item()
    else:
        logger.warning("No metadata found in NPZ. Using config values as fallback.")
        meta = {"physics": cfg.get("physics", {})}

    # Extract Physics Parameters 
    phys_meta = meta.get("physics", meta) 
    
    # Correct key mapping based on npz check
    t1_relax = float(phys_meta.get("t1_relaxation_us", 10.0))
    rabi_freq = float(phys_meta.get("rabi_frequency_rad_per_us", 2.0))
    
    logger.info(f"PINN Knowledge Configured -> Rabi Freq: {rabi_freq}, T1: {t1_relax}")

    # Prepare Physics Constants for PINN
    gamma = (1.0 / t1_relax) if t1_relax > 0 else 0.0
    omega_sq = rabi_freq ** 2

    # Define Known Physics Loss Dynamically 
    def lindblad_pinn_loss_known(u, t):
        du_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        d2u_dt2 = torch.autograd.grad(du_dt, t, torch.ones_like(du_dt), create_graph=True)[0]
        residual = d2u_dt2 + (2.0 * gamma * du_dt) + (omega_sq * u)
        return torch.mean(residual**2)

    # Prepare Data Splits
    t_arr = data["time"] if "time" in data else data["t"]
    y_ideal = data["signal_ideal"] if "signal_ideal" in data else data["ideal"]
    
    split_idx = int(len(t_arr) * 0.8)
    X_train_np = t_arr[:split_idx].reshape(-1, 1)
    X_test_np = t_arr[split_idx:].reshape(-1, 1)
    y_train_np = y_ideal[:split_idx]
    y_test_ideal_np = y_ideal[split_idx:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_buffer = defaultdict(list)

    # Training Loop 
    for seed in seeds:
        logger.info(">>> Running Seed: %d", seed)
        set_reproducibility(seed, device)

        t_train = torch.tensor(X_train_np, dtype=torch.float32, device=device).requires_grad_(True)
        y_train_th = torch.tensor(y_train_np, dtype=torch.float32, device=device).unsqueeze(1)
        t_test = torch.tensor(X_test_np, dtype=torch.float32, device=device)

        # 1. Classical Baselines
        rf = BaselineFactory.train_classical("RF", X_train_np, y_train_np, seed=seed)
        metrics_buffer["RF"].append(mean_squared_error(y_test_ideal_np, rf.predict(X_test_np)))

        xgb_model = BaselineFactory.train_classical("XGB", X_train_np, y_train_np, seed=seed)
        metrics_buffer["XGB"].append(mean_squared_error(y_test_ideal_np, xgb_model.predict(X_test_np)))

        # 2. Deep Baselines
        epochs = int(cfg.get("training", {}).get("epochs", 2000))
        
        pinn_inc = train_pinn(t_train, y_train_th, lindblad_pinn_loss_incomplete, epochs=epochs, device=device, physics_weight=0.1)
        pinn_inc.eval()
        with torch.no_grad():
            pred_inc = pinn_inc(t_test).cpu().numpy().flatten()
        metrics_buffer["PINN_Incomplete"].append(mean_squared_error(y_test_ideal_np, pred_inc))

        pinn_known = train_pinn(t_train, y_train_th, lindblad_pinn_loss_known, epochs=epochs, device=device, physics_weight=0.1)
        pinn_known.eval()
        with torch.no_grad():
            pred_known = pinn_known(t_test).cpu().numpy().flatten()
        metrics_buffer["PINN_Known"].append(mean_squared_error(y_test_ideal_np, pred_known))

        gan_gen = train_gan(t_train, y_train_th, epochs=epochs, device=device)
        gan_gen.eval()
        with torch.no_grad():
            pred_gan = gan_gen(t_test).cpu().numpy().flatten()
        metrics_buffer["GAN"].append(mean_squared_error(y_test_ideal_np, pred_gan))

        # 3. ParaQNN
        ckpt_dir_name = cfg.get("checkpointing", {}).get("save_dir", "checkpoints/lindblad")
        base_dir = project_root / ckpt_dir_name
        ckpt_path_seed = base_dir / f"seed_{seed}" / "best_model.pth"
        target_ckpt = ckpt_path_seed if ckpt_path_seed.exists() else base_dir / "best_model.pth"
        
        if target_ckpt.exists():
            model_cfg = cfg.get("model", {})
            paraqnn = ParaQNN(
                input_dim=int(model_cfg.get("input_dim", 1)),
                hidden_dim=int(model_cfg.get("neurons_per_layer", 128)),
                output_dim=int(model_cfg.get("output_dim", 1)),
                num_layers=int(model_cfg.get("hidden_layers", 2)),
                initial_alpha=float(model_cfg.get("initial_alpha", 5.0)),
                sharpness_k=float(model_cfg.get("sharpness_k", 1.0)),
            ).to(device)

            try:
                paraqnn.load_state_dict(torch.load(target_ckpt, map_location=device))
                paraqnn.eval()
                with torch.no_grad():
                    pred_para, _ = paraqnn(t_test)
                    pred_para = pred_para.cpu().numpy().flatten()
                metrics_buffer["ParaQNN"].append(mean_squared_error(y_test_ideal_np, pred_para))
            except RuntimeError as e:
                logger.error("ParaQNN Checkpoint error: %s", e)
                metrics_buffer["ParaQNN"].append(None)
        else:
            logger.warning("ParaQNN checkpoint not found")
            metrics_buffer["ParaQNN"].append(None)

    # Reporting 
    final_stats = {}
    benchmark_mse_mean = {}
    print("\n" + "=" * 65)
    print(f"{'MODEL':<20} | {'MSE (Mean)':<15} | {'Std Dev':<15}")
    print("=" * 65)

    for model, values in metrics_buffer.items():
        valid_values = [v for v in values if v is not None]
        if valid_values:
            mu = np.mean(valid_values)
            std = np.std(valid_values)
            final_stats[model] = {"mean": mu, "std": std, "values": valid_values}
            benchmark_mse_mean[model] = mu
            print(f"{model:<20} | {mu:.2e}        | {std:.2e}")
        else:
            final_stats[model] = None
            benchmark_mse_mean[model] = None
            print(f"{model:<20} | N/A             | N/A")
    print("=" * 65)

    save_path = project_root / "results" / "benchmarks" / "lindblad_metrics.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_mse": benchmark_mse_mean,
            "statistics": final_stats,
            "seeds_used": seeds,
            "noise_regime": {
                "t1_relaxation_us": t1_relax,
                "rabi_freq_rad_per_us": rabi_freq,
                "source": "npz_metadata" if "metadata" in data or "meta" in data else "config_fallback"
            }
        }, f, indent=4)
    
    logger.info("Benchmark metrics saved to: %s", save_path)


if __name__ == "__main__":
    run_benchmark()
    

