"""
Mixed Regime Benchmark

Evaluates ParaQNN and baselines on a non-stationary trajectory with regime switches.
All models are trained to predict the ideal signal y_ideal(t). Performance is reported
on an extrapolation split (train: first 80% of time, test: last 20%).
"""

from __future__ import annotations

import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict
from collections import defaultdict

import numpy as np
import torch
import yaml
from sklearn.metrics import mean_squared_error

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.paraqnn import ParaQNN
from src.benchmarks.baselines import BaselineFactory, train_pinn, train_gan
from src.utils.common import set_reproducibility, load_yaml as load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("mixed_regime_benchmark")


def mixed_pinn_loss_incomplete(u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Incomplete physics: Assumes simple first-order decay without knowing about
    driving fields or regime switches.
    """
    du_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    residual = du_dt + 0.1 * u
    return torch.mean(residual**2)


def run_benchmark() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mixed_regime.yaml")
    args, _ = parser.parse_known_args()
    
    cfg_path = project_root / args.config
    if not cfg_path.exists():
        cfg_path = Path(args.config)
    
    if not cfg_path.exists():
        LOGGER.error(f"Config missing: {cfg_path}")
        return

    cfg = load_config(cfg_path)
    seeds = cfg.get("experiment", {}).get("seeds", [42])
    if isinstance(seeds, int): seeds = [seeds]
    
    # Data Loading 
    data_path = project_root / cfg.get("data", {}).get("path", "data/synthetic/mixed_regime_data.npz")
    if not data_path.exists():
        data_path = project_root / "data" / "synthetic" / "mixed_regime_data.npz"

    if not data_path.exists():
        LOGGER.error(f"Dataset missing: {data_path}")
        return

    LOGGER.info(f"Loading dataset: {data_path}")
    archive = np.load(data_path, allow_pickle=True)
    
    # Extract Metadata
    if "metadata" in archive:
        meta = archive["metadata"].item()
    elif "meta" in archive:
        meta = archive["meta"].item()
    else:
        LOGGER.warning("No metadata in NPZ. Fallback to config.")
        meta = {"physics": cfg.get("physics", {})}

    # Extract Physics Parameters 
    phys_meta = meta.get("physics", meta)
    
    # Correct mapping based on npz check
    drive_freq = float(phys_meta.get("drive_freq_MHz", 3.0))
    t2_dephasing = float(phys_meta.get("t2_dephasing_us", 4.0))

    # Also capture regime switches for reporting
    regime_switches = list(archive["regime_switches"]) if "regime_switches" in archive else []

    LOGGER.info(f"PINN Knowledge Source -> Freq: {drive_freq} MHz, T2: {t2_dephasing} us")
    LOGGER.info(f"Regime Switches Detected: {regime_switches}")

    # Map to Damped Driven Oscillator (Static Prior)
    # gamma = 1/T2 (Dephasing rate)
    # omega = 2*pi*f
    pinn_gamma = 1.0 / t2_dephasing if t2_dephasing > 0 else 0.0
    pinn_omega = 2.0 * np.pi * drive_freq
    pinn_omega_sq = pinn_omega ** 2

    # Define Known Physics Loss Dynamically 
    def mixed_pinn_loss_static_prior(u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Represents 'Known' physics for the Drive Phase.
        Fails in mixed regime because it assumes parameters are constant (static prior)
        while the actual system switches regimes.
        """
        du_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        d2u_dt2 = torch.autograd.grad(du_dt, t, torch.ones_like(du_dt), create_graph=True)[0]
        residual = d2u_dt2 + (2.0 * pinn_gamma * du_dt) + (pinn_omega_sq * u)
        return torch.mean(residual**2)
  

    # Prepare Data Splits
    t_arr = archive["t"] if "t" in archive.files else archive["time"]
    y_ideal = archive["ideal"] if "ideal" in archive.files else archive["signal_ideal"]

    split_idx = int(len(t_arr) * 0.8)
    X_train_np = t_arr[:split_idx].reshape(-1, 1)
    X_test_np = t_arr[split_idx:].reshape(-1, 1)
    y_train_np = y_ideal[:split_idx]
    y_test_ideal_np = y_ideal[split_idx:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs_deep = int(cfg.get("training", {}).get("epochs", 4000))

    metrics_buffer = defaultdict(list)
    LOGGER.info("Starting benchmark loop over seeds: %s", seeds)

    for seed in seeds:
        set_reproducibility(seed, device)
        
        t_train_th = torch.tensor(X_train_np, dtype=torch.float32, device=device).requires_grad_(True)
        y_train_th = torch.tensor(y_train_np, dtype=torch.float32, device=device).unsqueeze(1)
        t_test_th = torch.tensor(X_test_np, dtype=torch.float32, device=device)

        # 1. Classical Baselines
        rf = BaselineFactory.train_classical("RF", X_train_np, y_train_np, seed=seed)
        metrics_buffer["RF"].append(mean_squared_error(y_test_ideal_np, rf.predict(X_test_np)))

        xgb_model = BaselineFactory.train_classical("XGB", X_train_np, y_train_np, seed=seed)
        metrics_buffer["XGB"].append(mean_squared_error(y_test_ideal_np, xgb_model.predict(X_test_np)))

        # 2. Deep Baselines
        pinn_inc = train_pinn(t_train_th, y_train_th, mixed_pinn_loss_incomplete, epochs=epochs_deep, device=device, physics_weight=0.1)
        pinn_inc.eval()
        with torch.no_grad():
            pred_inc = pinn_inc(t_test_th).cpu().numpy().flatten()
        metrics_buffer["PINN_Incomplete"].append(mean_squared_error(y_test_ideal_np, pred_inc))

        pinn_known = train_pinn(t_train_th, y_train_th, mixed_pinn_loss_static_prior, epochs=epochs_deep, device=device, physics_weight=0.1)
        pinn_known.eval()
        with torch.no_grad():
            pred_known = pinn_known(t_test_th).cpu().numpy().flatten()
        metrics_buffer["PINN_Known"].append(mean_squared_error(y_test_ideal_np, pred_known))

        gan_gen = train_gan(t_train_th, y_train_th, epochs=epochs_deep, device=device)
        gan_gen.eval()
        with torch.no_grad():
            pred_gan = gan_gen(t_test_th).cpu().numpy().flatten()
        metrics_buffer["GAN"].append(mean_squared_error(y_test_ideal_np, pred_gan))

        # 3. ParaQNN
        ckpt_dir_name = cfg.get("checkpointing", {}).get("save_dir", "checkpoints/mixed")
        base_dir = project_root / ckpt_dir_name
        ckpt_path_seed = base_dir / f"seed_{seed}" / "best_model.pth"
        target_ckpt = ckpt_path_seed if ckpt_path_seed.exists() else base_dir / "best_model.pth"

        if target_ckpt.exists():
            model_cfg = cfg.get("model", {})
            paraqnn = ParaQNN(
                input_dim=int(model_cfg.get("input_dim", 1)),
                hidden_dim=int(model_cfg.get("neurons_per_layer", 128)),
                output_dim=int(model_cfg.get("output_dim", 1)),
                num_layers=int(model_cfg.get("hidden_layers", 3)),
                initial_alpha=float(model_cfg.get("initial_alpha", 5.0)),
                sharpness_k=float(model_cfg.get("sharpness_k", 1.0)),
            ).to(device)

            try:
                paraqnn.load_state_dict(torch.load(target_ckpt, map_location=device))
                paraqnn.eval()
                with torch.no_grad():
                    pred_para, _ = paraqnn(t_test_th)
                    pred_para = pred_para.cpu().numpy().flatten()
                metrics_buffer["ParaQNN"].append(mean_squared_error(y_test_ideal_np, pred_para))
            except Exception as e:
                LOGGER.error("ParaQNN load failed for seed %d: %s", seed, e)
                metrics_buffer["ParaQNN"].append(None)
        else:
            LOGGER.warning("ParaQNN checkpoint missing for seed %d", seed)
            metrics_buffer["ParaQNN"].append(None)

    # Aggregation and Reporting 
    final_stats = {}
    benchmark_mse_mean = {}

    print("\n" + "=" * 65)
    print(f"{'MODEL':<20} | {'MSE (Mean)':<15} | {'Std Dev':<15}")
    print("=" * 65)
    for model, values in metrics_buffer.items():
        valid = [v for v in values if v is not None]
        if valid:
            mu, std = np.mean(valid), np.std(valid)
            final_stats[model] = {"mean": mu, "std": std, "values": valid}
            benchmark_mse_mean[model] = mu
            print(f"{model:<20} | {mu:.2e}        | {std:.2e}")
        else:
            final_stats[model] = None
            benchmark_mse_mean[model] = None
            print(f"{model:<20} | N/A             | N/A")
    print("=" * 65)

    out_dir = project_root / "results" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mixed_metrics.json"

    def convert_stats(stats_dict):
        clean = {}
        for k, v in stats_dict.items():
            if v is None:
                clean[k] = None
            else:
                clean[k] = {
                    "mean": float(v["mean"]),
                    "std": float(v["std"]),
                    "values": [float(x) for x in v["values"]]
                }
        return clean

    clean_metrics = {k: (float(v) if v is not None else None) for k, v in benchmark_mse_mean.items()}

    payload = {
        "benchmark_mse": clean_metrics,
        "statistics": convert_stats(final_stats),
        "seeds_used": seeds,
        "noise_regime": {
            "drive_freq_MHz": drive_freq,
            "t2_dephasing_us": t2_dephasing,
            "regime_switches": [float(x) for x in regime_switches],
            "source": "npz_metadata" if "metadata" in archive or "meta" in archive else "config_fallback"
        }
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    LOGGER.info("Metrics saved: %s", out_path)


def main() -> int:
    run_benchmark()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
