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
LOGGER = logging.getLogger("mixed_regime_benchmark")


def mixed_pinn_loss_incomplete(u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    du_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    residual = du_dt + 0.1 * u
    return torch.mean(residual**2)


def mixed_pinn_loss_static_prior(u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    du_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    d2u_dt2 = torch.autograd.grad(du_dt, t, torch.ones_like(du_dt), create_graph=True)[0]
    gamma = 0.25
    omega_sq = (2 * np.pi * 3.0)**2
    residual = d2u_dt2 + (2.0 * gamma * du_dt) + (omega_sq * u)
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
        raise FileNotFoundError(f"Config missing: {cfg_path}")

    cfg = load_config(cfg_path)
    
    seeds = cfg.get("experiment", {}).get("seeds", [42])
    if isinstance(seeds, int): seeds = [seeds]
    
    epochs_deep = int(cfg.get("training", {}).get("epochs", 4000))

    data_path = project_root / "data" / "synthetic" / "mixed_regime_data.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset missing: {data_path}")

    archive = np.load(data_path, allow_pickle=True)
    t = archive["t"] if "t" in archive.files else archive["time"]
    y_ideal = archive["ideal"] if "ideal" in archive.files else archive["signal_ideal"]

    split_idx = int(len(t) * 0.8)
    X_train_np = t[:split_idx].reshape(-1, 1)
    X_test_np = t[split_idx:].reshape(-1, 1)
    y_train_np = y_ideal[:split_idx]
    y_test_np = y_ideal[split_idx:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s | Seeds: %s", device, seeds)

    metrics_buffer = defaultdict(list)

    for seed in seeds:
        set_reproducibility(seed, device)
        
        t_train_th = torch.tensor(X_train_np, dtype=torch.float32, device=device).requires_grad_(True)
        y_train_th = torch.tensor(y_train_np, dtype=torch.float32, device=device).unsqueeze(1)
        t_test_th = torch.tensor(X_test_np, dtype=torch.float32, device=device)

        # 1. Classical Baselines
        rf = BaselineFactory.train_classical("RF", X_train_np, y_train_np, seed=seed)
        metrics_buffer["RF"].append(mean_squared_error(y_test_np, rf.predict(X_test_np)))

        xgb_model = BaselineFactory.train_classical("XGB", X_train_np, y_train_np, seed=seed)
        metrics_buffer["XGB"].append(mean_squared_error(y_test_np, xgb_model.predict(X_test_np)))

        # 2. Deep Baselines
        pinn_inc = train_pinn(t_train_th, y_train_th, mixed_pinn_loss_incomplete, epochs=epochs_deep, device=device, physics_weight=0.1)
        pinn_inc.eval()
        with torch.no_grad():
            pred_inc = pinn_inc(t_test_th).cpu().numpy().flatten()
        metrics_buffer["PINN_Incomplete"].append(mean_squared_error(y_test_np, pred_inc))

        pinn_known = train_pinn(t_train_th, y_train_th, mixed_pinn_loss_static_prior, epochs=epochs_deep, device=device, physics_weight=0.1)
        pinn_known.eval()
        with torch.no_grad():
            pred_known = pinn_known(t_test_th).cpu().numpy().flatten()
        metrics_buffer["PINN_Known"].append(mean_squared_error(y_test_np, pred_known))

        gan_gen = train_gan(t_train_th, y_train_th, epochs=epochs_deep, device=device)
        gan_gen.eval()
        with torch.no_grad():
            pred_gan = gan_gen(t_test_th).cpu().numpy().flatten()
        metrics_buffer["GAN"].append(mean_squared_error(y_test_np, pred_gan))

        # 3. ParaQNN
        m_cfg = cfg.get("model", {})
        ckpt_dir_name = cfg.get("checkpointing", {}).get("save_dir", "checkpoints/mixed")
        base_dir = project_root / ckpt_dir_name

        ckpt_path_seed = base_dir / f"seed_{seed}" / "best_model.pth"
        ckpt_path_gen = base_dir / "best_model.pth"
        target_ckpt = ckpt_path_seed if ckpt_path_seed.exists() else ckpt_path_gen

        if target_ckpt.exists():
            paraqnn = ParaQNN(
                input_dim=int(m_cfg.get("input_dim", 1)),
                hidden_dim=int(m_cfg.get("neurons_per_layer", 128)),
                output_dim=int(m_cfg.get("output_dim", 1)),
                num_layers=int(m_cfg.get("hidden_layers", 3)),
                initial_alpha=float(m_cfg.get("initial_alpha", 5.0)),
                sharpness_k=float(m_cfg.get("sharpness_k", 1.0)),
            ).to(device)

            paraqnn.load_state_dict(torch.load(target_ckpt, map_location=device))
            paraqnn.eval()
            with torch.no_grad():
                pred_para, _ = paraqnn(t_test_th)
                pred_para = pred_para.cpu().numpy().flatten()
            metrics_buffer["ParaQNN"].append(mean_squared_error(y_test_np, pred_para))
        else:
            LOGGER.warning("ParaQNN checkpoint missing for seed %d", seed)
            metrics_buffer["ParaQNN"].append(None)

    # --- Aggregation ---
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
            # Flattened keys for backward compatibility
            key = model.replace("PINN_", "PINN_") 
            benchmark_mse_mean[key] = mu
            print(f"{model:<20} | {mu:.2e}        | {std:.2e}")
        else:
            print(f"{model:<20} | N/A             | N/A")
    print("=" * 65)

    out_dir = project_root / "results" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mixed_metrics.json"

    # Fix for JSON Serialization of NumPy types
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
        "protocol": "extrapolation",
        "seeds": seeds,
        "epochs_deep": epochs_deep,
        "metrics": clean_metrics, 
        "statistics": convert_stats(final_stats) 
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    LOGGER.info("Metrics saved: %s", out_path)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mixed_regime.yaml", help="Path to config file")
    parser.add_argument("--log-level", type=str, default="INFO")
    
    args, _ = parser.parse_known_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_benchmark()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
