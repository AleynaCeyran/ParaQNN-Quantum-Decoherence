"""
Lindblad Regime Benchmark

Compares ParaQNN against classical (RF, XGB) and deep baselines (PINN, GAN)
on the Lindblad open-system dataset. All models are trained to predict the
ideal signal y_ideal(t) and evaluated on an extrapolation split (last 20%).
"""

import sys
import json
import yaml
import torch
import numpy as np
import logging
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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("lindblad_benchmark")


def lindblad_pinn_loss_incomplete(u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    du_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    residual = du_dt + 0.1 * u
    return torch.mean(residual**2)


def lindblad_pinn_loss_known(u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    du_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    d2u_dt2 = torch.autograd.grad(du_dt, t, torch.ones_like(du_dt), create_graph=True)[0]
    zeta = 0.1
    w0 = 9.0
    residual = d2u_dt2 + (2.0 * zeta * w0 * du_dt) + ((w0**2) * u)
    return torch.mean(residual**2)


def run_benchmark() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/lindblad.yaml")
    args, _ = parser.parse_known_args()

    data_path = project_root / "data" / "synthetic" / "lindblad_data.npz"
    config_path = project_root / args.config

    if not config_path.exists():
        config_path = Path(args.config)

    if not data_path.exists():
        logger.error("Dataset missing: %s", data_path)
        return

    cfg = load_config(config_path)
    seeds = cfg.get("experiment", {}).get("seeds", [42])
    if isinstance(seeds, int): seeds = [seeds]

    logger.info("Loading dataset: %s", data_path)
    archive = np.load(data_path, allow_pickle=True)

    t = archive["time"] if "time" in archive else archive["t"]
    y_ideal = archive["signal_ideal"] if "signal_ideal" in archive else archive["ideal"]

    split_idx = int(len(t) * 0.8)
    X_train_np = t[:split_idx].reshape(-1, 1)
    X_test_np = t[split_idx:].reshape(-1, 1)
    y_train_np = y_ideal[:split_idx]
    y_test_ideal_np = y_ideal[split_idx:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    metrics_buffer = defaultdict(list)
    epochs = int(cfg.get("training", {}).get("epochs", 2000))

    logger.info("Starting benchmark loop over seeds: %s", seeds)

    for seed in seeds:
        set_reproducibility(seed, device)
        
        t_train_th = torch.tensor(X_train_np, dtype=torch.float32, device=device).requires_grad_(True)
        y_train_th = torch.tensor(y_train_np, dtype=torch.float32, device=device).unsqueeze(1)
        t_test_th = torch.tensor(X_test_np, dtype=torch.float32, device=device)

        # 1. Classical
        rf = BaselineFactory.train_classical("RF", X_train_np, y_train_np, seed=seed)
        metrics_buffer["RF"].append(mean_squared_error(y_test_ideal_np, rf.predict(X_test_np)))

        xgb_mod = BaselineFactory.train_classical("XGB", X_train_np, y_train_np, seed=seed)
        metrics_buffer["XGB"].append(mean_squared_error(y_test_ideal_np, xgb_mod.predict(X_test_np)))

        # 2. Deep Baselines
        pinn_inc = train_pinn(t_train_th, y_train_th, lindblad_pinn_loss_incomplete, epochs=epochs, device=device, physics_weight=0.1)
        pinn_inc.eval()
        with torch.no_grad():
            pred_inc = pinn_inc(t_test_th).cpu().numpy().flatten()
        metrics_buffer["PINN_Incomplete"].append(mean_squared_error(y_test_ideal_np, pred_inc))

        pinn_known = train_pinn(t_train_th, y_train_th, lindblad_pinn_loss_known, epochs=epochs, device=device, physics_weight=0.1)
        pinn_known.eval()
        with torch.no_grad():
            pred_known = pinn_known(t_test_th).cpu().numpy().flatten()
        metrics_buffer["PINN_Known"].append(mean_squared_error(y_test_ideal_np, pred_known))

        gan_gen = train_gan(t_train_th, y_train_th, epochs=epochs, device=device)
        gan_gen.eval()
        with torch.no_grad():
            pred_gan = gan_gen(t_test_th).cpu().numpy().flatten()
        metrics_buffer["GAN"].append(mean_squared_error(y_test_ideal_np, pred_gan))

       # 3. ParaQNN
        m_cfg = cfg.get("model", {})
        ckpt_dir_name = cfg.get("checkpointing", {}).get("save_dir", "checkpoints/lindblad")
        base_dir = project_root / ckpt_dir_name

        ckpt_path_seed = base_dir / f"seed_{seed}" / "best_model.pth"
        ckpt_path_gen = base_dir / "best_model.pth"
        target_ckpt = ckpt_path_seed if ckpt_path_seed.exists() else ckpt_path_gen

        if target_ckpt.exists():
            
            paraqnn = ParaQNN(
                input_dim=int(m_cfg.get("input_dim", 1)),
                hidden_dim=int(m_cfg.get("neurons_per_layer", 64)),
                output_dim=int(m_cfg.get("output_dim", 1)),
                num_layers=int(m_cfg.get("hidden_layers", 2)),
                initial_alpha=float(m_cfg.get("initial_alpha", 5.0)),
                sharpness_k=float(m_cfg.get("sharpness_k", 1.0)),
            ).to(device)

            try:
                paraqnn.load_state_dict(torch.load(target_ckpt, map_location=device))
                paraqnn.eval()
                with torch.no_grad():
                    pred_para, _ = paraqnn(t_test_th)
                    pred_para = pred_para.cpu().numpy().flatten()
                
                metrics_buffer["ParaQNN"].append(mean_squared_error(y_test_ideal_np, pred_para))
                
            except Exception as e:
               
                logger.error("ParaQNN load failed for seed %d: %s", seed, e)
                metrics_buffer["ParaQNN"].append(None)
        else:
            
            logger.warning("ParaQNN checkpoint missing for seed %d", seed)
            metrics_buffer["ParaQNN"].append(None)

    # --- Reporting ---
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

    save_path = project_root / "results" / "benchmarks" / "lindblad_metrics.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            **benchmark_mse_mean, 
            "statistics": final_stats
        }, f, indent=4)

    logger.info("Metrics saved: %s", save_path)


if __name__ == "__main__":
    run_benchmark()
    

