"""
Rabi Regime Benchmarking Script

Compares ParaQNN against baselines (RF, XGB, PINN, GAN) on the damped Rabi dataset.
All models are trained to predict the ideal signal y_ideal(t) and evaluated by MSE on
the extrapolation split (last 20% of the trajectory).
"""

import sys
import json
import logging
import math
from pathlib import Path

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def rabi_pinn_loss_incomplete(u, t):
    du_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    residual = du_dt + 0.1 * u
    return torch.mean(residual**2)


def load_config(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_reproducibility(seed: int, device: torch.device) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_benchmark() -> None:
    data_path = project_root / "data" / "synthetic" / "rabi_oscillations.npz"
    if not data_path.exists():
        logger.error("Dataset not found: %s. Run simulation first.", data_path)
        return

    logger.info("Loading data: %s", data_path)
    data = np.load(data_path, allow_pickle=True)

    # META 
    meta = data["meta"].item()

    noise_report = {
        "T1_us": meta["physics"].get("relaxation_time_T1_us"),
        "T2_us": meta["physics"].get("coherence_time_T2_us"),
        "rabi_frequency_MHz": meta["physics"].get("rabi_frequency_MHz"),
        "gaussian_std": meta["noise"].get("gaussian_std"),
        "telegraph_amplitude": meta["noise"].get("telegraph_amplitude"),
        "telegraph_switching_rate": meta["noise"].get("telegraph_switching_rate"),
        "saturation_total": meta.get("metrics", {}).get("saturation_total"),
    }

    # PHYSICS CONSTANTS (from meta) 
    # Use dataset meta so 'PINN_Known' always matches the simulation settings.
    T2_us = float(meta["physics"].get("coherence_time_T2_us", 15.0))
    f_MHz = float(meta["physics"].get("rabi_frequency_MHz", 2.5))
    # Damped-oscillator surrogate: u'' + 2γ u' + ω^2 u = 0
    # Here t is in microseconds, f is in MHz (= 1/µs), so ω has units rad/µs.
    gamma = 1.0 / T2_us
    omega_sq = (2.0 * math.pi * f_MHz) ** 2

    def rabi_pinn_loss_known_meta(u, t):
        du_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        d2u_dt2 = torch.autograd.grad(du_dt, t, torch.ones_like(du_dt), create_graph=True)[0]
        residual = d2u_dt2 + (2.0 * gamma * du_dt) + (omega_sq * u)
        return torch.mean(residual**2)

    # DATA 
    X = data["t"]
    y_noisy = data["signal"]
    y_ideal = data["ideal"]

    split_idx = int(len(X) * 0.8)

    X_train_np = X[:split_idx].reshape(-1, 1)
    X_test_np = X[split_idx:].reshape(-1, 1)

    y_train_np = y_ideal[:split_idx]
    y_test_ideal_np = y_ideal[split_idx:]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    set_reproducibility(seed, device)
    logger.info("Device: %s | Seed: %d", device, seed)

    t_train = torch.tensor(X_train_np, dtype=torch.float32, device=device).requires_grad_(True)
    y_train_th = torch.tensor(y_train_np, dtype=torch.float32, device=device).unsqueeze(1)
    t_test = torch.tensor(X_test_np, dtype=torch.float32, device=device)

    results = {}

    logger.info("Running classical baselines (RF, XGB) ...")
    rf = BaselineFactory.train_classical("RF", X_train_np, y_train_np)
    pred_rf = rf.predict(X_test_np)
    results["RF"] = mean_squared_error(y_test_ideal_np, pred_rf)

    xgb_model = BaselineFactory.train_classical("XGB", X_train_np, y_train_np)
    pred_xgb = xgb_model.predict(X_test_np)
    results["XGB"] = mean_squared_error(y_test_ideal_np, pred_xgb)

    logger.info("Running deep baselines (PINN, GAN) ...")
    pinn_inc = train_pinn(t_train, y_train_th, rabi_pinn_loss_incomplete, epochs=1500, device=device)
    pinn_inc.eval()
    with torch.no_grad():
        pred_inc = pinn_inc(t_test).cpu().numpy().flatten()
    results["PINN_Incomplete"] = mean_squared_error(y_test_ideal_np, pred_inc)

    pinn_known = train_pinn(t_train, y_train_th, rabi_pinn_loss_known_meta, epochs=1500, device=device)
    pinn_known.eval()
    with torch.no_grad():
        pred_known = pinn_known(t_test).cpu().numpy().flatten()
    results["PINN_Known"] = mean_squared_error(y_test_ideal_np, pred_known)

    gan_gen = train_gan(t_train, y_train_th, epochs=1500, device=device)
    gan_gen.eval()
    with torch.no_grad():
        pred_gan = gan_gen(t_test).cpu().numpy().flatten()
    results["GAN"] = mean_squared_error(y_test_ideal_np, pred_gan)

    logger.info("Evaluating ParaQNN ...")
    config_path = project_root / "configs" / "rabi.yaml"
    ckpt_path = project_root / "checkpoints" / "rabi" / "best_model.pth"

    if ckpt_path.exists():
        cfg = load_config(config_path)
        model_cfg = cfg["model"]

        paraqnn = ParaQNN(
            input_dim=int(model_cfg["input_dim"]),
            hidden_dim=int(model_cfg.get("neurons_per_layer", 128)),
            output_dim=int(model_cfg["output_dim"]),
            num_layers=int(model_cfg.get("hidden_layers", 3)),
            initial_alpha=float(model_cfg.get("initial_alpha", 6.0)),
            sharpness_k=float(model_cfg.get("sharpness_k", 1.0)),
        ).to(device)

        try:
            paraqnn.load_state_dict(torch.load(ckpt_path, map_location=device))
            paraqnn.eval()
            with torch.no_grad():
                pred_para, _ = paraqnn(t_test)
                pred_para = pred_para.cpu().numpy().flatten()
            results["ParaQNN"] = mean_squared_error(y_test_ideal_np, pred_para)
        except RuntimeError as e:
            logger.error("Checkpoint mismatch: %s", e)
            logger.error("Ensure configs/rabi.yaml matches the training configuration.")
            results["ParaQNN"] = None
    else:
        logger.warning("ParaQNN checkpoint not found: %s", ckpt_path)
        results["ParaQNN"] = None

    print("\n" + "=" * 44)
    print("BENCHMARK RESULTS (MSE on extrapolation)")
    print("=" * 44)
    for model, mse in results.items():
        val = f"{mse:.2e}" if mse is not None else "N/A"
        print(f"{model:<20} | {val}")
    print("=" * 44)

    save_path = project_root / "results" / "benchmarks" / "rabi_metrics.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        clean_results = {k: (float(v) if v is not None else None) for k, v in results.items()}
        json.dump({"benchmark_mse": clean_results, "noise_regime": noise_report}, f, indent=4)
    logger.info("Metrics saved: %s", save_path)


if __name__ == "__main__":
    run_benchmark()

