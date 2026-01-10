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

LOGGER = logging.getLogger("mixed_regime_benchmark")


def set_reproducibility(seed: int, device: torch.device) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    cfg_path = project_root / "configs" / "mixed_regime.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config missing: {cfg_path}")

    cfg = load_yaml(cfg_path)
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    epochs_deep = int(cfg.get("training", {}).get("epochs", 4000))

    data_path = project_root / "data" / "synthetic" / "mixed_regime_data.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset missing: {data_path}. Run mixed_regime_simulation.py first.")

    archive = np.load(data_path, allow_pickle=True)

    t = archive["t"] if "t" in archive.files else archive["time"]
    y_ideal = archive["ideal"] if "ideal" in archive.files else archive["signal_ideal"]

    split_idx = int(len(t) * 0.8)
    X_train_np = t[:split_idx].reshape(-1, 1)
    X_test_np = t[split_idx:].reshape(-1, 1)

    y_train_np = y_ideal[:split_idx]
    y_test_np = y_ideal[split_idx:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_reproducibility(seed, device)
    LOGGER.info("Project root: %s", project_root)
    LOGGER.info("Using config: %s", cfg_path)
    LOGGER.info("Device: %s | Seed: %d", device, seed)
    LOGGER.info("Protocol: extrapolation (train first 80%%, test last 20%%)")

    t_train_th = torch.tensor(X_train_np, dtype=torch.float32, device=device).requires_grad_(True)
    y_train_th = torch.tensor(y_train_np, dtype=torch.float32, device=device).unsqueeze(1)
    t_test_th = torch.tensor(X_test_np, dtype=torch.float32, device=device)

    results: Dict[str, float | None] = {}

    rf = BaselineFactory.train_classical("RF", X_train_np, y_train_np)
    pred_rf = rf.predict(X_test_np)
    results["RF"] = mean_squared_error(y_test_np, pred_rf)

    xgb_model = BaselineFactory.train_classical("XGB", X_train_np, y_train_np)
    pred_xgb = xgb_model.predict(X_test_np)
    results["XGB"] = mean_squared_error(y_test_np, pred_xgb)

    pinn_inc = train_pinn(t_train_th, y_train_th, mixed_pinn_loss_incomplete, epochs=epochs_deep, device=device)
    pinn_inc.eval()
    with torch.no_grad():
        pred_inc = pinn_inc(t_test_th).cpu().numpy().flatten()
    results["PINN_Incomplete"] = mean_squared_error(y_test_np, pred_inc)

    pinn_known = train_pinn(t_train_th, y_train_th, mixed_pinn_loss_static_prior, epochs=epochs_deep, device=device)
    pinn_known.eval()
    with torch.no_grad():
        pred_known = pinn_known(t_test_th).cpu().numpy().flatten()
    results["PINN_Known"] = mean_squared_error(y_test_np, pred_known)

    gan_gen = train_gan(t_train_th, y_train_th, epochs=epochs_deep, device=device)
    gan_gen.eval()
    with torch.no_grad():
        pred_gan = gan_gen(t_test_th).cpu().numpy().flatten()
    results["GAN"] = mean_squared_error(y_test_np, pred_gan)

    ckpt_path = project_root / "checkpoints" / "mixed" / "best_model.pth"
    if ckpt_path.exists():
        m_cfg = cfg.get("model", {})
        paraqnn = ParaQNN(
            input_dim=int(m_cfg.get("input_dim", 1)),
            hidden_dim=int(m_cfg.get("neurons_per_layer", 128)),
            output_dim=int(m_cfg.get("output_dim", 1)),
            num_layers=int(m_cfg.get("hidden_layers", 3)),
            initial_alpha=float(m_cfg.get("initial_alpha", 5.0)),
            sharpness_k=float(m_cfg.get("sharpness_k", 1.0)),
        ).to(device)

        paraqnn.load_state_dict(torch.load(ckpt_path, map_location=device))
        paraqnn.eval()
        with torch.no_grad():
            pred_para, _ = paraqnn(t_test_th)
            pred_para = pred_para.cpu().numpy().flatten()
        results["ParaQNN"] = mean_squared_error(y_test_np, pred_para)
    else:
        results["ParaQNN"] = None
        LOGGER.warning("ParaQNN checkpoint not found: %s", ckpt_path)

    print("\n" + "=" * 50)
    print(f"{'MODEL':<20} | {'MSE (Extrapolation)'}")
    print("=" * 50)
    for model, mse in results.items():
        val = f"{mse:.2e}" if mse is not None else "N/A"
        print(f"{model:<20} | {val}")
    print("=" * 50)

    out_dir = project_root / "results" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mixed_metrics.json"

    payload = {
        "protocol": "extrapolation",
        "seed": seed,
        "epochs_deep": epochs_deep,
        "metrics": {k: (float(v) if v is not None else None) for k, v in results.items()},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    LOGGER.info("Metrics saved: %s", out_path)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_benchmark()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

