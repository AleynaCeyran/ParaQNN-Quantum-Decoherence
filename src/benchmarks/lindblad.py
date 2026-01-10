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

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.paraqnn import ParaQNN
from src.benchmarks.baselines import BaselineFactory, train_pinn, train_gan

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


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
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
    data_path = project_root / "data" / "synthetic" / "lindblad_data.npz"
    if not data_path.exists():
        logger.error(
            "Dataset missing: %s. Generate it via src/simulations/lindblad_simulation.py",
            data_path,
        )
        return

    logger.info("Loading dataset: %s", data_path)
    archive = np.load(data_path, allow_pickle=True)

    t = archive["time"] if "time" in archive else archive["t"]
    y_noisy = archive["signal_noisy"] if "signal_noisy" in archive else archive["signal"]
    y_ideal = archive["signal_ideal"] if "signal_ideal" in archive else archive["ideal"]

    split_idx = int(len(t) * 0.8)

    X_train_np = t[:split_idx].reshape(-1, 1)
    X_test_np = t[split_idx:].reshape(-1, 1)

    y_train_np = y_ideal[:split_idx]
    y_test_ideal_np = y_ideal[split_idx:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    set_reproducibility(seed, device)
    logger.info("Device: %s | Seed: %d", device, seed)

    t_train_th = torch.tensor(X_train_np, dtype=torch.float32, device=device).requires_grad_(True)
    y_train_th = torch.tensor(y_train_np, dtype=torch.float32, device=device).unsqueeze(1)
    t_test_th = torch.tensor(X_test_np, dtype=torch.float32, device=device)

    results: dict[str, float | None] = {}

    logger.info("Training classical baselines (RF, XGB)")
    rf = BaselineFactory.train_classical("RF", X_train_np, y_train_np)
    pred_rf = rf.predict(X_test_np)
    results["RF"] = mean_squared_error(y_test_ideal_np, pred_rf)

    xgb_model = BaselineFactory.train_classical("XGB", X_train_np, y_train_np)
    pred_xgb = xgb_model.predict(X_test_np)
    results["XGB"] = mean_squared_error(y_test_ideal_np, pred_xgb)

    logger.info("Training deep baselines (PINN, GAN)")

    pinn_inc = train_pinn(
        t_train_th,
        y_train_th,
        lindblad_pinn_loss_incomplete,
        epochs=2000,
        device=device,
    )
    pinn_inc.eval()
    with torch.no_grad():
        pred_inc = pinn_inc(t_test_th).cpu().numpy().flatten()
    results["PINN_Incomplete"] = mean_squared_error(y_test_ideal_np, pred_inc)

    pinn_known = train_pinn(
        t_train_th,
        y_train_th,
        lindblad_pinn_loss_known,
        epochs=2000,
        device=device,
    )
    pinn_known.eval()
    with torch.no_grad():
        pred_known = pinn_known(t_test_th).cpu().numpy().flatten()
    results["PINN_Known"] = mean_squared_error(y_test_ideal_np, pred_known)

    gan_gen = train_gan(t_train_th, y_train_th, epochs=2000, device=device)
    gan_gen.eval()
    with torch.no_grad():
        pred_gan = gan_gen(t_test_th).cpu().numpy().flatten()
    results["GAN"] = mean_squared_error(y_test_ideal_np, pred_gan)

    logger.info("Evaluating ParaQNN")
    config_path = project_root / "configs" / "lindblad.yaml"
    ckpt_path = project_root / "checkpoints" / "lindblad" / "best_model.pth"

    if not ckpt_path.exists():
        logger.warning("ParaQNN checkpoint not found: %s", ckpt_path)
        results["ParaQNN"] = None
    else:
        try:
            cfg = load_config(config_path)
            m_cfg = cfg["model"]

            paraqnn = ParaQNN(
                input_dim=int(m_cfg["input_dim"]),
                hidden_dim=int(m_cfg.get("neurons_per_layer", 64)),
                output_dim=int(m_cfg["output_dim"]),
                num_layers=int(m_cfg.get("hidden_layers", 2)),
                initial_alpha=float(m_cfg.get("initial_alpha", 5.0)),
                sharpness_k=float(m_cfg.get("sharpness_k", 1.0)),
            ).to(device)

            paraqnn.load_state_dict(torch.load(ckpt_path, map_location=device))
            paraqnn.eval()

            with torch.no_grad():
                pred_para, _ = paraqnn(t_test_th)
                pred_para = pred_para.cpu().numpy().flatten()

            results["ParaQNN"] = mean_squared_error(y_test_ideal_np, pred_para)
        except Exception as e:
            logger.error("ParaQNN evaluation failed: %s", e)
            results["ParaQNN"] = None

    print("\n" + "=" * 50)
    print(f"{'MODEL':<20} | {'MSE (Extrapolation)':<20}")
    print("=" * 50)
    for model, mse in results.items():
        val = f"{mse:.2e}" if mse is not None else "N/A"
        print(f"{model:<20} | {val}")
    print("=" * 50)

    save_path = project_root / "results" / "benchmarks" / "lindblad_metrics.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        clean_results = {k: (float(v) if v is not None else None) for k, v in results.items()}
        json.dump(clean_results, f, indent=4)

    logger.info("Metrics saved: %s", save_path)


if __name__ == "__main__":
    run_benchmark()

