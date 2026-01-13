"""
Rabi regime simulation script.

Generates synthetic data for a damped Rabi oscillation with added Gaussian
and Telegraph noise.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from src.utils.common import find_project_root, load_yaml, deep_get
from src.utils.data import make_train_test_labels, save_npz
from src.utils.physics import telegraph_process, clip01

LOGGER = logging.getLogger("rabi_simulation")
_MISSING = object()


@dataclass(frozen=True)
class RabiConfig:
    seed: int
    data_relpath: str
    test_split: float
    shuffle: bool
    T2_us: float
    T1_us: float
    rabi_frequency_MHz: float
    time_span_us: float
    total_samples: int
    gaussian_std: float
    telegraph_amplitude: float
    telegraph_switching_rate: float  # flip probability per step
    clip_min: float
    clip_max: float
    report_metrics: Tuple[str, ...]


def parse_config(yaml_obj: Dict[str, Any]) -> RabiConfig:
    """Parse YAML config into a validated RabiConfig."""

    def req(path: Tuple[str, ...], default: Any) -> Any:
        val = deep_get(yaml_obj, path, _MISSING)
        if val is _MISSING:
            LOGGER.warning("Missing config key; using default %s=%r", ".".join(path), default)
            return default
        return val

    seed = int(req(("experiment", "seed"), 42))

    data_relpath = str(req(("data", "path"), "data/synthetic/rabi_oscillations.npz"))
    test_split = float(req(("data", "test_split"), 0.2))
    shuffle = bool(req(("data", "shuffle"), True))

    T2_us = float(req(("physics", "coherence_time"), 15.0))
    T1_us = float(req(("physics", "T1"), 1e9))  # optional; very large => effectively no T1 damping
    rabi_freq = float(req(("physics", "rabi_frequency"), 2.5))
    time_span = float(req(("physics", "time_span"), 10.0))
    total_samples = int(req(("physics", "total_samples"), 10_000))

    gaussian_std = float(req(("noise", "gaussian_std"), 0.08))
    tele_amp = float(req(("noise", "telegraph_amplitude"), 0.15))
    tele_rate = float(req(("noise", "telegraph_switching_rate"), 0.05))
    clip_min = float(req(("measurement", "clip_min"), 0.0))
    clip_max = float(req(("measurement", "clip_max"), 1.0))
    report_metrics_raw = req(("measurement", "report_metrics"), [])

    if report_metrics_raw is None:
        report_metrics_raw = []
    if isinstance(report_metrics_raw, str):
        report_metrics = (report_metrics_raw,)
    else:
        report_metrics = tuple(str(x) for x in report_metrics_raw)

    if clip_max <= clip_min:
        raise ValueError(f"measurement.clip_max must be > clip_min; got {clip_min}, {clip_max}")

    if not (0.0 < test_split < 1.0):
        raise ValueError(f"data.test_split must be in (0, 1); got {test_split}")
    if total_samples < 100:
        raise ValueError(f"physics.total_samples too small; got {total_samples}")
    if time_span <= 0:
        raise ValueError(f"physics.time_span must be > 0; got {time_span}")
    if T2_us <= 0:
        raise ValueError(f"physics.coherence_time (T2) must be > 0; got {T2_us}")
    if rabi_freq <= 0:
        raise ValueError(f"physics.rabi_frequency must be > 0; got {rabi_freq}")
    if T1_us <= 0:
        raise ValueError(f"physics.T1 must be > 0; got {T1_us}")
    if gaussian_std < 0 or tele_amp < 0:
        raise ValueError("Noise amplitudes must be non-negative.")
    if not (0.0 <= tele_rate <= 1.0):
        raise ValueError(
            "noise.telegraph_switching_rate must be a flip probability per step in [0, 1]."
        )

    return RabiConfig(
        seed=seed,
        data_relpath=data_relpath,
        test_split=test_split,
        shuffle=shuffle,
        T2_us=T2_us,
        T1_us=T1_us,
        rabi_frequency_MHz=rabi_freq,
        time_span_us=time_span,
        total_samples=total_samples,
        gaussian_std=gaussian_std,
        telegraph_amplitude=tele_amp,
        telegraph_switching_rate=tele_rate,
        clip_min=clip_min,
        clip_max=clip_max,
        report_metrics=report_metrics,
    )


def rabi_population_ideal(t_us: np.ndarray, rabi_frequency_MHz: float, T2_us: float, T1_us: float) -> np.ndarray:
    """
    Open-system-inspired Rabi population signal (heuristic but consistent):
        base(t) = 1/2 [1 - exp(-t/T2) cos(2π f t)]
    Add amplitude relaxation (T1) so the signal relaxes toward 1/2:
        y(t) = base(t) * exp(-t/T1) + 1/2*(1 - exp(-t/T1))
    This keeps y in [0,1] before noise/clipping, and matches y(0)=0.
    Units: f in MHz, t in microseconds.
    """
    omega = 2.0 * math.pi * float(rabi_frequency_MHz)
    env_T2 = np.exp(-t_us / float(T2_us))
    base = 0.5 * (1.0 - env_T2 * np.cos(omega * t_us))
    env_T1 = np.exp(-t_us / float(T1_us))
    return base * env_T1 + 0.5 * (1.0 - env_T1)


def generate(cfg: RabiConfig, yaml_path: Path) -> Dict[str, np.ndarray]:
    """Generate dataset arrays and provenance metadata."""
    rng = np.random.default_rng(cfg.seed)

    t = np.linspace(0.0, cfg.time_span_us, cfg.total_samples, dtype=np.float64)
    ideal = rabi_population_ideal(t, cfg.rabi_frequency_MHz, cfg.T2_us, cfg.T1_us)

    gaussian = rng.normal(0.0, cfg.gaussian_std, size=cfg.total_samples).astype(np.float64)
    rtn = telegraph_process(rng, cfg.total_samples, cfg.telegraph_amplitude, cfg.telegraph_switching_rate)

    signal = np.clip(ideal + gaussian + rtn, cfg.clip_min, cfg.clip_max)

    # Optional reported metrics
    eps = 1e-12
    metrics: Dict[str, Any] = {}
    if any(m.lower() == "saturation" for m in cfg.report_metrics):
        sat_low = float(np.mean(signal <= (cfg.clip_min + eps)))
        sat_high = float(np.mean(signal >= (cfg.clip_max - eps)))
        metrics["saturation_low"] = sat_low
        metrics["saturation_high"] = sat_high
        metrics["saturation_total"] = float(sat_low + sat_high)

    labels = make_train_test_labels(rng, cfg.total_samples, cfg.test_split, cfg.shuffle)

    for name, arr in (("t", t), ("ideal", ideal), ("signal", signal)):
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1D; got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")

    meta: Dict[str, Any] = {
        "yaml_path": str(yaml_path),
        "seed": int(cfg.seed),
        "units": {"t": "microseconds", "rabi_frequency": "MHz"},
        "physics": {
            "coherence_time_T2_us": float(cfg.T2_us),
            "relaxation_time_T1_us": float(cfg.T1_us),
            "rabi_frequency_MHz": float(cfg.rabi_frequency_MHz),
            "time_span_us": float(cfg.time_span_us),
            "total_samples": int(cfg.total_samples),
        },
        "noise": {
            "gaussian_std": float(cfg.gaussian_std),
            "telegraph_amplitude": float(cfg.telegraph_amplitude),
            "telegraph_switching_rate": float(cfg.telegraph_switching_rate),
            "telegraph_rate_interpretation": "flip probability per time step",
        },
        "labels": {
            "type": "train_test_mask",
            "encoding": {"0": "train", "1": "test"},
            "test_split": float(cfg.test_split),
            "shuffle": bool(cfg.shuffle),
            "split_strategy": "random" if bool(cfg.shuffle) else "extrapolation_last_segment",
        },
        "noise_injection_order": "signal = clip01(ideal + gaussian + telegraph)",
        "measurement": {
            "clip_min": float(cfg.clip_min),
            "clip_max": float(cfg.clip_max),
            "report_metrics": list(cfg.report_metrics),
        },
        "metrics": metrics,
        "generator": "src/simulations/rabi_simulation.py",
    }

    return {
        "t": t,
        "ideal": ideal,
        "signal": signal,
        "labels": labels,
        "meta": np.array(meta, dtype=object),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a damped-Rabi synthetic dataset (NPZ).")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML path (default: <project_root>/configs/rabi.yaml).",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG/INFO/WARNING/ERROR")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    project_root = find_project_root(Path(__file__).resolve())
    default_cfg = (project_root / "configs" / "rabi.yaml").resolve()
    yaml_path = Path(args.config).expanduser().resolve() if args.config else default_cfg

    LOGGER.info("Project root: %s", project_root)
    LOGGER.info("Using config: %s", yaml_path)

    yaml_obj = load_yaml(yaml_path)
    if not yaml_obj:
        raise ValueError(f"YAML is empty: {yaml_path}")

    cfg = parse_config(yaml_obj)
    data = generate(cfg, yaml_path=yaml_path)
    out_path = save_npz(project_root, cfg.data_relpath, data)

    LOGGER.info("Dataset saved: %s", out_path)
    LOGGER.info("Keys: %s", sorted(list(data.keys())))
    LOGGER.info(
        "Shapes: t=%s, ideal=%s, signal=%s, labels=%s",
        data["t"].shape,
        data["ideal"].shape,
        data["signal"].shape,
        data["labels"].shape,
    )

    meta = data["meta"].item()
    LOGGER.info(
        "Meta: seed=%s, T1_us=%s, T2_us=%s, f_MHz=%s, sigma=%s, tele_amp=%s, tele_p=%s, sat_total=%s",
        meta.get("seed"),
        meta.get("physics", {}).get("relaxation_time_T1_us"),
        meta.get("physics", {}).get("coherence_time_T2_us"),
        meta.get("physics", {}).get("rabi_frequency_MHz"),
        meta.get("noise", {}).get("gaussian_std"),
        meta.get("noise", {}).get("telegraph_amplitude"),
        meta.get("noise", {}).get("telegraph_switching_rate"),
        meta.get("metrics", {}).get("saturation_total"),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
