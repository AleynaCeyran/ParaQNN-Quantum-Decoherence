"""
Mixed regime simulation script.

Generates data for a time-dependent scenario with multiple regimes (Drive, Decay, Probe),
including non-Markovian memory effects and complex noise (Pink noise, RTN, SPAM).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from src.utils.common import find_project_root, load_yaml, deep_get
from src.utils.data import make_train_test_labels, save_npz
from src.utils.physics import PauliBasis, pink_noise, telegraph_process, clip_soft

LOGGER = logging.getLogger("mixed_regime_simulation")
_MISSING = object()


@dataclass(frozen=True)
class MixedConfig:
    seed: int
    data_relpath: str
    test_split: float
    shuffle: bool

    time_span_us: float
    total_samples: int
    switch_t1_us: float
    switch_t2_us: float
    drive_freq_MHz: float
    detuning_MHz: float
    t1_relaxation_us: float
    t2_dephasing_us: float

    pink_exponent: float
    pink_std: float
    rtn_amplitude: float
    rtn_prob: float
    rtn_model: str
    spam_error: float
    spam_model: str


def parse_config(yaml_obj: Dict[str, Any]) -> MixedConfig:
    def req(path: Tuple[str, ...], default: Any) -> Any:
        val = deep_get(yaml_obj, path, _MISSING)
        if val is _MISSING:
            LOGGER.warning("Missing config key; using default %s=%r", ".".join(path), default)
            return default
        return val

    seed = int(req(("experiment", "seed"), 42))

    data_relpath = str(req(("data", "path"), "data/synthetic/mixed_regime_data.npz"))
    test_split = float(req(("data", "test_split"), 0.2))
    shuffle = bool(req(("data", "shuffle"), True))

    time_span = float(req(("physics", "time_span"), 10.0))
    total_samples = int(req(("physics", "total_samples"), 50_000))
    switch_t1 = float(req(("physics", "switch_t1"), 4.0))
    switch_t2 = float(req(("physics", "switch_t2"), 7.0))

    drive_freq = float(req(("physics", "drive_freq"), 3.0))
    detuning = float(req(("physics", "detuning"), 0.0))
    t1_us = float(req(("physics", "t1_relaxation"), 6.0))
    t2_us = float(req(("physics", "t2_dephasing"), 4.0))

    pink_exp = float(req(("noise", "pink_noise_exponent"), 1.0))
    pink_std = float(req(("noise", "pink_noise_std"), 0.06))
    rtn_amp = float(req(("noise", "rtn_amplitude"), 0.04))
    rtn_prob = float(req(("noise", "rtn_prob"), 0.005))
    rtn_model = str(req(("noise", "rtn_model"), "flip_prob_per_step"))
    spam_error = float(req(("noise", "spam_error"), 0.02))
    spam_model = str(req(("noise", "spam_model"), "affine_population"))

    if not (0.0 < test_split < 1.0):
        raise ValueError(f"data.test_split must be in (0,1), got {test_split}")
    if total_samples < 200:
        raise ValueError(f"physics.total_samples too small, got {total_samples}")
    if time_span <= 0.0:
        raise ValueError(f"physics.time_span must be > 0, got {time_span}")
    if not (0.0 < switch_t1 < switch_t2 < time_span):
        raise ValueError(f"Regime switches must satisfy 0 < t1 < t2 < time_span; got {switch_t1}, {switch_t2}, {time_span}")
    if t1_us <= 0.0 or t2_us <= 0.0:
        raise ValueError(f"T1/T2 must be > 0; got T1={t1_us}, T2={t2_us}")
    if pink_std < 0.0 or rtn_amp < 0.0 or spam_error < 0.0:
        raise ValueError("Noise amplitudes must be non-negative.")
    if not (0.0 <= rtn_prob <= 1.0):
        raise ValueError(f"noise.rtn_prob must be in [0,1] as flip probability per step; got {rtn_prob}")
    if rtn_model != "flip_prob_per_step":
        raise ValueError(f"noise.rtn_model must be 'flip_prob_per_step' for this generator; got {rtn_model}")
    if spam_model != "affine_population":
        raise ValueError(f"noise.spam_model must be 'affine_population' for this generator; got {spam_model}")

    return MixedConfig(
        seed=seed,
        data_relpath=data_relpath,
        test_split=test_split,
        shuffle=shuffle,
        time_span_us=time_span,
        total_samples=total_samples,
        switch_t1_us=switch_t1,
        switch_t2_us=switch_t2,
        drive_freq_MHz=drive_freq,
        detuning_MHz=detuning,
        t1_relaxation_us=t1_us,
        t2_dephasing_us=t2_us,
        pink_exponent=pink_exp,
        pink_std=pink_std,
        rtn_amplitude=rtn_amp,
        rtn_prob=rtn_prob,
        rtn_model=rtn_model,
        spam_error=spam_error,
        spam_model=spam_model,
    )


class MixedRegimeSolver:
    def __init__(self, cfg: MixedConfig):
        self.cfg = cfg
        self.gamma1 = 1.0 / cfg.t1_relaxation_us
        self.gamma2_base = 1.0 / cfg.t2_dephasing_us

    def omega_t(self, t_us: float) -> float:
        if t_us < self.cfg.switch_t1_us:
            return self.cfg.drive_freq_MHz
        if t_us < self.cfg.switch_t2_us:
            return 0.0
        return 0.15 * self.cfg.drive_freq_MHz

    def liouvillian(self, t_us: float, rho_vec: np.ndarray) -> np.ndarray:
        rho = rho_vec.reshape((2, 2))

        omega_rad = 2 * np.pi * self.omega_t(t_us)
        detuning_rad = 2 * np.pi * self.cfg.detuning_MHz
        
        # H = 0.5 * Delta * Z + 0.5 * Omega * X
        H = 0.5 * detuning_rad * PauliBasis.SZ + 0.5 * omega_rad * PauliBasis.SX

        comm = -1j * (H @ rho - rho @ H)

        # Time-dependent non-Markovian memory in dephasing rate
        memory = (1.0 + 0.4 * np.sin(2.5 * t_us) + 0.1 * np.cos(5.1 * t_us))
        gamma2_t = self.gamma2_base * memory

        def dissipator(L: np.ndarray, gamma: float) -> np.ndarray:
            Ld = L.conj().T
            return gamma * (L @ rho @ Ld - 0.5 * (Ld @ L @ rho + rho @ Ld @ L))

        d1 = dissipator(PauliBasis.SM, self.gamma1)
        d2 = dissipator(PauliBasis.SZ, 0.5 * gamma2_t)

        return (comm + d1 + d2).reshape(-1)

    def simulate(self) -> Tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0.0, self.cfg.time_span_us, self.cfg.total_samples, dtype=np.float64)
        rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)

        sol = solve_ivp(
            fun=self.liouvillian,
            t_span=(float(t[0]), float(t[-1])),
            y0=rho0.reshape(-1),
            t_eval=t,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
        )
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")

        # Population in |1> is rho[1,1] -> index 3
        ideal = np.real(sol.y[3, :]).astype(np.float64)
        if not np.all(np.isfinite(ideal)):
            raise ValueError("Ideal signal contains non-finite values.")
        return t, ideal


def build_meta(cfg: MixedConfig, yaml_path: Path, dt_us: float) -> Dict[str, Any]:
    return {
        "yaml_path": str(yaml_path),
        "seed": int(cfg.seed),
        "units": {"t": "microseconds", "signal": "population"},
        "sampling": {
            "grid": "uniform",
            "time_span_us": float(cfg.time_span_us),
            "n_points": int(cfg.total_samples),
            "dt_us": float(dt_us),
            "dt_definition": "dt_us = time_span_us / (n_points - 1)",
        },
        "physics": {
            "switch_t1_us": float(cfg.switch_t1_us),
            "switch_t2_us": float(cfg.switch_t2_us),
            "drive_freq_MHz": float(cfg.drive_freq_MHz),
            "detuning_MHz": float(cfg.detuning_MHz),
            "t1_relaxation_us": float(cfg.t1_relaxation_us),
            "t2_dephasing_us": float(cfg.t2_dephasing_us),
        },
        "noise": {
            "signal_scale": "population",
            "pink_noise_exponent": float(cfg.pink_exponent),
            "pink_noise_std": float(cfg.pink_std),
            "rtn_model": cfg.rtn_model,
            "rtn_amplitude": float(cfg.rtn_amplitude),
            "rtn_flip_prob_per_step": float(cfg.rtn_prob),
            "spam_model": cfg.spam_model,
            "spam_error": float(cfg.spam_error),
        },
        "labels": {"type": "train_test_mask", "encoding": {"0": "train", "1": "test"}},
        "noise_injection_order": "signal = clip[-0.05,1.05]((1-2e)*ideal + e + pink + rtn)",
        "generator": "src/simulations/mixed_regime_simulation.py",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    project_root = find_project_root(Path(__file__).resolve())
    default_cfg = project_root / "configs" / "mixed_regime.yaml"
    yaml_path = Path(args.config).expanduser().resolve() if args.config else default_cfg.resolve()

    LOGGER.info("Project root: %s", project_root)
    LOGGER.info("Using config: %s", yaml_path)

    yaml_obj = load_yaml(yaml_path)
    cfg = parse_config(yaml_obj)

    solver = MixedRegimeSolver(cfg)
    t, ideal = solver.simulate()
    dt_us = float(t[1] - t[0])

    rng = np.random.default_rng(cfg.seed)
    pink = pink_noise(rng, cfg.total_samples, cfg.pink_exponent, cfg.pink_std)
    rtn = telegraph_process(rng, cfg.total_samples, cfg.rtn_amplitude, cfg.rtn_prob)

    e = float(cfg.spam_error)
    # SPAM (State Preparation And Measurement) error typically modeled as affine map
    # P_meas = (1 - 2*epsilon) * P_ideal + epsilon
    signal = (1.0 - 2.0 * e) * ideal + e
    signal = clip_soft(signal + pink + rtn)

    labels = make_train_test_labels(rng, cfg.total_samples, cfg.test_split, cfg.shuffle)
    meta = build_meta(cfg, yaml_path=yaml_path, dt_us=dt_us)

    out_path = save_npz(
        project_root,
        cfg.data_relpath,
        {
            "t": t.astype(np.float32),
            "ideal": ideal.astype(np.float32),
            "signal": signal.astype(np.float32),
            "labels": labels,
            "dt_us": np.float32(dt_us),
            "regime_switches": np.array([cfg.switch_t1_us, cfg.switch_t2_us], dtype=np.float32),
            "meta": np.array(meta, dtype=object),
        },
    )

    LOGGER.info("Dataset saved: %s", out_path)
    LOGGER.info("Keys: %s", sorted(["t", "ideal", "signal", "labels", "dt_us", "regime_switches", "meta"]))
    LOGGER.info(
        "Meta: seed=%s, N=%s, dt_us=%.6g, pink_std=%s, rtn_amp=%s, rtn_p=%s, spam_e=%s",
        meta.get("seed"),
        cfg.total_samples,
        dt_us,
        meta.get("noise", {}).get("pink_noise_std"),
        meta.get("noise", {}).get("rtn_amplitude"),
        meta.get("noise", {}).get("rtn_flip_prob_per_step"),
        meta.get("noise", {}).get("spam_error"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
