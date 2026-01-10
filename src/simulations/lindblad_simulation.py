"""
Lindblad regime simulation script.

Generates synthetic data by solving the Lindblad master equation for a qubit,
then adding Gaussian and Telegraph noise.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
from scipy.integrate import solve_ivp

from src.utils.common import find_project_root, load_yaml, deep_get
from src.utils.data import make_train_test_labels, save_npz
from src.utils.physics import PauliBasis, telegraph_process, clip01

LOGGER = logging.getLogger("lindblad_simulation")


@dataclass(frozen=True)
class LindbladConfig:
    seed: int
    data_relpath: str
    test_split: float
    shuffle: bool

    # physics (units: microseconds for time; rates in 1/us; drive params in rad/us)
    t1_us: float
    t2_us: float
    time_span_us: float
    total_samples: int
    rabi_freq_rad_per_us: float
    detuning_rad_per_us: float

    # noise
    gaussian_std: float
    telegraph_amplitude: float
    telegraph_switching_rate: float  # flip probability per step


def parse_config(yaml_obj: Dict[str, Any]) -> LindbladConfig:
    """Parse YAML config into a validated LindbladConfig."""
    _MISSING = object()

    def req(path: Tuple[str, ...], default: Any) -> Any:
        val = deep_get(yaml_obj, path, _MISSING)
        if val is _MISSING:
            LOGGER.warning("Config key missing; using default %s=%r", ".".join(path), default)
            return default
        return val

    seed = int(req(("experiment", "seed"), 42))
    data_relpath = str(req(("data", "path"), "data/synthetic/lindblad_data.npz"))
    test_split = float(req(("data", "test_split"), 0.2))
    shuffle = bool(req(("data", "shuffle"), True))

    t1_us = float(req(("physics", "t1_relaxation"), 10.0))
    t2_us = float(req(("physics", "t2_dephasing"), 8.0))
    time_span_us = float(req(("physics", "time_span"), 5.0))
    total_samples = int(req(("physics", "total_samples"), 25_000))
    rabi_freq = float(req(("physics", "rabi_frequency"), 2.0))  # rad/us
    detuning = float(req(("physics", "detuning"), 0.0))         # rad/us

    gaussian_std = float(req(("noise", "gaussian_std"), 0.08))
    tele_amp = float(req(("noise", "telegraph_amplitude"), 0.0))
    tele_rate = float(req(("noise", "telegraph_switching_rate"), 0.0))

    if not (0.0 < test_split < 1.0):
        raise ValueError(f"data.test_split must be in (0,1), got {test_split}")
    if total_samples < 100:
        raise ValueError(f"physics.total_samples too small: {total_samples}")
    if time_span_us <= 0:
        raise ValueError(f"physics.time_span must be >0, got {time_span_us}")
    if t1_us <= 0 or t2_us <= 0:
        raise ValueError(f"T1/T2 must be >0, got T1={t1_us}, T2={t2_us}")
    if gaussian_std < 0 or tele_amp < 0:
        raise ValueError("Noise amplitudes must be non-negative.")
    if not (0.0 <= tele_rate <= 1.0):
        raise ValueError(
            f"noise.telegraph_switching_rate must be in [0,1] as flip prob/step, got {tele_rate}"
        )

    return LindbladConfig(
        seed=seed,
        data_relpath=data_relpath,
        test_split=test_split,
        shuffle=shuffle,
        t1_us=t1_us,
        t2_us=t2_us,
        time_span_us=time_span_us,
        total_samples=total_samples,
        rabi_freq_rad_per_us=rabi_freq,
        detuning_rad_per_us=detuning,
        gaussian_std=gaussian_std,
        telegraph_amplitude=tele_amp,
        telegraph_switching_rate=tele_rate,
    )


# Physics: Lindblad solver
def build_jump_operators(t1_us: float, t2_us: float) -> List[np.ndarray]:
    """
    Construct jump operators for amplitude damping (T1) and pure dephasing (T2).
    """
    gamma1 = 1.0 / t1_us
    
    # Pure dephasing rate gamma_phi = 1/T2 - 1/(2*T1).
    # Here, we treat t2_us strictly as the dephasing timescale parameter.
    gamma2 = 1.0 / t2_us
    
    c_ops: List[np.ndarray] = []

    if gamma1 > 0:
        c_ops.append(np.sqrt(gamma1) * PauliBasis.Sm)

    if gamma2 > 0:
        c_ops.append(np.sqrt(gamma2 * 0.5) * PauliBasis.Z)

    return c_ops


def lindblad_rhs(H: np.ndarray, c_ops: List[np.ndarray], t: float, rho_vec: np.ndarray) -> np.ndarray:
    """Compute d(rho)/dt for the Lindblad master equation."""
    # pylint: disable=unused-argument
    rho = rho_vec.reshape((2, 2))

    # Von Neumann term: -i [H, rho]
    d_rho = -1j * (H @ rho - rho @ H)

    # Dissipators
    for L in c_ops:
        Ld = L.conj().T
        # L rho Ldag - 0.5 {Ldag L, rho}
        d_rho += (L @ rho @ Ld) - 0.5 * (Ld @ L @ rho + rho @ Ld @ L)

    return d_rho.reshape(-1)


def simulate_population(cfg: LindbladConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the Lindblad master equation."""
    t = np.linspace(0.0, cfg.time_span_us, cfg.total_samples, dtype=np.float64)

    # Hamiltonian H = 0.5 * delta * Z + 0.5 * Omega * X
    H = 0.5 * cfg.detuning_rad_per_us * PauliBasis.Z + 0.5 * cfg.rabi_freq_rad_per_us * PauliBasis.X
    c_ops = build_jump_operators(cfg.t1_us, cfg.t2_us)

    # Initial state initialization: |0><0| (Ground State)
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)

    sol = solve_ivp(
        fun=lambda tt, yy: lindblad_rhs(H, c_ops, tt, yy),
        t_span=(t[0], t[-1]),
        y0=rho0.reshape(-1),
        t_eval=t,
        method="RK45",
        rtol=1e-9,
        atol=1e-11,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver did not converge: {sol.message}")

    # Extract excited state population P(|1>) = rho[1,1]
    # Since solve_ivp returns flattened density matrices (rho00, rho01, rho10, rho11),
    # we extract the last component (index 3).
    rho_11 = sol.y[3, :]
    pop = np.real(rho_11).astype(np.float64)
    return t, pop


def generate(cfg: LindbladConfig, yaml_path: Path) -> Dict[str, np.ndarray]:
    """Generate dataset."""
    rng = np.random.default_rng(cfg.seed)

    t, ideal = simulate_population(cfg)

    gaussian = rng.normal(loc=0.0, scale=cfg.gaussian_std, size=cfg.total_samples).astype(np.float64)
    rtn = telegraph_process(
        rng=rng,
        n=cfg.total_samples,
        amp=cfg.telegraph_amplitude,
        flip_prob=cfg.telegraph_switching_rate,
    )
    signal = clip01(ideal + gaussian + rtn)

    labels = make_train_test_labels(rng=rng, n=cfg.total_samples, test_split=cfg.test_split, shuffle=cfg.shuffle)

    for name, arr in [("t", t), ("ideal", ideal), ("signal", signal)]:
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1D; got shape={arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")

    meta: Dict[str, Any] = {
        "yaml_path": str(yaml_path),
        "seed": int(cfg.seed),
        "units": {"t": "microseconds", "rabi_frequency": "rad/us", "detuning": "rad/us"},
        "physics": {
            "t1_relaxation_us": float(cfg.t1_us),
            "t2_dephasing_us": float(cfg.t2_us),
            "time_span_us": float(cfg.time_span_us),
            "total_samples": int(cfg.total_samples),
            "rabi_frequency_rad_per_us": float(cfg.rabi_freq_rad_per_us),
            "detuning_rad_per_us": float(cfg.detuning_rad_per_us),
        },
        "noise": {
            "gaussian_std": float(cfg.gaussian_std),
            "telegraph_amplitude": float(cfg.telegraph_amplitude),
            "telegraph_switching_rate": float(cfg.telegraph_switching_rate),
            "telegraph_rate_interpretation": "flip probability per time step",
        },
        "labels": {"type": "train_test_mask", "encoding": {"0": "train", "1": "test"}},
        "noise_injection_order": "signal = clip01(ideal + gaussian + telegraph)",
        "generator": "src/simulations/lindblad_simulation.py",
    }

    return {
        "t": t,
        "ideal": ideal,
        "signal": signal,
        "labels": labels,
        "meta": np.array(meta, dtype=object),
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

    this_file = Path(__file__).resolve()
    project_root = find_project_root(this_file)
    default_cfg = project_root / "configs" / "lindblad.yaml"
    yaml_path = Path(args.config).resolve() if args.config else default_cfg

    LOGGER.info("Project root: %s", project_root)
    LOGGER.info("Using config: %s", yaml_path)

    yaml_obj = load_yaml(yaml_path)
    cfg = parse_config(yaml_obj)
    data = generate(cfg, yaml_path=yaml_path)
    out_path = save_npz(project_root, cfg.data_relpath, data)

    meta = data["meta"].item()
    LOGGER.info("Dataset saved: %s", out_path)
    LOGGER.info("Keys: %s", sorted(list(data.keys())))
    LOGGER.info(
        "Meta: seed=%s, T1_us=%s, T2_us=%s, sigma=%s, tele_amp=%s, tele_p=%s",
        meta.get("seed"),
        meta.get("physics", {}).get("t1_relaxation_us"),
        meta.get("physics", {}).get("t2_dephasing_us"),
        meta.get("noise", {}).get("gaussian_std"),
        meta.get("noise", {}).get("telegraph_amplitude"),
        meta.get("noise", {}).get("telegraph_switching_rate"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
