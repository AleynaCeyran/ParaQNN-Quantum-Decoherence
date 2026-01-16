"""
Physics and noise utilities for ParaQNN simulations.
"""

from typing import Optional
import numpy as np


class PauliBasis:
    """
    Pauli matrices and standard quantum operators.
    """
    # Standard Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    # Ladder operators
    Sm = np.array([[0, 0], [1, 0]], dtype=np.complex128) # Sigma minus (lowering)
    Sp = np.array([[0, 1], [0, 0]], dtype=np.complex128) # Sigma plus (raising)

    # Aliases to match mixed regime notation
    SX = X
    SY = Y
    SZ = Z
    SM = Sm # 0.5 * (SX - 1j * SY) is exactly Sm


def telegraph_process(
    rng: np.random.Generator,
    n: int,
    amp: float,
    flip_prob: float
) -> np.ndarray:
    """
    Generate Random Telegraph Noise (RTN).

    Args:
        rng: NumPy random generator.
        n: Number of samples.
        amp: Amplitude of the noise (signals jumps between +amp and -amp).
        flip_prob: Probability of flipping state at each step.

    Returns:
        Array of telegraph noise.
    """
    x = np.empty(n, dtype=np.float64)
    state = 1.0 if rng.random() < 0.5 else -1.0
    for i in range(n):
        if rng.random() < flip_prob:
            state *= -1.0
        x[i] = state * amp
    return x


def pink_noise(
    rng: np.random.Generator,
    n: int,
    exponent: float,
    std: float
) -> np.ndarray:
    """
    Generate Pink Noise (1/f noise).

    Args:
        rng: NumPy random generator.
        n: Number of samples.
        exponent: The power spectral density exponent (S(f) ~ 1/f^exponent).
                  1.0 corresponds to pink noise.
        std: Standard deviation of the resulting noise.

    Returns:
        Array of pink noise.
    """
    white = rng.normal(0.0, 1.0, n).astype(np.float64)
    w_fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)

    # Avoid division by zero at f=0
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / (freqs[1:] ** (exponent / 2.0))

    x = np.fft.irfft(w_fft * scale, n=n)

    # Normalize
    s = float(np.std(x))
    if s > 0:
        x = x / s

    return x.astype(np.float64) * float(std)


def clip01(x: np.ndarray) -> np.ndarray:
    """Clip values to [0, 1]."""
    return np.clip(x, 0.0, 1.0)


def clip_soft(x: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """Clip values to [-margin, 1+margin]."""
    return np.clip(x, -margin, 1.0 + margin)
