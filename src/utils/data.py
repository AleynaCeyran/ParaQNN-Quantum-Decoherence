"""
Data manipulation utilities for ParaQNN.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Set

import numpy as np


def make_train_test_labels(
    rng: np.random.Generator,
    n: int,
    test_split: float,
    shuffle: bool
) -> np.ndarray:
    """
    Generate train/test split labels.

    Args:
        rng: NumPy random generator.
        n: Total number of samples.
        test_split: Fraction of samples to use for testing.
        shuffle: Whether to shuffle the data (random split) or not (extrapolation split).

    Returns:
        A boolean array or 0/1 integer array where 1 indicates test set.
        (Here returning uint8 0/1 to match existing conventions).
    """
    n_test = int(round(n * test_split))
    n_test = max(1, min(n - 1, n_test))

    labels = np.zeros(n, dtype=np.uint8)

    if shuffle:
        idx = np.arange(n)
        rng.shuffle(idx)
        labels[idx[:n_test]] = 1
    else:
        # Extrapolation: last segment is test
        labels[n - n_test :] = 1

    return labels


def save_npz(
    project_root: Path,
    data_relpath: str,
    data: Dict[str, Any],
    required_keys: Optional[Set[str]] = None
) -> Path:
    """
    Save data to a compressed NPZ file and verify keys.

    Args:
        project_root: The root directory of the project.
        data_relpath: Relative path to save the data.
        data: Dictionary of data to save.
        required_keys: Set of keys that must be present in the saved file.
                       Defaults to {"t", "signal", "ideal"}.

    Returns:
        The absolute path to the saved file.

    Raises:
        RuntimeError: If required keys are missing from the saved file.
    """
    if required_keys is None:
        required_keys = {"t", "signal", "ideal"}

    out_path = (project_root / data_relpath).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use save_compressed if available, or just savez
    np.savez_compressed(out_path, **data)

    # Verify
    chk = np.load(out_path, allow_pickle=True)
    missing = required_keys - set(chk.files)
    if missing:
        raise RuntimeError(
            f"Saved NPZ missing required keys: {sorted(missing)}; found={sorted(chk.files)}"
        )

    return out_path
