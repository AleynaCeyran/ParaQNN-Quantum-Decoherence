"""
Common utilities for the ParaQNN project.
"""

import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import yaml

LOGGER = logging.getLogger(__name__)

_MISSING = object()


def find_project_root(start: Path) -> Path:
    """
    Resolve a stable project root by walking upward until 'src' is found.

    Args:
        start: The starting path (usually __file__).

    Returns:
        The path to the project root.
    """
    cur = start.resolve()
    for _ in range(12):
        if (cur / "src").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # Fallback to 2 levels up if strictly following src/simulations structure
    return start.resolve().parents[2]


def load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        A dictionary containing the configuration.

    Raises:
        RuntimeError: If PyYAML is missing.
        FileNotFoundError: If the file does not exist.
        ValueError: If the YAML root is not a dictionary.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}

    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping/dict, got {type(obj)}.")
    return obj


def deep_get(d: Dict[str, Any], keys: Tuple[str, ...], default: Any = None) -> Any:
    """
    Retrieve a value from a nested dictionary safely.

    Args:
        d: The dictionary to search.
        keys: A tuple of keys representing the path.
        default: The value to return if the key is not found.

    Returns:
        The value at the path or the default value.
    """
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def set_reproducibility(seed: int, device: Union[str, torch.device] = "cpu") -> None:
    """
    Set seeds for random number generators to ensure reproducibility.

    Args:
        seed: The seed value.
        device: The PyTorch device ('cpu' or 'cuda').
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
