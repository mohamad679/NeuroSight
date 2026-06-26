"""Deterministic seeding helpers."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch is a core project dependency.
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SeedState:
    """Record the deterministic seed applied to runtime libraries."""

    seed: int
    deterministic_torch: bool


def set_global_seed(seed: int = 42, *, deterministic_torch: bool = True) -> SeedState:
    """Seed Python, NumPy, and Torch when available.

    Args:
        seed: Integer seed applied to all supported random generators.
        deterministic_torch: Whether to request deterministic Torch kernels.

    Returns:
        Seed state for logging or assertions.
    """
    normalized_seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(normalized_seed)
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)

    if torch is not None:
        torch.manual_seed(normalized_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(normalized_seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

    return SeedState(seed=normalized_seed, deterministic_torch=deterministic_torch)
