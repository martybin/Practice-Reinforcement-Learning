"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA).

    Fully bit-exact GPU determinism is not guaranteed; on CPU, runs with the same
    seed, config and library versions reproduce.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch
    except ImportError:  # physics/environment-only installs
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def derive_seed(base: int, *salt: int) -> int:
    """Deterministically derive an independent seed from ``base`` and integer salts."""
    seq = np.random.SeedSequence([int(base), *[int(s) for s in salt]])
    return int(seq.generate_state(1)[0])
