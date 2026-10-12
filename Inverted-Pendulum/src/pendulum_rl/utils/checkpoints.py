"""Checkpoint I/O with explicit overwrite protection and helpful error messages."""

from __future__ import annotations

import os
from pathlib import Path

import torch


class CheckpointError(RuntimeError):
    """Base class for checkpoint problems."""


class CheckpointNotFoundError(CheckpointError, FileNotFoundError):
    pass


class CheckpointExistsError(CheckpointError, FileExistsError):
    pass


def save_checkpoint(path: str | Path, payload: dict, overwrite: bool = False) -> Path:
    """Atomically write ``payload`` to ``path``.

    Refuses to replace an existing file unless ``overwrite=True`` so trained
    policies are never lost by accident.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise CheckpointExistsError(
            f"Refusing to overwrite existing checkpoint '{path}'. Pass overwrite=True "
            f"(CLI: --overwrite) or choose a different --run-name / path.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def list_checkpoints(root: str | Path = "checkpoints") -> list[Path]:
    root = Path(root)
    return sorted(root.rglob("*.pt")) if root.exists() else []


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    path = Path(path)
    if not path.is_file():
        found = list_checkpoints(path.parent if path.parent.exists() else "checkpoints")
        hint = ("Available checkpoints:\n  " + "\n  ".join(str(p) for p in found[:10])) if found else \
            "No checkpoints found yet. Train one first, e.g.: python scripts/train.py --mode stabilize --episodes 300"
        raise CheckpointNotFoundError(f"Checkpoint '{path}' does not exist.\n{hint}")
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:  # corrupted file, wrong format, ...
        raise CheckpointError(f"Could not read checkpoint '{path}': {exc}") from exc
