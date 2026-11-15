"""Determinism (rule 6: every random operation reads from a logged seed)."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch, and pin cuDNN to deterministic kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # matmul precision is a determinism knob too on Ampere and later
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def worker_init_fn(worker_id: int) -> None:
    """Give every dataloader worker a distinct, reproducible stream.

    Without this, forked workers share the parent's numpy state and every
    worker draws the *same* augmentations.
    """
    import torch
    base = torch.initial_seed() % 2 ** 31
    np.random.seed((base + worker_id) % 2 ** 31)
    random.seed(base + worker_id)


def derived_rng(*parts) -> np.random.Generator:
    """A generator keyed on a tuple of ints, independent of call order."""
    return np.random.default_rng([int(p) for p in parts])
