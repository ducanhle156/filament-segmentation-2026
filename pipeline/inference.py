"""Step 2 - probability map. The disk is cut into overlapping 512 px tiles,
every tile goes through the network, the tile outputs are averaged back
into one 2048 x 2048 map; this is repeated per test-time view and per fold
model, and all maps are averaged.
"""
from __future__ import annotations

import numpy as np

from . import settings
from .tta import predict_views


def predict_probability(models: list, cfg: dict, img: np.ndarray, valid: np.ndarray,
                        n_views: int = settings.N_VIEWS, overlap: float = settings.OVERLAP) -> np.ndarray:
    """Mean filament probability over `models` and `n_views` views; float32 [H, W], 0 outside the disk."""
    tile, amp = cfg["data"]["tile_size"], cfg["train"].get("amp", True)
    acc = np.zeros(img.shape, dtype=np.float64)
    for m in models:
        device = next(m.parameters()).device
        acc += np.mean(predict_views(m, img, valid, tile, overlap, device, amp, n_views), axis=0)
    return (acc / len(models)).astype(np.float32)
