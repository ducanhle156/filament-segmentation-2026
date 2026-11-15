"""Flip / rotation test-time augmentation for full-disk inference.

A filament has no preferred orientation on the disk, so the same network
is run on mirrored / rotated copies of the disk and the probability maps
are mapped back to the original frame and averaged. Views are the dihedral
group D4: the first four are the flips (identity, left-right, up-down,
both), the last four the same after a 90-degree rotation.
"""
from __future__ import annotations

import numpy as np

VIEWS = [(0, False, False), (0, False, True), (0, True, False), (0, True, True),
         (1, False, False), (1, False, True), (1, True, False), (1, True, True)]   # (rot90 k, flip ud, flip lr)


def forward_view(a: np.ndarray, v) -> np.ndarray:
    """Apply view `v` to the last two axes."""
    k, ud, lr = v
    a = np.rot90(a, k, axes=(-2, -1))
    if ud:
        a = a[..., ::-1, :]
    if lr:
        a = a[..., ::-1]
    return np.ascontiguousarray(a)


def inverse_view(a: np.ndarray, v) -> np.ndarray:
    """Undo view `v` (inverse_view(forward_view(a, v), v) == a)."""
    k, ud, lr = v
    if lr:
        a = a[..., ::-1]
    if ud:
        a = a[..., ::-1, :]
    return np.ascontiguousarray(np.rot90(a, -k, axes=(-2, -1)))


def predict_views(model, inp: np.ndarray, valid: np.ndarray, tile: int, overlap: float, device,
                  amp: bool = True, n_views: int = 8) -> list[np.ndarray]:
    """One full-disk probability map per view, each mapped back to the original frame."""
    from common.engine.stitch import predict_full_disk
    out = []
    for v in VIEWS[:n_views]:
        p = predict_full_disk(model, forward_view(inp, v), forward_view(valid, v), tile, overlap, device, amp=amp)
        out.append(inverse_view(p, v))
    return out
