"""Shared plumbing for the metrics package (Phase 2, Step 12).

Every metric here obeys the `00_conventions.md` contract: it takes
`(pred[B,H,W] bool, target[B,H,W] bool)`, takes an optional `valid_mask` of
the same shape, and returns **per-image** values of shape `[B]`. Nothing is
reduced to a scalar; aggregation happens once, in `aggregate.py`.

`valid_mask` is the Phase 1 analysis region (r <= 0.95 R). Excluding it is not
cosmetic: off-disk pixels are background in both prediction and target, so
counting them inflates background IoU towards 1 and makes every arm look
better than it is. The 45% of each frame that lies outside the analysis region
would dominate the background class outright.

Empty-mask conventions follow the Phase 0 audit (`common/data/audit.pair_metrics`)
so model scores are directly comparable to the inter-annotator ceiling:

* target and prediction both empty  -> 1.0 (perfect agreement)
* exactly one of them empty         -> 0.0

The 71 filament-free images make this convention load-bearing rather than a
corner case, and `aggregate.py` reports both-empty counts so the two readings
(with and without them) stay available.
"""

from __future__ import annotations

import numpy as np

try:  # torch is the contract's type, but the metrics work on numpy too
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only on a torch-less host
    torch = None
    _HAS_TORCH = False

BOTH_EMPTY = 1.0
ONE_EMPTY = 0.0


def is_tensor(x) -> bool:
    return _HAS_TORCH and isinstance(x, torch.Tensor)


def to_numpy_bool(x) -> np.ndarray:
    """[B,H,W] (or [H,W]) of anything -> bool ndarray, batch dimension added."""
    if is_tensor(x):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3:
        raise ValueError(f"expected [B,H,W] or [H,W], got shape {a.shape}")
    return a.astype(bool)


def prepare(pred, target, valid_mask=None):
    """Validate shapes and flatten to [B, N] bool, with valid applied.

    Returns (pred, target, valid) where invalid pixels are already cleared from
    pred and target, plus the per-image valid pixel count.
    """
    p = to_numpy_bool(pred)
    t = to_numpy_bool(target)
    if p.shape != t.shape:
        raise ValueError(f"pred {p.shape} and target {t.shape} differ")
    if valid_mask is None:
        v = np.ones_like(p)
    else:
        v = to_numpy_bool(valid_mask)
        if v.shape[-2:] != p.shape[-2:]:
            raise ValueError(f"valid_mask {v.shape} does not match {p.shape}")
        if v.shape[0] == 1 and p.shape[0] > 1:
            v = np.broadcast_to(v, p.shape)
        if v.shape != p.shape:
            raise ValueError(f"valid_mask {v.shape} does not match {p.shape}")
    b = p.shape[0]
    return (np.logical_and(p, v).reshape(b, -1),
            np.logical_and(t, v).reshape(b, -1),
            v.reshape(b, -1))


def like(values: np.ndarray, reference):
    """Return [B] as a torch Tensor when the caller passed tensors, else ndarray."""
    out = np.asarray(values, dtype=np.float64)
    if is_tensor(reference):
        return torch.as_tensor(out, dtype=torch.float32,
                               device=reference.device)
    return out


def ratio(numerator: np.ndarray, denominator: np.ndarray,
          empty_value: float) -> np.ndarray:
    """Elementwise n/d, taking `empty_value` where the denominator is zero."""
    n = np.asarray(numerator, dtype=np.float64)
    d = np.asarray(denominator, dtype=np.float64)
    return np.where(d > 0, n / np.where(d > 0, d, 1.0), empty_value)
