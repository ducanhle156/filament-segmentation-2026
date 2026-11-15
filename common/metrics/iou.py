"""Region-overlap metrics (Phase 2, Step 12).

All return per-image `[B]`. See `common.py` for the valid-mask and empty-mask
conventions.
"""

from __future__ import annotations

import numpy as np

from common.metrics.common import BOTH_EMPTY, like, prepare, ratio


def _iou(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    inter = np.logical_and(p, t).sum(1)
    union = np.logical_or(p, t).sum(1)
    return ratio(inter, union, BOTH_EMPTY)


def foreground_iou(pred, target, valid_mask=None):
    """|P and T| / |P or T| over valid pixels. Returns [B].

    Both empty -> 1.0; exactly one empty -> 0.0 (falls out of the ratio).
    """
    p, t, _ = prepare(pred, target, valid_mask)
    return like(_iou(p, t), pred)


def background_iou(pred, target, valid_mask=None):
    """Foreground IoU of the complement, computed inside the valid region."""
    p, t, v = prepare(pred, target, valid_mask)
    return like(_iou(np.logical_and(~p, v), np.logical_and(~t, v)), pred)


def mean_iou(pred, target, valid_mask=None):
    """(foreground IoU + background IoU) / 2, per image."""
    p, t, v = prepare(pred, target, valid_mask)
    fg = _iou(p, t)
    bg = _iou(np.logical_and(~p, v), np.logical_and(~t, v))
    return like((fg + bg) / 2.0, pred)


def dice(pred, target, valid_mask=None):
    """2|P and T| / (|P| + |T|) over valid pixels. Returns [B]."""
    p, t, _ = prepare(pred, target, valid_mask)
    inter = np.logical_and(p, t).sum(1)
    total = p.sum(1) + t.sum(1)
    return like(ratio(2 * inter, total, BOTH_EMPTY), pred)
