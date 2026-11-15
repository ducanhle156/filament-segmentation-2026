"""Centerline metrics (Phase 2, Step 12).

Filaments are thin, elongated structures roughly 10 px wide at 2048x2048, and
Phase 0 found that annotators agree on *where* a filament runs far more than
on *how wide* it is (clDice 0.66 and centerline recall 0.69 against an area
Dice of 0.58). Region overlap alone therefore understates how well a model has
found the structures; these metrics score the skeleton instead.

Skeletons are **morphological** (`skimage.morphology.skeletonize`), not the
differentiable soft-skeleton. That is deliberate: the Phase 0 inter-annotator
ceiling was computed with morphological skeletons, and a model's clDice is
only comparable to that ceiling if it is measured the same way. The
soft-skeleton belongs in the training loss, where differentiability is what
matters; see `soft_skeleton` in `cldice_metric.py`.
"""

from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize as _sk_skeletonize

from common.metrics.common import BOTH_EMPTY, like, prepare, ratio


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Morphological skeleton of a 2-D bool mask.

    This host's skimage build segfaults on bool memoryviews, so the input is
    fed as uint8 — the same wrapper Phase 0 used (`common/data/audit.py`).
    """
    return _sk_skeletonize(np.ascontiguousarray(mask, dtype=np.uint8)).astype(bool)


def skeletons(flat: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Per-image skeletons of a [B, H*W] bool array, returned flattened."""
    out = np.zeros_like(flat)
    for i in range(flat.shape[0]):
        img = flat[i].reshape(shape)
        if img.any():
            out[i] = skeletonize(img).reshape(-1)
    return out


def _shape_of(pred, valid_mask=None) -> tuple[int, int]:
    from common.metrics.common import to_numpy_bool
    return to_numpy_bool(pred).shape[-2:]


def centerline_recall(pred, target, valid_mask=None):
    """Fraction of the ground-truth skeleton covered by the prediction.

    |skel(T) and P| / |skel(T)| — "did the model find the filament, wherever
    it drew the edges". Target empty -> 1.0 if the prediction is empty too,
    else 0.0.
    """
    p, t, _ = prepare(pred, target, valid_mask)
    st = skeletons(t, _shape_of(pred, valid_mask))
    hit = np.logical_and(st, p).sum(1)
    n = st.sum(1)
    out = ratio(hit, n, BOTH_EMPTY)
    out = np.where((n == 0) & (p.sum(1) > 0), 0.0, out)
    return like(out, pred)


def centerline_precision(pred, target, valid_mask=None):
    """Fraction of the predicted skeleton lying inside the ground truth.

    |skel(P) and T| / |skel(P)| — "is what the model drew actually filament".
    """
    p, t, _ = prepare(pred, target, valid_mask)
    sp = skeletons(p, _shape_of(pred, valid_mask))
    hit = np.logical_and(sp, t).sum(1)
    n = sp.sum(1)
    out = ratio(hit, n, BOTH_EMPTY)
    out = np.where((n == 0) & (t.sum(1) > 0), 0.0, out)
    return like(out, pred)


def centerline_agreement(pred, target, valid_mask=None):
    """Symmetrized (precision + recall) / 2.

    This is the quantity Phase 0 reported as "centerline recall" between two
    annotators, where there is no asymmetry to break. Kept under its own name
    so the ceiling comparison uses like for like.
    """
    p, t, _ = prepare(pred, target, valid_mask)
    shape = _shape_of(pred, valid_mask)
    sp, st = skeletons(p, shape), skeletons(t, shape)
    prec = ratio(np.logical_and(sp, t).sum(1), sp.sum(1), BOTH_EMPTY)
    rec = ratio(np.logical_and(st, p).sum(1), st.sum(1), BOTH_EMPTY)
    both_empty = (p.sum(1) == 0) & (t.sum(1) == 0)
    one_empty = ((p.sum(1) == 0) ^ (t.sum(1) == 0))
    out = (prec + rec) / 2.0
    out = np.where(one_empty, 0.0, out)
    out = np.where(both_empty, BOTH_EMPTY, out)
    return like(out, pred)
