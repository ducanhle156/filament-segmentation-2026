"""Probability map -> instance mask decoders (Phase 7-1 / 7-2).

The benchmark decode grew in three val-fit / test-apply steps
(post_processing/decode_rescore, confidence_filter): binarize at a
threshold, drop components under a minimum area, drop components whose
mean probability is below a floor. `decode_components` is that chain in
one place so training-time validation (train.select_metric: pq) and the
post-processing grids apply exactly the same rule.

`hysteresis_decode` (Phase 7-2 Task E) keeps the tuned chain as the SEED
rule and then grows every surviving component into the connected region of
a lower threshold. The failure review showed fragmented filaments are
bridged by pixels the model scores 0.2-0.5 -- too low to survive the 0.50
cut, high enough to link islands that share one low-threshold component.
lo == hi is the identity (the tuned decode unchanged).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from common.metrics.instance import STRUCT8, label_instances


def decode_components(prob: np.ndarray, valid: np.ndarray,
                      threshold: float = 0.5, min_area: int = 0,
                      min_prob: float = 0.0) -> np.ndarray:
    """Boolean mask of the components kept by the (t, min-area, min-prob)
    chain. Components are 8-connected; `min_prob` is the component's mean
    probability over its own pixels."""
    mask = (prob >= threshold) & valid
    lab, n = label_instances(mask, min_px=int(min_area))
    if n == 0:
        return np.zeros_like(mask)
    if min_prob > 0:
        area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        sums = np.bincount(lab.ravel(), weights=prob.ravel(),
                           minlength=n + 1)[1:]
        keep = (sums / np.maximum(area, 1)) >= min_prob
        lab = np.where(np.isin(lab, np.nonzero(keep)[0] + 1), lab, 0)
    return lab > 0


def hysteresis_decode(prob: np.ndarray, valid: np.ndarray, hi: float,
                      lo: float, min_area: int = 0,
                      min_prob: float = 0.0) -> np.ndarray:
    """Seeds = decode_components(hi, min_area, min_prob); output = every
    lo-threshold 8-connected component that contains at least one seed
    pixel. No new components can appear (every kept region holds a seed)
    but two seeds inside one lo-region merge into one instance."""
    seed = decode_components(prob, valid, hi, min_area, min_prob)
    if lo >= hi:
        return seed
    low = (prob >= lo) & valid
    lab, n = ndi.label(low, structure=STRUCT8)
    if n == 0:
        return seed
    keep = np.unique(lab[seed])
    keep = keep[keep > 0]
    return np.isin(lab, keep)
