"""Instance-level (object) metrics: panoptic quality and its relatives.

The benchmark's own rubric (IoU, Dice, clDice, centerline recall) is pixel-
and skeleton-level and stays untouched so numbers remain comparable with the
published filament papers. This module adds the *object* view used by the
Kaggle "Solar Filament Segmentation Challenge 2026" / IEEE Big Data Cup 2026
leaderboard: a filament is one coherent object, and a model is also judged
on how many filaments it finds as objects, not only on how many pixels it
gets right.

Definitions (Kirillov et al., "Panoptic Segmentation", CVPR 2019, restricted
to the single `filament` class):

    instances   = 8-connected components of the binary mask inside `valid`
    match       = a predicted and a ground-truth instance with IoU > tau
                  (tau = 0.5 by default; above 0.5 the matching is unique,
                  for smaller tau it is greedy by descending IoU)
    TP, FP, FN  = matched pairs, unmatched predictions, unmatched truths

    SQ  = mean IoU over matched pairs            (segmentation quality)
    RQ  = TP / (TP + FP/2 + FN/2)                (recognition quality = F1)
    PQ  = SQ * RQ = sum IoU_matched / (TP + FP/2 + FN/2)

    inst_precision = TP / (TP + FP)
    inst_recall    = TP / (TP + FN)              ("hit rate")
    miss_rate      = FN / (TP + FN)              = 1 - hit rate
    rq_avg         = mean RQ over tau in {0.50, 0.55, ..., 0.95} -- the
                     score-free analogue of AP@[.5:.95] for a hard mask
    pixel_precision, pixel_recall = pixel-level, inside `valid`

The exact Kaggle text was not reachable from the cluster (the overview pages
are rendered client-side), so the thresholds are parameters and the names
above follow the public challenge description ("IoU, precision, recall,
AP@IoU theta, hit rate, miss rate"). Adjust `INSTANCE_TAUS` / `tau` once the
page's wording is confirmed; every reported number records the tau it used.

Empty-mask conventions match the rest of the package (`common.py`): both
empty -> 1.0 (miss rate 0.0); exactly one empty -> 0.0 (miss rate 1.0 when
the truth had instances). Every function returns per-image `[B]`.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from common.metrics.common import like, prepare, to_numpy_bool

STRUCT8 = np.ones((3, 3), dtype=bool)
DEFAULT_TAU = 0.5
INSTANCE_TAUS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
MIN_INSTANCE_PX = 0     # components smaller than this are dropped (both sides)


def label_instances(mask: np.ndarray, min_px: int = MIN_INSTANCE_PX):
    """8-connected components of a 2-D bool mask -> (labels int32, n)."""
    lab, n = ndi.label(mask, structure=STRUCT8)
    if n and min_px > 1:
        sizes = np.bincount(lab.ravel())
        keep = sizes >= min_px
        keep[0] = False
        lab = np.where(keep[lab], lab, 0)
        # relabel densely so n is the count that survives
        lab, n = ndi.label(lab > 0, structure=STRUCT8)
    return lab.astype(np.int32), int(n)


def iou_matrix(lab_p: np.ndarray, n_p: int, lab_t: np.ndarray, n_t: int):
    """[n_p, n_t] IoU between every predicted and every truth instance."""
    if n_p == 0 or n_t == 0:
        return np.zeros((n_p, n_t), dtype=np.float64)
    pair = lab_p.astype(np.int64) * (n_t + 1) + lab_t.astype(np.int64)
    inter = np.bincount(pair.ravel(), minlength=(n_p + 1) * (n_t + 1))
    inter = inter.reshape(n_p + 1, n_t + 1)[1:, 1:].astype(np.float64)
    area_p = np.bincount(lab_p.ravel(), minlength=n_p + 1)[1:].astype(np.float64)
    area_t = np.bincount(lab_t.ravel(), minlength=n_t + 1)[1:].astype(np.float64)
    union = area_p[:, None] + area_t[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def match(iou: np.ndarray, tau: float):
    """Greedy one-to-one matching by descending IoU; pairs need IoU > tau.

    For tau >= 0.5 at most one candidate per instance clears the bar, so the
    greedy result equals the PQ paper's unique matching.
    """
    n_p, n_t = iou.shape
    if n_p == 0 or n_t == 0:
        return []
    order = np.argsort(-iou, axis=None)
    used_p = np.zeros(n_p, bool)
    used_t = np.zeros(n_t, bool)
    pairs = []
    for flat in order:
        i, j = divmod(int(flat), n_t)
        v = iou[i, j]
        if v <= tau:
            break
        if used_p[i] or used_t[j]:
            continue
        used_p[i] = used_t[j] = True
        pairs.append((i, j, float(v)))
    return pairs


def _counts_at(iou: np.ndarray, tau: float) -> dict:
    n_p, n_t = iou.shape
    pairs = match(iou, tau)
    tp = len(pairs)
    return {"tp": tp, "fp": n_p - tp, "fn": n_t - tp,
            "iou_sum": float(sum(v for _, _, v in pairs))}


def _scores_from_counts(c: dict, n_p: int, n_t: int) -> dict:
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    denom = tp + 0.5 * fp + 0.5 * fn
    if n_p == 0 and n_t == 0:
        return {"pq": 1.0, "sq": 1.0, "rq": 1.0, "inst_precision": 1.0,
                "inst_recall": 1.0, "miss_rate": 0.0}
    if n_p == 0 or n_t == 0:
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "inst_precision": 0.0,
                "inst_recall": 0.0, "miss_rate": 1.0 if n_t else 0.0}
    return {
        "pq": c["iou_sum"] / denom if denom > 0 else 0.0,
        "sq": c["iou_sum"] / tp if tp > 0 else 0.0,
        "rq": tp / denom if denom > 0 else 0.0,
        "inst_precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "inst_recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "miss_rate": fn / (tp + fn) if (tp + fn) > 0 else 0.0,
    }


def instance_scores(pred2d: np.ndarray, target2d: np.ndarray,
                    valid2d: np.ndarray | None = None, tau: float = DEFAULT_TAU,
                    taus=INSTANCE_TAUS, min_px: int = MIN_INSTANCE_PX) -> dict:
    """Every instance metric for ONE image, computed from a single labeling.

    Returns a flat dict: pq, sq, rq, inst_precision, inst_recall, miss_rate
    (all at `tau`), rq_avg (mean RQ over `taus`), pixel_precision,
    pixel_recall, and the raw counts n_pred_instances / n_target_instances /
    tp / fp / fn so a table can be re-derived without the masks.
    """
    p = np.asarray(pred2d, bool)
    t = np.asarray(target2d, bool)
    if valid2d is not None:
        v = np.asarray(valid2d, bool)
        p, t = p & v, t & v
    lab_p, n_p = label_instances(p, min_px)
    lab_t, n_t = label_instances(t, min_px)
    iou = iou_matrix(lab_p, n_p, lab_t, n_t)

    c = _counts_at(iou, tau)
    out = _scores_from_counts(c, n_p, n_t)
    out["rq_avg"] = float(np.mean([
        _scores_from_counts(_counts_at(iou, th), n_p, n_t)["rq"] for th in taus]))

    inter = float(np.logical_and(p, t).sum())
    ps, ts = float(p.sum()), float(t.sum())
    if ps == 0 and ts == 0:
        out["pixel_precision"] = out["pixel_recall"] = 1.0
    else:
        out["pixel_precision"] = inter / ps if ps > 0 else 0.0
        out["pixel_recall"] = inter / ts if ts > 0 else 0.0

    out.update({"n_pred_instances": n_p, "n_target_instances": n_t,
                "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "tau": float(tau)})
    return out


INSTANCE_METRICS = ("pq", "sq", "rq", "rq_avg", "inst_precision", "inst_recall",
                    "miss_rate", "pixel_precision", "pixel_recall")


def _batched(name: str):
    def fn(pred, target, valid_mask=None, tau: float = DEFAULT_TAU):
        p, t, v = prepare(pred, target, valid_mask)
        shape = to_numpy_bool(pred).shape[-2:]
        vals = [instance_scores(p[i].reshape(shape), t[i].reshape(shape),
                                v[i].reshape(shape), tau=tau)[name]
                for i in range(p.shape[0])]
        return like(np.asarray(vals), pred)
    fn.__name__ = name
    fn.__doc__ = f"Per-image {name} (see module docstring). Returns [B]."
    return fn


panoptic_quality = pq = _batched("pq")
segmentation_quality = sq = _batched("sq")
recognition_quality = rq = _batched("rq")
rq_avg = _batched("rq_avg")
instance_precision = _batched("inst_precision")
instance_recall = hit_rate = _batched("inst_recall")
miss_rate = _batched("miss_rate")
pixel_precision = _batched("pixel_precision")
pixel_recall = _batched("pixel_recall")
