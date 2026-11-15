"""Shared helpers for the try_error experiments (2026-11-13).

Local validation = the inner-test parts of `kaggle/splits/folds_k707_v1.json`
(5 folds over the 707 competition train disks, grouped by 7-day observation
window, every disk predicted once by a model that never saw it). Scoring
is the official self-evaluation rule (IoU > 0.5 is a hit, full frame, every
annotator record scored separately, counts pooled), computed on label maps;
`score_rows` reproduces `kaggle/src/scorer.py` exactly (asserted by
diagnose.py against the stored 4858/3322/3341).

Nothing here reads a test label, a file-name timestamp or a station code.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from paths import INTERIM, LADDER, OUT, SPLITS, stem_of  # noqa: E402
import rle  # noqa: E402
from common.data.preprocess import load_interim  # noqa: E402
from common.data.splits import load_folds  # noqa: E402
from common.engine.decode import hysteresis_decode  # noqa: E402

H = W = 2048
STRUCT8 = np.ones((3, 3), dtype=bool)
EVAL = OUT / "results" / "evaluate"
ARM = "kag13_sam2_pqsel"
# the submitted decode cell: overlap 0.50, seed t 0.50 / area 400 / mean p 0.85, grow to 0.45
BASE = dict(overlap=0.50, hi=0.50, lo=0.45, min_area=400, min_prob=0.85)
IOU_T = 0.5

_GT = None


def jobs() -> list[tuple[int, str]]:
    """(fold, stem) for every inner-test disk; each of the 707 exactly once."""
    out = []
    for f, v in sorted(load_folds(SPLITS).items()):
        out += [(f, stem_of(i)) for i in v["test"]]
    return out


def gt_records(stem: str) -> dict[str, list[np.ndarray]]:
    """{record id: [flat pixel indices of each filament]} for one train disk."""
    global _GT
    if _GT is None:
        import pandas as pd
        g = pd.read_csv(EVAL / "gt_train_all_records.csv")
        rec = g.filament_id.str.split("_", n=1).str[0]
        _GT = defaultdict(dict)
        for ai, rles in g.groupby(rec, sort=False)["segmentation_rle"]:
            _GT[ai.split("-", 1)[1]][ai] = rles.tolist()
    return {r: [np.flatnonzero(rle.decode(x, H, W).ravel()) for x in v]
            for r, v in _GT.get(stem, {}).items()}


def load_prob(fold: int, stem: str, overlap: float = 0.50, arm: str = ARM) -> np.ndarray:
    with np.load(LADDER / arm / "seed0" / f"fold{fold}" / "probs" / f"{stem}.npz") as z:
        return z[f"o{overlap:.2f}"].astype(np.float32) / 255.0


def load_disk(stem: str, source: str = "cache") -> dict:
    """Preprocessed image + analysis-disk mask (the cached label is not returned).
    source "cache": project cache data/interim; "kaggle": the competition JPEG run
    through the same preprocessing (prepare_kaggle_train.py)."""
    d = load_interim((INTERIM if source == "cache" else HERE / "data" / "interim_k") / f"{stem}.npz")
    return {"img": d["img"], "valid": d["valid"]}


def baseline_labels(prob: np.ndarray, valid: np.ndarray, **kw) -> tuple[np.ndarray, int]:
    c = dict(BASE, **kw)
    mask = hysteresis_decode(prob, valid, c["hi"], c["lo"], c["min_area"], c["min_prob"])
    return ndi.label(mask, structure=STRUCT8)


def overlaps(lab: np.ndarray, n: int, gts: list[np.ndarray]) -> dict:
    """Intersection / IoU / Dice of one annotator record against a label map."""
    flat = lab.ravel()
    pa = np.bincount(flat, minlength=n + 1)[1:].astype(np.float64)
    ga = np.array([len(i) for i in gts], dtype=np.float64)
    inter = (np.stack([np.bincount(flat[i], minlength=n + 1)[1:] for i in gts]).astype(np.float64)
             if gts else np.zeros((0, n)))
    union = ga[:, None] + pa[None, :] - inter
    iou = inter / np.maximum(union, 1)
    dice = 2 * inter / np.maximum(ga[:, None] + pa[None, :], 1)
    return {"inter": inter, "iou_matrix": iou, "dice_matrix": dice, "gt_area": ga,
            "pred_area": pa, "n_gt": len(gts), "n_pred": n}


def score_rows(rows: list[dict]) -> dict:
    """Pooled PQ / SQ / RQ + the rubric's extras, same rules as scorer.score."""
    s = tp = fp = fn = 0
    one_many = many_one = 0
    dices, ious, tp_dice = [], [], []
    for r in rows:
        if r["n_gt"] == 0:
            fp += r["n_pred"]
            continue
        if r["n_pred"] == 0:
            fn += r["n_gt"]
            continue
        iou = r["iou_matrix"]
        hit = iou > IOU_T
        s += float(iou[hit].sum())
        tp += int(hit.sum())
        fp += int((hit.sum(0) == 0).sum())
        fn += int((hit.sum(1) == 0).sum())
        ov = iou > 0
        one_many += int((ov.sum(1) > 1).sum())       # one GT filament touched by several predictions
        many_one += int((ov.sum(0) > 1).sum())       # one prediction touching several GT filaments
        ious.append(iou[ov])
        dices.append(r["dice_matrix"][ov])
        tp_dice.append(r["dice_matrix"][hit])
    den = tp + 0.5 * fp + 0.5 * fn
    def mean(x):
        x = np.concatenate(x) if x else np.zeros(0)
        return float(x.mean()) if x.size else 0.0
    return {"pq": s / den if den else 0.0, "sq": s / tp if tp else 0.0, "rq": tp / den if den else 0.0,
            "tp": tp, "fp": fp, "fn": fn,
            "dice_overlapping": mean(dices),
            "dice_matched": mean(tp_dice),
            "iou_overlapping": mean(ious),
            "one_to_many": one_many, "many_to_one": many_one, "n_rows": len(rows)}


def pixel_dice(lab: np.ndarray, gts: list[np.ndarray]) -> float:
    """Semantic (foreground) Dice of the whole prediction against one record."""
    g = np.zeros(H * W, dtype=bool)
    for i in gts:
        g[i] = True
    p = lab.ravel() > 0
    d = g.sum() + p.sum()
    return float(2 * (g & p).sum() / d) if d else 1.0


def fmt(m: dict) -> str:
    return (f"PQ {m['pq']:.4f}  SQ {m['sq']:.3f}  RQ {m['rq']:.3f}  TP/FP/FN {m['tp']}/{m['fp']}/{m['fn']}  "
            f"Dice(matched) {m['dice_matched']:.3f}  1:n {m['one_to_many']}  n:1 {m['many_to_one']}")
