"""Port of the competition's self-evaluation notebook
(https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook, cells 3-4).

Semantics kept exactly:
  * one row per GT *annotator-image* (`<annotator>-<stem>`); the predictions
    for `<stem>` are scored once per annotator record of that stem;
  * predictions for stems without GT are ignored;
  * every GT/pred pair with IoU > 0.5 is a hit (no one-to-one matching);
    FP = pred columns with no hit, FN = GT rows with no hit;
  * PQ = sum IoU(hits) / (TP + 0.5 FP + 0.5 FN), pooled over all rows;
  * masks are full 2048 x 2048 frames (no analysis-disk masking).
Intersections use a prediction label map (`np.bincount`), which equals the
notebook's matrix product because submitted masks are pixel-disjoint.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import rle
from .settings import H, W


def stem_of(image_id_or_file: str) -> str:
    """`010102-20140321195814Mh` or `20140321195814Mh.jpeg` -> `20140321195814Mh`."""
    s = image_id_or_file
    if "-" in s and s.split("-", 1)[0].isdigit():
        s = s.split("-", 1)[1]
    return s.rsplit(".", 1)[0] if s.endswith((".jpeg", ".jpg")) else s

IOU_THRESHOLD = 0.5


# ------------------------------------------------------------------ GT frames

def gt_frame_from_coco(coco: dict, stems: set[str] | None = None,
                       records: set[str] | None = None) -> pd.DataFrame:
    """Every annotation of every record -> one row (`<record>_<k>`, RLE).
    `stems` keeps records of those stems only; `records` keeps those ids."""
    by = defaultdict(list)
    for a in coco["annotations"]:
        by[a["image_id"]].append(a)
    rows = []
    for im in coco["images"]:
        rid = im["id"]
        if stems is not None and stem_of(im["file_name"]) not in stems:
            continue
        if records is not None and rid not in records:
            continue
        for k, a in enumerate(by.get(rid, [])):
            m = rle.polygons_to_mask(a["segmentation"], H, W)
            rows.append({"filament_id": f"{rid}_{k}", "segmentation_rle": rle.encode(m)})
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def gt_frame(coco_path: Path, stems: set[str] | None = None,
             records: set[str] | None = None) -> pd.DataFrame:
    return gt_frame_from_coco(json.loads(Path(coco_path).read_text()), stems, records)


# ------------------------------------------------------------------ scoring

def _layers(rles: list[str]) -> np.ndarray:
    if not rles:
        return np.zeros((0, H, W), dtype=bool)
    return np.stack([rle.decode(r, H, W) for r in rles])


def overlap_matrices(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """IoU and Dice, shape (n_gt, n_pred). Predictions must be disjoint."""
    n_gt, n_pred = gt.shape[0], pred.shape[0]
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred)), np.zeros((n_gt, n_pred))
    lab = np.zeros((H, W), dtype=np.int32)
    for j in range(n_pred):
        lab[pred[j]] = j + 1
    if (pred.sum(0) > 1).any():
        raise ValueError("overlapping predicted masks")
    pred_area = pred.reshape(n_pred, -1).sum(1).astype(np.float64)
    gt_area = gt.reshape(n_gt, -1).sum(1).astype(np.float64)
    inter = np.stack([np.bincount(lab[gt[i]], minlength=n_pred + 1)[1:]
                      for i in range(n_gt)]).astype(np.float64)
    union = gt_area[:, None] + pred_area[None, :] - inter
    iou = np.where(union == 0, 0.0, inter / np.maximum(union, 1))
    dice = np.where(union == 0, 0.0, 2 * inter / np.maximum(gt_area[:, None] + pred_area[None, :], 1))
    return iou, dice


def overlap_rows(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> list[dict]:
    """The notebook's `get_overlap_df`, as a list of dicts."""
    gt_key = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_key = pred_df["filament_id"].str.split("_", n=1).str[0]
    pred_by_stem = {s: g["segmentation_rle"].tolist()
                    for s, g in pred_df.groupby(pred_key, sort=False)}
    pred_cache: dict[str, np.ndarray] = {}
    rows = []
    for ai, g in gt_df.groupby(gt_key, sort=False):
        stem = ai.split("-", 1)[1]
        gt = _layers(g["segmentation_rle"].tolist())
        if stem not in pred_cache:
            pred_cache[stem] = _layers(pred_by_stem.get(stem, []))
        pred = pred_cache[stem]
        iou, dice = overlap_matrices(gt, pred)
        rows.append({"annotator_image": ai, "iou_matrix": iou, "dice_matrix": dice,
                     "n_gt": gt.shape[0], "n_pred": pred.shape[0]})
    return rows


def pq_from_rows(rows: list[dict]) -> dict:
    tp_ious: list[float] = []
    fp = fn = 0
    for r in rows:
        if r["n_gt"] == 0:
            fp += r["n_pred"]
            continue
        if r["n_pred"] == 0:
            fn += r["n_gt"]
            continue
        hit = r["iou_matrix"] > IOU_THRESHOLD
        tp_ious.extend(r["iou_matrix"][hit].tolist())
        fp += int((hit.sum(0) == 0).sum())
        fn += int((hit.sum(1) == 0).sum())
    tp = len(tp_ious)
    denom = tp + 0.5 * fp + 0.5 * fn
    return {"pq": (sum(tp_ious) / denom) if denom > 0 else 0.0,
            "tp": tp, "fp": fp, "fn": fn,
            "sq": (sum(tp_ious) / tp) if tp else 0.0,
            "rq": (tp / denom) if denom else 0.0,
            "n_rows": len(rows)}


def score(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    rows = overlap_rows(gt_df, pred_df)
    out = pq_from_rows(rows)
    # the rubric's distributions: IoU / Dice of overlapping pairs, 1:n / n:1
    ious = np.concatenate([r["iou_matrix"].ravel() for r in rows] or [np.zeros(0)])
    dices = np.concatenate([r["dice_matrix"].ravel() for r in rows] or [np.zeros(0)])
    gt_deg, pred_deg = [], []
    for r in rows:
        h = r["iou_matrix"] > 0
        gt_deg += h.sum(1).tolist() if r["n_pred"] else [0] * r["n_gt"]
        pred_deg += h.sum(0).tolist() if r["n_gt"] else [0] * r["n_pred"]
    out.update(iou_mean_nonzero=float(ious[ious > 0].mean()) if (ious > 0).any() else 0.0,
               dice_mean_nonzero=float(dices[dices > 0].mean()) if (dices > 0).any() else 0.0,
               gt_one_to_many=int(sum(d > 1 for d in gt_deg)),
               pred_many_to_one=int(sum(d > 1 for d in pred_deg)),
               gt_degree_hist={str(k): int(v) for k, v in
                               zip(*np.unique(gt_deg, return_counts=True))} if gt_deg else {},
               pred_degree_hist={str(k): int(v) for k, v in
                                 zip(*np.unique(pred_deg, return_counts=True))} if pred_deg else {})
    return out


def read_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != ["filament_id", "segmentation_rle"]:
        raise ValueError(f"{path}: columns {list(df.columns)}")
    return df
