"""Shared machinery for the zero-shot probes (Phase 4, Step 20).

arm04 (SAM 2 automatic masks) and arm05 (SAM 3 concept prompts) both produce
*sets of candidate masks* for a full disk, not one probability map. This
module turns such a set into the per-image rows every other arm writes, under
two explicitly different readings of "the prediction":

`union`   every generated mask OR'd together. What a prompt-free / text-only
          user would actually get. Expected to be terrible: SAM covers the
          whole disk with regions and most of them are not filaments.
`oracle`  only the generated masks whose best IoU against a single ground-truth
          filament component is >= `match_tau` (default 0.5), OR'd together.
          The ground truth is used to *select*, never to edit, masks — so this
          is the upper bound of what the model's mask vocabulary could give a
          perfect selector. It answers "does SAM ever draw a filament?"
          separately from "does it know which region is one?".

Both are scored with the same `image_metrics` (both rubrics) as the trained
arms, on the same valid mask, so the numbers sit in the same tables. `union`
is the arm's headline `metrics.json`; `oracle` is `metrics_oracle.json`.

Images. SAM expects natural RGB. `render_disk` maps the harness' z-scored
disk (clipped to [-`lo`, `hi`], off-disk 0) linearly to uint8 and replicates
it to three channels. No contrast tricks beyond that fixed window, so the
protocol is one line.

Sharding. A shard writes `metrics_shard{I}of{N}.json` (and the oracle twin);
`scripts/merge_shards.py` combines them into the unsharded schema.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from common.data.preprocess import load_interim
from common.data.splits import load_folds
from common.engine.evaluate import image_metrics
from common.metrics.instance import STRUCT8
from common.utils.config import config_hash
from common.utils.shard import shard_ids, shard_suffix

READINGS = ("union", "oracle")


# ------------------------------------------------------------------ inputs

def render_disk(img: np.ndarray, valid: np.ndarray, lo: float = -2.5,
                hi: float = 2.5) -> np.ndarray:
    """z-scored [H,W] float -> uint8 RGB [H,W,3]; off-disk black."""
    x = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    x = np.where(valid, x, 0.0)
    g = np.round(x * 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=-1)


def fold_test_ids(cfg: dict, root: Path, fold: int, shard: str | None = None,
                  limit: int | None = None) -> list[str]:
    ids = load_folds(root / cfg["data"]["splits_file"])[fold]["test"]
    ids = shard_ids(ids, shard)
    return ids[:limit] if limit else ids


def load_disk(cfg: dict, root: Path, image_id: str) -> dict:
    stem = image_id.partition("-")[2]
    return load_interim(root / cfg["data"]["interim_dir"] / f"{stem}.npz")


# ---------------------------------------------------------------- matching

def gt_components(target: np.ndarray, valid: np.ndarray):
    lab, n = ndi.label(target & valid, structure=STRUCT8)
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    return lab, n, areas


def best_iou_per_mask(masks, lab: np.ndarray, n: int, areas: np.ndarray,
                      valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per candidate mask: best IoU against any single GT component, and
    which component (0 = none)."""
    best = np.zeros(len(masks), dtype=np.float32)
    comp = np.zeros(len(masks), dtype=np.int64)
    if n == 0:
        return best, comp
    for k, m in enumerate(masks):
        m = np.asarray(m, dtype=bool) & valid
        area = int(m.sum())
        if area == 0:
            continue
        inter = np.bincount(lab[m], minlength=n + 1)[1:]
        hit = np.nonzero(inter > 0)[0]
        if hit.size == 0:
            continue
        iou = inter[hit] / (area + areas[1:][hit] - inter[hit])
        a = int(np.argmax(iou))
        best[k], comp[k] = float(iou[a]), int(hit[a]) + 1
    return best, comp


def readings(masks, target: np.ndarray, valid: np.ndarray,
             match_tau: float = 0.5) -> tuple[dict, dict]:
    """The two predictions plus the per-image candidate statistics."""
    shape = target.shape
    union = np.zeros(shape, dtype=bool)
    oracle = np.zeros(shape, dtype=bool)
    lab, n, areas = gt_components(target, valid)
    ious, comps = best_iou_per_mask(masks, lab, n, areas, valid)
    matched = ious >= match_tau
    for m, ok in zip(masks, matched):
        m = np.asarray(m, dtype=bool)
        union |= m
        if ok:
            oracle |= m
    union &= valid
    oracle &= valid
    drawn = np.unique(comps[matched])
    stats = {"n_candidates": int(len(masks)),
             "n_matched_candidates": int(matched.sum()),
             "n_gt_components": int(n),
             "n_gt_components_drawn": int((drawn > 0).sum()),
             "best_candidate_iou": float(ious.max()) if len(ious) else 0.0,
             "candidate_iou_mean": float(ious.mean()) if len(ious) else 0.0}
    return {"union": union, "oracle": oracle}, stats


# ------------------------------------------------------------------- rows

def image_rows(image_id: str, preds: dict, target: np.ndarray,
               valid: np.ndarray, cfg: dict, stats: dict | None = None,
               extra: dict | None = None) -> dict[str, dict]:
    """One per-image row per reading, in the trained arms' schema."""
    rows = {}
    for name, pred in preds.items():
        row = {"image_id": image_id,
               "target_empty": bool(not (target & valid).any()),
               "n_pred_pixels": int((pred & valid).sum()),
               "n_target_pixels": int((target & valid).sum())}
        row.update(image_metrics(pred, target, valid, cfg["eval"]["metrics"],
                                 float(cfg["eval"].get("instance_tau", 0.5))))
        row.update(stats or {})
        row.update(extra or {})
        rows[name] = row
    return rows


def run_dir(cfg: dict, root: Path, fold: int, seed: int = 0) -> Path:
    return root / cfg["results_dir"] / cfg["arm_id"] / f"seed{seed}" / f"fold{fold}"


def write_metrics(cfg: dict, root: Path, fold: int, rows_by_reading: dict,
                  protocol: dict, t_start: float, shard: str | None,
                  device_name: str, n_train: int = 0, n_val: int = 0,
                  seed: int = 0, sub_run: str | None = None) -> list[Path]:
    """metrics[_oracle][_shardIofN].json, same top-level keys as train_fold
    so collect_phase3 / make_figures / insights need no special case."""
    import torch
    out = run_dir(cfg, root, fold, seed)
    if sub_run:
        out = out / sub_run
    out.mkdir(parents=True, exist_ok=True)
    peak = (torch.cuda.max_memory_allocated() / 2 ** 30
            if torch.cuda.is_available() else 0.0)
    written = []
    for reading, rows in rows_by_reading.items():
        doc = {
            "arm_id": cfg["arm_id"], "fold": fold, "seed": seed,
            "sub_run": sub_run, "reading": reading,
            "config_sha256": config_hash(cfg),
            "arm_config_file": cfg.get("_arm_config_file"),
            "cache_version": cfg["data"]["processed_dir"],
            "splits_file": cfg["data"]["splits_file"],
            "git_commit": cfg.get("_git_commit"),
            "device": device_name, "smoke": False,
            "n_train_images": n_train, "n_val_images": n_val,
            "n_test_images": len(rows),
            "best_epoch": -1, "best_val_cldice": None, "epochs_run": 0,
            "epochs_budget": 0, "stop_epoch": None, "anneal_epochs": 0,
            "wall_clock_s": time.time() - t_start,
            "peak_gpu_memory_gib": peak,
            "total_parameters": protocol.get("total_parameters", 0),
            "trainable_parameters": 0,
            "loss_terms": None, "augmentation": None,
            "zero_shot": True, "protocol": protocol,
            "history": [], "per_image": rows,
        }
        name = "metrics" + ("" if reading == "union" else f"_{reading}")
        path = out / f"{name}{shard_suffix(shard)}.json"
        path.write_text(json.dumps(doc, indent=1) + "\n")
        written.append(path)
    return written


# ----------------------------------------------------------------- figure

def qualitative_figure(records: list[dict], path: Path, title: str) -> Path:
    """Rows = images; columns = rendered disk, GT, every candidate mask
    (random colours), oracle-selected masks over GT."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(records)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n), squeeze=False)
    rng = np.random.default_rng(0)
    for r, rec in zip(axes, records):
        rgb, target, valid = rec["rgb"], rec["target"], rec["valid"]
        r[0].imshow(rgb)
        r[0].set_title(rec["image_id"], fontsize=8)
        r[1].imshow(rgb)
        r[1].contour(target & valid, levels=[0.5], colors="lime", linewidths=0.6)
        r[1].set_title("ground truth", fontsize=8)
        canvas = np.zeros(target.shape + (3,), dtype=np.float32)
        for m in rec["masks"]:
            canvas[np.asarray(m, dtype=bool)] = rng.uniform(0.2, 1.0, 3)
        r[2].imshow(canvas)
        r[2].set_title(f"{len(rec['masks'])} candidate masks ({rec['label']})",
                       fontsize=8)
        r[3].imshow(rgb)
        r[3].contour(target & valid, levels=[0.5], colors="lime", linewidths=0.6)
        if rec["oracle"].any():
            r[3].contour(rec["oracle"], levels=[0.5], colors="red", linewidths=0.6)
        r[3].set_title("oracle-selected (red) vs GT (green)", fontsize=8)
        for a in r:
            a.set_axis_off()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
