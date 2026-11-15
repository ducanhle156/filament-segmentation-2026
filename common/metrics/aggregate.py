"""Fold and seed aggregation (Phase 2, Step 12).

Aggregation happens here and nowhere else: the metrics themselves always
return per-image values, so nothing is reduced twice or averaged over the
wrong axis.

The three spreads reported are different quantities and answer different
questions:

`std_across_seeds`   how much the *training run* moves the number. This is
                     the one that decides whether two arms differ: rule 5
                     ("three seeds per arm, report mean and standard
                     deviation") means this one.
`std_across_images`  how variable the *data* is. Always much larger, and not
                     an error bar on the arm's mean.
`ci95`               bootstrap 95% interval of the mean, resampling images —
                     images are the sampling unit, seeds are repeated
                     measurements of the same images.

Empty ground truth is tracked explicitly. The 71 filament-free images score
1.0 whenever the model also predicts nothing, which inflates a pooled mean; so
every metric is reported both over all images and over filament-bearing images
only, exactly as Phase 0 reported the annotator ceiling.
"""

from __future__ import annotations

import collections

import numpy as np

DEFAULT_N_BOOT = 10000


def _per_image_mean(records, metric):
    """Average repeated measurements (seeds) of the same image."""
    by_image = collections.defaultdict(list)
    for r in records:
        v = r.get(metric)
        if v is not None and np.isfinite(v):
            by_image[r["image_id"]].append(float(v))
    ids = sorted(by_image)
    return ids, np.array([np.mean(by_image[i]) for i in ids], dtype=np.float64)


def _bootstrap_ci(values, n_boot, seed, alpha=0.05):
    if len(values) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return [float(lo), float(hi)]


def summarize(records, metric, n_boot=DEFAULT_N_BOOT, seed=0):
    """One metric over a list of per-image records. Returns a dict."""
    ids, per_image = _per_image_mean(records, metric)
    if len(per_image) == 0:
        return {"n_images": 0}

    by_seed = collections.defaultdict(list)
    for r in records:
        v = r.get(metric)
        if v is not None and np.isfinite(v):
            by_seed[r.get("seed", 0)].append(float(v))
    seed_means = np.array([np.mean(v) for _, v in sorted(by_seed.items())])

    return {
        "n_images": len(per_image),
        "n_seeds": len(seed_means),
        "n_folds": len({r.get("fold") for r in records}),
        "mean": float(per_image.mean()),
        "median": float(np.median(per_image)),
        "p10": float(np.percentile(per_image, 10)),
        "std_across_seeds": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0,
        "std_across_images": float(per_image.std(ddof=1)) if len(per_image) > 1 else 0.0,
        "seed_means": [float(v) for v in seed_means],
        "ci95": _bootstrap_ci(per_image, n_boot, seed),
    }


POOLED_KEYS = ("pooled_iou", "pooled_dice", "pooled_pq", "pooled_sq", "pooled_rq",
               "pooled_inst_precision", "pooled_inst_recall", "pooled_miss_rate",
               "n_gt_instances", "n_pred_instances")


def pooled(records):
    """Dataset-level (micro) scores: sums over every image first, ratio once.

    The per-image mean above weights a disk with one small filament exactly
    like a disk with twenty; pooling weights every pixel and every filament
    instance equally, which is how a leaderboard-style PQ is computed. Both
    are reported; neither replaces the other. Needs the per-image counts the
    evaluators write (`tp/fp/fn`, `sq`, `n_pred_pixels`, `n_target_pixels`,
    `pixel_recall`); records without them are skipped. Repeated seeds pool
    naturally — every image contributes once per seed, so the ratio is the
    seed-averaged pooled score.
    """
    inter = union = psum = tsum = 0.0
    tp = fp = fn = iou_sum = 0.0
    n = 0
    for r in records:
        if "tp" not in r or "n_target_pixels" not in r:
            continue
        n += 1
        P, T = float(r["n_pred_pixels"]), float(r["n_target_pixels"])
        I = float(r.get("pixel_recall", 0.0)) * T if T else 0.0
        inter += I; union += P + T - I; psum += P; tsum += T
        tp += r["tp"]; fp += r["fp"]; fn += r["fn"]
        iou_sum += float(r.get("sq", 0.0)) * r["tp"]
    if n == 0:
        return {}
    denom = tp + 0.5 * fp + 0.5 * fn
    return {
        "n_records": n,
        "pooled_iou": inter / union if union else float("nan"),
        "pooled_dice": 2 * inter / (psum + tsum) if (psum + tsum) else float("nan"),
        "pooled_pq": iou_sum / denom if denom else float("nan"),
        "pooled_sq": iou_sum / tp if tp else 0.0,
        "pooled_rq": tp / denom if denom else float("nan"),
        "pooled_inst_precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "pooled_inst_recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "pooled_miss_rate": fn / (tp + fn) if (tp + fn) else 0.0,
        "n_gt_instances": int(tp + fn), "n_pred_instances": int(tp + fp),
    }


def aggregate(records, metrics=None, n_boot=DEFAULT_N_BOOT, seed=0):
    """Aggregate per-image records across folds and seeds.

    Each record is a mapping with at least `image_id` and the metric values;
    `seed`, `fold` and `target_empty` are used when present. Returns
    {"all": {metric: summary}, "filament_bearing": {...}, "counts": {...},
     "pooled": {...}} — see `pooled()` for the dataset-level (micro) scores.
    """
    records = list(records)
    if metrics is None:
        skip = {"image_id", "seed", "fold", "arm_id", "target_empty", "file_name"}
        metrics = sorted({k for r in records for k, v in r.items()
                          if k not in skip and isinstance(v, (int, float))})

    bearing = [r for r in records if not r.get("target_empty", False)]
    empty = [r for r in records if r.get("target_empty", False)]

    return {
        "metrics": list(metrics),
        "counts": {
            "n_records": len(records),
            "n_images": len({r["image_id"] for r in records}),
            "n_images_filament_free": len({r["image_id"] for r in empty}),
            "n_images_filament_bearing": len({r["image_id"] for r in bearing}),
            "seeds": sorted({r.get("seed", 0) for r in records}),
            "folds": sorted({r.get("fold") for r in records if "fold" in r}),
        },
        "all": {m: summarize(records, m, n_boot, seed) for m in metrics},
        "filament_bearing": {m: summarize(bearing, m, n_boot, seed)
                             for m in metrics},
        "pooled": pooled(records),
    }


def compare(summary_a, summary_b, metric):
    """Difference between two arms on one metric, with a seed-noise yardstick.

    `significant` is deliberately conservative: the gap must exceed the sum of
    the two arms' seed-to-seed standard deviations. With three seeds there is
    no power for anything stronger, and rule 5 exists precisely so single-run
    differences are not reported as results.
    """
    a, b = summary_a["all"][metric], summary_b["all"][metric]
    delta = a["mean"] - b["mean"]
    noise = a["std_across_seeds"] + b["std_across_seeds"]
    return {
        "metric": metric,
        "mean_a": a["mean"], "mean_b": b["mean"], "delta": delta,
        "seed_noise": noise,
        "significant": bool(abs(delta) > noise > 0),
    }
