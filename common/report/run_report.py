"""Per-run figures: what one (arm, fold, seed) produced (`common/report/`).

Split in two on purpose.

`write_run_figures` needs nothing but the `metrics.json` a finished run already
wrote, so it is cheap, CPU-only, and can be re-run over the whole results tree
at any time to backfill figures for jobs that finished before this module
existed.

`write_qualitative_figures` reloads the checkpoint and re-runs full-disk
inference, so it needs the GPU the training job is already holding. That is why
the run job emits it at the end rather than a reporting pass doing it later.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.report.figures import (metric_distribution_figure,
                                   prediction_grid_figure,
                                   stitched_prediction_figure,
                                   training_curves_figure)

# What "works / typical / fails" means for a run, in selection order.
QUALITATIVE_PICKS = ("best", "median", "worst")

# Full-disk gallery: the same fixed test disks for every arm and seed of a
# fold, so a reader can flip between arms and look at one disk. Chosen
# deterministically (first N filament-bearing test ids in sorted order),
# never by score -- a score-picked disk shows the run at its best.
GALLERY_N = 8


def gallery_images(per_image: list[dict], n: int = GALLERY_N) -> list[dict]:
    rows = sorted((r for r in per_image if not r.get("target_empty", False)),
                  key=lambda r: r["image_id"])
    return rows[:n]


def provenance(metrics: dict) -> str:
    """The one-line stamp every figure carries."""
    commit = (metrics.get("git_commit") or "")[:8]
    return (f"{metrics.get('arm_id')}  fold {metrics.get('fold')}  "
            f"seed {metrics.get('seed')}  commit {commit or '?'}  "
            f"cache {metrics.get('cache_version')}  "
            f"splits {metrics.get('splits_file')}")


def pick_images(per_image: list[dict], metric: str = "cldice",
                picks=QUALITATIVE_PICKS) -> list[tuple[str, dict]]:
    """Best / median / worst test disk by `metric`.

    Filament-free disks are excluded: a blank disk scores well by predicting
    nothing, so it would take the "best" slot every time and show nothing.
    """
    rows = [r for r in per_image
            if metric in r and not r.get("target_empty", False)]
    if not rows:
        return []
    rows.sort(key=lambda r: r[metric])
    idx = {"worst": 0, "median": len(rows) // 2, "best": len(rows) - 1}
    out, seen = [], set()
    for p in picks:
        r = rows[idx[p]]
        if r["image_id"] not in seen:
            seen.add(r["image_id"])
            out.append((p, r))
    return out


def write_run_figures(run_dir: Path, metrics: dict,
                      metric_names: list[str] | None = None) -> list[Path]:
    """Training curve + per-image distribution for one run. No GPU, no model."""
    run_dir = Path(run_dir)
    fig_dir = run_dir / "figures"
    stamp = provenance(metrics)
    label = f"{metrics.get('arm_id')} s{metrics.get('seed')}f{metrics.get('fold')}"
    written = []

    if metrics.get("history"):
        written.append(training_curves_figure(
            [{"label": label, "history": metrics["history"]}],
            fig_dir / "training_curves.png",
            title=f"{label} — training", provenance=stamp))

    rows = metrics.get("per_image") or []
    if rows:
        names = metric_names or [m for m in ("iou", "dice", "cldice", "miou",
                                             "centerline_recall")
                                 if m in rows[0]]
        written.append(metric_distribution_figure(
            [{"label": label, "rows": rows}], names,
            fig_dir / "per_image_metrics.png",
            title=f"{label} — {len(rows)} test images", provenance=stamp))
        inst_names = [m for m in ("pq", "sq", "rq", "inst_precision",
                                  "inst_recall", "miss_rate") if m in rows[0]]
        if inst_names:
            written.append(metric_distribution_figure(
                [{"label": label, "rows": rows}], inst_names,
                fig_dir / "per_image_instance_metrics.png",
                title=f"{label} — object-level (PQ) view, {len(rows)} test images",
                provenance=stamp))
    return written


def write_qualitative_figures(run_dir: Path, cfg: dict, metrics: dict,
                              root: Path, device=None) -> list[Path]:
    """Prediction-vs-truth panels for the best, median and worst test disk.

    Rebuilds the model from `cfg`, loads `checkpoint.pt` from `run_dir`, and
    re-stitches those disks. Returns an empty list if the checkpoint is missing
    (a `save_checkpoint: false` arm, or a job killed before it wrote one).
    """
    import torch

    from common.data.channels import ChannelSource, stack_input
    from common.data.preprocess import load_interim
    from common.engine.registry import build_model
    from common.engine.stitch import predict_full_disk

    run_dir, root = Path(run_dir), Path(root)
    channels = ChannelSource.from_config(cfg, root)
    ckpt_path = run_dir / "checkpoint.pt"
    picks = pick_images(metrics.get("per_image") or [])
    if not ckpt_path.exists() or not picks:
        return []

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])

    threshold = float(cfg["eval"].get("threshold", 0.5))
    interim = root / cfg["data"]["interim_dir"]
    label = f"{metrics.get('arm_id')} s{metrics.get('seed')}f{metrics.get('fold')}"
    stamp = provenance(metrics)
    # the headline subset of both rubrics; the full row is in metrics.json
    show = ("iou", "dice", "cldice", "centerline_recall",
            "pq", "sq", "rq", "inst_recall", "miss_rate")
    keep = [m for m in show if m in picks[0][1]]

    def infer(iid):
        stem = iid.partition("-")[2]
        d = load_interim(interim / f"{stem}.npz")
        prob = predict_full_disk(model, stack_input(d["img"], channels, stem),
                                 d["valid"],
                                 cfg["data"]["tile_size"], cfg["data"]["overlap"],
                                 device, amp=cfg["train"].get("amp", True))
        return d, prob

    rows, detail = [], None
    for kind, row in picks:
        iid = row["image_id"]
        d, prob = infer(iid)
        rows.append({"image_id": f"{kind}: {iid}", "img": d["img"],
                     "valid": d["valid"], "target": d["mask"],
                     "probs": {label: prob},
                     "metrics": {label: {k: row[k] for k in keep}}})
        if kind == "median" or detail is None:
            detail = (iid, d, prob, {k: row[k] for k in keep})

    fig_dir = run_dir / "figures"
    written = [prediction_grid_figure(
        rows, [label], fig_dir / "predictions.png", threshold=threshold,
        title=f"{label} — prediction vs ground truth", provenance=stamp)]

    iid, d, prob, keep_m = detail
    written.append(stitched_prediction_figure(
        iid, d["img"], d["mask"], prob, d["valid"],
        fig_dir / "prediction_detail.png", threshold, keep_m, provenance=stamp))

    # Full-disk gallery: input | ground truth | probability | outcome, one
    # file per disk, the same disks for every arm/seed of this fold. The
    # binarized prediction and probability map are saved next to it so the
    # disk can be re-drawn or re-scored without a GPU.
    gallery = gallery_images(metrics.get("per_image") or [],
                             int(cfg.get("eval", {}).get("gallery_n", GALLERY_N)))
    gal_dir = fig_dir / "disks"
    pred_dir = run_dir / "predictions_gallery"
    for row in gallery:
        iid = row["image_id"]
        d, prob = infer(iid)
        written.append(stitched_prediction_figure(
            iid, d["img"], d["mask"], prob, d["valid"],
            gal_dir / f"{iid}.png", threshold, {k: row[k] for k in keep},
            provenance=stamp))
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pred_dir / f"{iid}.npz",
                            prob=(prob * 255).astype(np.uint8),
                            pred=np.packbits(prob >= threshold),
                            shape=np.asarray(prob.shape))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return written
