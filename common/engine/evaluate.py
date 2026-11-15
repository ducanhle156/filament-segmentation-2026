"""Evaluation: per-image metrics over stitched full disks (Phase 2, Step 13)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.data.channels import stack_input
from common.data.preprocess import load_interim
from common.engine.decode import decode_components
from common.engine.stitch import binarize, predict_full_disk
from common.metrics import centerline as cl
from common.metrics import cldice_metric as cd
from common.metrics import instance as inst
from common.metrics import iou as I

METRIC_FNS = {
    "iou": I.foreground_iou,
    "background_iou": I.background_iou,
    "miou": I.mean_iou,
    "dice": I.dice,
    "cldice": cd.cldice,
    "centerline_recall": cl.centerline_recall,
    "centerline_precision": cl.centerline_precision,
    # object-level view (Kaggle / IEEE Big Data Cup 2026 rubric); all of
    # these come from one labeling, so image_metrics computes them together
    "pq": inst.pq, "sq": inst.sq, "rq": inst.rq, "rq_avg": inst.rq_avg,
    "inst_precision": inst.instance_precision,
    "inst_recall": inst.instance_recall, "miss_rate": inst.miss_rate,
    "pixel_precision": inst.pixel_precision,
    "pixel_recall": inst.pixel_recall,
}
INSTANCE_NAMES = set(inst.INSTANCE_METRICS)


def image_metrics(pred: np.ndarray, target: np.ndarray, valid: np.ndarray,
                  metrics=None, instance_tau: float = inst.DEFAULT_TAU) -> dict:
    """All requested metrics for one image. Every one is valid-masked.

    Instance metrics also add the raw counts (n_pred_instances,
    n_target_instances, tp, fp, fn) so tables can be re-derived later."""
    names = metrics or list(METRIC_FNS)
    p, t, v = pred[None], target[None], valid[None]
    out = {}
    shared = None
    for n in names:
        fn = METRIC_FNS.get(n)
        if fn is None:
            raise KeyError(f"unknown metric {n!r}; have {sorted(METRIC_FNS)}")
        if n in INSTANCE_NAMES:
            if shared is None:       # one connected-component pass per image
                shared = inst.instance_scores(pred, target, valid,
                                              tau=instance_tau)
            out[n] = float(shared[n])
        else:
            out[n] = float(np.asarray(fn(p, t, v))[0])
    if shared is not None:
        for k in ("n_pred_instances", "n_target_instances", "tp", "fp", "fn"):
            out[k] = int(shared[k])
    return out


def evaluate_images(model, image_ids, interim_dir: Path, cfg: dict, device,
                    threshold: float = 0.5, save_predictions: Path | None = None,
                    limit: int | None = None, channels=None) -> list[dict]:
    """Per-image records: one row per test image, never a pre-reduced score.

    `channels` (a `ChannelSource`, Phase 8) stacks the arm's side channels
    under the H-alpha disk before stitching; None is the single-channel path."""
    tile = cfg["data"]["tile_size"]
    overlap = cfg["data"]["overlap"]
    metrics = cfg["eval"]["metrics"]
    amp = cfg["train"].get("amp", True)
    rows = []
    ids = sorted(image_ids)[:limit] if limit else sorted(image_ids)

    for iid in ids:
        stem = iid.partition("-")[2]
        d = load_interim(Path(interim_dir) / f"{stem}.npz")
        prob = predict_full_disk(model, stack_input(d["img"], channels, stem),
                                 d["valid"], tile, overlap, device, amp=amp)
        pred = binarize(prob, threshold)
        target = d["mask"]
        row = {
            "image_id": iid,
            "target_empty": bool(not (target & d["valid"]).any()),
            "n_pred_pixels": int((pred & d["valid"]).sum()),
            "n_target_pixels": int((target & d["valid"]).sum()),
        }
        row.update(image_metrics(pred, target, d["valid"], metrics,
                                 float(cfg["eval"].get("instance_tau", 0.5))))
        rows.append(row)
        if save_predictions is not None:
            save_predictions.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save_predictions / f"{stem}.npz",
                                prob=(prob * 255).astype(np.uint8),
                                pred=np.packbits(pred))
    return rows


def validation_score(model, loader, device, cfg, threshold: float = 0.5) -> dict:
    """Tile-level validation for early stopping.

    Early stopping watches **clDice, not loss** (spec, Step 13): with 1%
    foreground the loss is dominated by background pixels and keeps improving
    long after the thin structures have stopped getting better.
    """
    import torch

    amp = cfg["train"].get("amp", True)
    scores, losses = [], []
    fg_px, valid_px = 0, 0
    from common.losses.combo import build_loss
    criterion = build_loss(cfg).to(device)

    model.eval()
    with torch.no_grad():
        for img, mask, valid in loader:
            img, mask, valid = (t.to(device) for t in (img, mask, valid))
            if amp and device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(img)
            else:
                logits = model(img)
            logits = logits.float()
            loss, _ = criterion(logits, mask, valid,
                                img[:, 1:].float() if img.shape[1] > 1 else None)
            losses.append(float(loss))
            pred = (torch.sigmoid(logits) >= threshold).squeeze(1).cpu().numpy()
            t = mask.squeeze(1).cpu().numpy() > 0.5
            v = valid.squeeze(1).cpu().numpy() > 0.5
            scores.extend(np.asarray(cd.cldice(pred, t, v)).tolist())
            fg_px += int((pred & v).sum())
            valid_px += int(v.sum())

    return {"cldice": float(np.mean(scores)) if scores else 0.0,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "n_tiles": len(scores),
            # predicted foreground fraction of the analysis region: the
            # Mean Teacher collapse guard (Phase 9) watches it
            "fg_frac": fg_px / valid_px if valid_px else 0.0}


def validation_pq(model, image_ids, interim_dir: Path, cfg: dict, device,
                  decode=(0.5, 400, 0.85), limit: int | None = None,
                  channels=None) -> dict:
    """Full-disk pooled PQ over the validation images at a FIXED decode.

    Used when `train.select_metric` is `pq` (Phase 7-2, arm13): a
    component-aware loss needs a component-aware stop / checkpoint signal,
    and tile-level clDice cannot see instance counts. Pooled, not averaged:
    the benchmark number is pooled and per-image PQ is noisy on disks with
    one or two filaments. Costs one stitched inference per val image per
    epoch (~1/5 of the test evaluation).
    """
    tile = cfg["data"]["tile_size"]
    overlap = cfg["data"]["overlap"]
    amp = cfg["train"].get("amp", True)
    tau = float(cfg["eval"].get("instance_tau", inst.DEFAULT_TAU))
    th, ma, mp = float(decode[0]), int(decode[1]), float(decode[2])
    ids = sorted(image_ids)[:limit] if limit else sorted(image_ids)
    tp = fp = fn = 0
    iou_sum = 0.0
    for iid in ids:
        stem = iid.partition("-")[2]
        d = load_interim(Path(interim_dir) / f"{stem}.npz")
        prob = predict_full_disk(model, stack_input(d["img"], channels, stem),
                                 d["valid"], tile, overlap, device, amp=amp)
        pred = decode_components(prob, d["valid"], th, ma, mp)
        lab_p, n_p = inst.label_instances(pred)
        lab_t, n_t = inst.label_instances(d["mask"] & d["valid"])
        pairs = inst.match(inst.iou_matrix(lab_p, n_p, lab_t, n_t), tau)
        tp += len(pairs)
        fp += n_p - len(pairs)
        fn += n_t - len(pairs)
        iou_sum += float(sum(v for _, _, v in pairs))
    denom = tp + 0.5 * fp + 0.5 * fn
    return {"pq": iou_sum / denom if denom else 0.0, "tp": tp, "fp": fp,
            "fn": fn, "iou_sum": iou_sum, "n_images": len(ids),
            "decode": [th, ma, mp]}
