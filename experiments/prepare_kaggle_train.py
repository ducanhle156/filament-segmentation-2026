"""Preprocess the 707 competition TRAIN JPEGs exactly as `kaggle/src/prepare_test.py`
preprocesses the test JPEGs (Phase 1 contract: limb fit, limb-darkening removal,
z-score on the 0.95 R analysis disk) -> kaggle/try_error/data/interim_k/<stem>.npz.

Why: the networks were trained on the project cache `data/interim`, built from an
RGB re-encode of the same frames. The submission notebook must start from the
competition's own JPEGs, so local validation has to be run on that input too
(experiment "input = competition JPEG" in EXPERIMENTS.md). No label is stored.

    python kaggle/try_error/prepare_kaggle_train.py --workers 8
"""
from __future__ import annotations

import argparse
from multiprocessing import get_context

import numpy as np
import yaml

import lib
from paths import H, KAGGLE_TRAIN_IMAGES, ROOT, W
from common.data import preprocess

OUT = lib.HERE / "data" / "interim_k"
CFG = yaml.safe_load((ROOT / "common" / "configs" / "preprocess_v1.yaml").read_text())


def one(stem: str) -> dict:
    lim = CFG["limb"]
    r = preprocess.preprocess_image(
        KAGGLE_TRAIN_IMAGES / f"{stem}.jpeg", [], H, W, limb_method=lim["method"], n_bins=lim["n_bins"],
        smooth_bins=lim["smooth_bins"], floor_frac=lim["floor_frac"], flat_bins=lim["flat_bins"],
        analysis_scale=lim["disk"]["analysis_scale"])
    preprocess.save_interim(OUT / f"{stem}.npz", r["img"], r["valid"], r["mask"],
                            dict(r["stats"], image_id=stem, source="kaggle train_images"))
    d = lib.load_disk(stem)
    both = d["valid"] | r["valid"]
    return {"stem": stem, "valid_xor_px": int((d["valid"] ^ r["valid"]).sum()),
            "mean_abs": float(np.abs(d["img"] - r["img"])[both].mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stems = sorted({s for _, s in lib.jobs()})
    with get_context("spawn").Pool(a.workers) as pool:
        rows = pool.map(one, stems, chunksize=4)
    print(f"{len(rows)} disks -> {OUT}; vs project cache: mean |dimg| "
          f"{np.mean([r['mean_abs'] for r in rows]):.4f} z-units, largest analysis-disk difference "
          f"{max(r['valid_xor_px'] for r in rows)} px")
