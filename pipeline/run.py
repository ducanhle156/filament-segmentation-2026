"""The whole pipeline: a folder of test JPEGs -> submission CSV.

    python -m pipeline.run --test-dir <folder with *.jpeg> --out submission.csv

Per image: preprocess -> probability map (fold models x views) -> keep rule
-> one CSV row per filament (`<image id>_<k>`, COCO RLE counts). Nothing
but the image is read: no annotation, no neighbouring frame, and the file
name is used only as the row id.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import rle, settings
from .inference import predict_probability
from .keep_rule import KeepRule
from .model import load_fold_model
from .preprocess import preprocess_jpeg


def segment_image(path: Path, models: list, cfg: dict, keep: KeepRule, n_views: int = settings.N_VIEWS):
    """-> (label map, number of filaments, probability map, preprocessed image, analysis mask)."""
    img, valid = preprocess_jpeg(path)
    prob = predict_probability(models, cfg, img, valid, n_views)
    lab, n, _ = keep.decode(prob, valid, img)
    return lab, n, prob, img, valid


def rows_for(stem: str, lab: np.ndarray, n: int) -> list[dict]:
    return [{"filament_id": f"{stem}_{k}", "segmentation_rle": rle.encode(lab == k)} for k in range(1, n + 1)]


def run(test_dir: Path, out_csv: Path, folds=settings.FOLDS, n_views: int = settings.N_VIEWS,
        limit: int | None = None, save_prob_dir: Path | None = None) -> pd.DataFrame:
    t0 = time.time()
    paths = sorted(Path(test_dir).glob("*.jpeg"))[:limit]
    loaded = [load_fold_model(f) for f in folds]
    models, cfg = [m for m, _ in loaded], loaded[0][1]
    keep = KeepRule()
    rows = []
    for i, p in enumerate(paths):
        lab, n, prob, _, _ = segment_image(p, models, cfg, keep, n_views)
        rows += rows_for(p.stem, lab, n)
        if save_prob_dir is not None:
            Path(save_prob_dir).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(Path(save_prob_dir) / f"{p.stem}.npz", prob=(prob * 255).round().astype(np.uint8))
        if (i + 1) % 20 == 0 or i + 1 == len(paths):
            print(f"  {i + 1}/{len(paths)} images, {len(rows)} filaments ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}: {len(df)} filaments on {len(paths)} images "
          f"({(time.time() - t0) / max(len(paths), 1):.1f} s per image)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    ap.add_argument("--folds", nargs="+", type=int, default=list(settings.FOLDS))
    ap.add_argument("--views", type=int, default=settings.N_VIEWS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-prob-dir", type=Path, default=None)
    a = ap.parse_args()
    run(a.test_dir, a.out, tuple(a.folds), a.views, a.limit, a.save_prob_dir)
