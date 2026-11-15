"""Full-disk inference with flip / rotation test-time augmentation.

Runs every fold model on its inner-test disks with the D4 views of tta.py
(4 flips, then the same after a 90-degree rotation) and stores the averages.

  python kaggle/try_error/infer.py --folds 0 1 2 3 4        # inner-test disks of each fold
writes kaggle/results/ladder/<arm>_tta4 and _tta8 /seed0/fold<F>/probs/<stem>.npz
(key o0.50, uint8), the layout lib.load_prob reads. `--source kaggle` reads the
disks preprocessed from the competition JPEGs (prepare_kaggle_train.py) instead
of the project cache and writes <arm>_k_tta1 / _k_tta4 / _k_tta8.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import lib
from lib import LADDER

from tta import predict_views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=lib.ARM)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--overlap", type=float, default=0.50)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--source", choices=["cache", "kaggle"], default="cache")
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    tag, ns = ("_k", (1, 4, 8)) if a.source == "kaggle" else ("", (4, 8))
    import torch
    from model_io import load_run
    from paths import RUNS
    from common.data.channels import stack_input
    t0 = time.time()
    for f in a.folds:
        stems = [s for ff, s in lib.jobs() if ff == f][: a.limit]
        if a.reverse:            # a second job on the same fold works from the other end
            stems = stems[::-1]
        model, cfg, _, channels = load_run(RUNS / a.arm / "seed0" / f"fold{f}")
        tile, amp = cfg["data"]["tile_size"], cfg["train"].get("amp", True)
        device = next(model.parameters()).device
        dirs = {n: LADDER / f"{a.arm}{tag}_tta{n}" / "seed0" / f"fold{f}" / "probs" for n in ns}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(stems):
            if all((d / f"{s}.npz").exists() for d in dirs.values()):
                continue
            disk = lib.load_disk(s, a.source)
            views = predict_views(model, stack_input(disk["img"], channels, s), disk["valid"], tile,
                                  a.overlap, device, amp)
            if i == 0 and a.source == "cache":   # the identity view must equal the stored single-view map
                ref = lib.load_prob(f, s, a.overlap, a.arm)
                print(f"  fold {f} check: max |identity view - stored map| = {np.abs(views[0] - ref).max():.4f}", flush=True)
            for n, d in dirs.items():
                np.savez_compressed(d / f"{s}.npz", **{f"o{a.overlap:.2f}":
                                                       (np.mean(views[:n], axis=0) * 255).round().astype(np.uint8)})
            if (i + 1) % 20 == 0:
                print(f"  fold {f}: {i + 1}/{len(stems)} ({time.time() - t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
