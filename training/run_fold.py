"""Single-job entry point: one (arm, seed, fold). Phase 2, Steps 13-14.

    python3 scripts/run_fold.py --config models/unet/configs/arm01_unet.yaml \
        --fold 0 --seed 0 [--smoke]

`--smoke` runs the Step 14 rehearsal: 20 images, 50 iterations and a 2-layer
model, so the whole path — cache, augmentation, loss, stitching, metrics — is
exercised in a minute before anything real is submitted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.utils.config import load_config  # noqa: E402


def git_commit(root: Path):
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None,
                   help="arm YAML; omitted means base.yaml alone")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--figure", type=Path, default=None,
                   help="also copy the detail figure to this path")
    p.add_argument("--no-figures", action="store_true",
                   help="skip the run's figures (they are on by default)")
    p.add_argument("--set", nargs="*", default=[],
                   help="dotted overrides, e.g. --set train.epochs=1")
    args = p.parse_args()

    overrides: dict = {}
    for item in args.set:
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        cur = overrides
        parts = key.split(".")
        for k in parts[:-1]:
            cur = cur.setdefault(k, {})
        cur[parts[-1]] = value

    cfg = load_config(args.config, args.root, overrides)
    cfg["seed"] = args.seed
    cfg["_git_commit"] = git_commit(args.root)

    if args.smoke:
        from common.engine.registry import get_trainer as _gt
        if _gt(cfg["model"].get("type", "monolithic")) is None:
            # the model is replaced, not the pipeline: everything else is real
            cfg["model"] = {"type": "monolithic", "arch": {"name": "tiny_cnn"}}
        # a family with its own trainer (YOLO) keeps its model: the fixture
        # would not exercise that trainer at all
        cfg["results_dir"] = "models/_harness/results"
        cfg["arm_id"] = f"smoke_{cfg['arm_id']}"
        cfg["train"]["num_workers"] = min(cfg["train"].get("num_workers", 4), 4)

    from common.utils.runtime import configure_threads
    n_threads = configure_threads(cfg["train"].get("torch_threads"))

    import common.engine.smoke_model  # noqa: F401  (registers the fixtures)
    from common.engine.registry import get_trainer
    from common.engine.train import train_fold as _shared_train_fold
    train_fold = get_trainer(cfg["model"].get("type", "monolithic")) or _shared_train_fold

    print(f"arm {cfg['arm_id']}  fold {args.fold}  seed {args.seed}"
          f"{'  [smoke]' if args.smoke else ''}", flush=True)
    print(f"  config {args.config or 'base.yaml only'} "
          f"sha {cfg['_config_sha256'][:12]}  torch threads {n_threads}",
          flush=True)

    m = train_fold(cfg, args.fold, args.seed, args.root, smoke=args.smoke)
    n = len(m["per_image"])
    if n:
        import numpy as np
        for k in cfg["eval"]["metrics"]:
            vals = [r[k] for r in m["per_image"] if k in r]
            if vals:
                print(f"  test {k:20s} mean {np.mean(vals):.4f} "
                      f"median {np.median(vals):.4f} (n={len(vals)})")
    print(f"  {m['trainable_parameters']:,} trainable / "
          f"{m['total_parameters']:,} total params, "
          f"{m['wall_clock_s']:.1f}s, peak {m['peak_gpu_memory_gib']:.2f} GiB")

    if not args.no_figures and m["per_image"]:
        # Figures are a deliverable of every run, not an opt-in: a 16-hour job
        # that reports only numbers cannot be sanity-checked by looking. They
        # are also wrapped -- a matplotlib failure must never lose a finished
        # training run, whose metrics.json is already on disk by this point.
        run_dir = (args.root / cfg["results_dir"] / cfg["arm_id"]
                   / f"seed{args.seed}" / f"fold{args.fold}")
        written = []
        try:
            from common.report.run_report import write_run_figures
            written += write_run_figures(run_dir, m)
        except Exception as exc:
            print(f"  ! run figures skipped: {exc}", flush=True)
        try:
            from common.report.run_report import write_qualitative_figures
            written += write_qualitative_figures(run_dir, cfg, m, args.root)
        except Exception as exc:
            print(f"  ! qualitative figures skipped: {exc}", flush=True)
        for path in written:
            print(f"  figure -> {path}")
        if args.figure and written:
            import shutil
            detail = run_dir / "figures" / "prediction_detail.png"
            if detail.exists():
                args.figure.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(detail, args.figure)
                print(f"  figure -> {args.figure}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
