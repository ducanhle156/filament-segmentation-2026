"""Phase 2 Step 11 driver: build and freeze the cross-validation folds.

    python3 scripts/build_splits.py [--group-key obs_window] [--min-gap-days 7]

Writes `splits/folds_v1.json` (frozen — load it, never regenerate) and
`docs/phase2/folds_summary.json`. Refuses to overwrite an existing folds file
unless the regenerated content is byte-identical or --force is given.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.data import splits  # noqa: E402

COCO = "cleaned_magfilo_2026_SEED_42_coco.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--coco", default=COCO)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--group-key", default="obs_window",
                   choices=["obs_date", "obs_window"])
    p.add_argument("--min-gap-days", type=int, default=7)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    root = args.root
    out = args.out or root / "splits" / "folds_v1.json"
    coco = json.loads((root / args.coco).read_text())
    images = sorted(coco["images"], key=lambda im: im["file_name"])
    ids = [im["id"] for im in images]
    n_ann = collections.Counter(a["image_id"] for a in coco["annotations"])
    meta = [{"image_id": im["id"]} for im in images]

    folds = splits.build_folds(meta, n_folds=args.n_folds,
                               group_key=args.group_key, seed=args.seed,
                               val_fraction=args.val_fraction,
                               min_gap_days=args.min_gap_days)
    doc = splits.folds_document(folds, ids, args.group_key, args.seed,
                                args.n_folds, args.val_fraction,
                                args.min_gap_days)

    chk = splits.check_folds(folds, ids, args.group_key, args.min_gap_days)
    print(f"grouping {args.group_key}"
          + (f" (gap {args.min_gap_days}d)" if args.group_key == "obs_window" else "")
          + f": {chk['n_groups']} groups -> {args.n_folds} folds", flush=True)
    print(f"  test fold sizes {chk['fold_sizes']}, spread "
          f"{100 * chk['size_spread']:.2f}% (acceptance: <= 10%)")
    print(f"  no image in two folds: {not chk['duplicated_image_ids']}; "
          f"every image covered: {not chk['uncovered_image_ids']}; "
          f"no group spans two folds: {not chk['groups_spanning_two_folds']}; "
          f"internal: {chk['internal_errors'] or 'ok'}")
    if not chk["passed"]:
        print("ACCEPTANCE FAILED", file=sys.stderr)
        return 1

    leak = {t: splits.temporal_leakage(folds, ids, t) for t in (0, 1, 3, 7, 14)}
    print("  train/test temporal leakage (test images with a train image within N days):")
    for t, v in leak.items():
        print(f"    +-{t:2d} d: {100 * v['fraction_total']:5.1f}%")

    rows = []
    print(f"  {'fold':>4} {'train':>6} {'val':>5} {'test':>5} {'val%':>6} "
          f"{'ff':>3} {'anns/img':>9} {'years':>6} {'sites':>6}")
    for k in sorted(folds):
        tr, va, te = (folds[k][s] for s in ("train", "val", "test"))
        row = {
            "fold": k, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "val_fraction_actual": len(va) / (len(tr) + len(va)),
            "n_filament_free_test": sum(1 for i in te if n_ann[i] == 0),
            "mean_annotations_test": float(np.mean([n_ann[i] for i in te])),
            "years_test": sorted({splits.observation_date(i)[:4] for i in te}),
            "sites_test": sorted({splits.observation_site(i) for i in te}),
        }
        rows.append(row)
        print(f"  {k:>4} {len(tr):>6} {len(va):>5} {len(te):>5} "
              f"{100 * row['val_fraction_actual']:>5.1f}% "
              f"{row['n_filament_free_test']:>3} "
              f"{row['mean_annotations_test']:>9.2f} "
              f"{len(row['years_test']):>6} {len(row['sites_test']):>6}")

    payload = json.dumps(doc, sort_keys=True, separators=(",", ": "), indent=1) + "\n"
    if out.exists() and not args.force:
        if out.read_text() == payload:
            print(f"  {out} already frozen and byte-identical — left untouched")
        else:
            print(f"REFUSING to overwrite existing {out} with different content "
                  f"(rule 3: splits are frozen). Pass --force only to create a "
                  f"new version deliberately.", file=sys.stderr)
            return 1
    else:
        splits.write_folds(doc, out)
        print(f"  wrote {out}")

    # regeneration must be byte-identical: no timestamps, stable ordering
    again = splits.build_folds(meta, n_folds=args.n_folds,
                               group_key=args.group_key, seed=args.seed,
                               val_fraction=args.val_fraction,
                               min_gap_days=args.min_gap_days)
    again_doc = splits.folds_document(again, ids, args.group_key, args.seed,
                                      args.n_folds, args.val_fraction,
                                      args.min_gap_days)
    same = (json.dumps(again_doc, sort_keys=True, separators=(",", ": "),
                       indent=1) + "\n") == payload
    print(f"  regeneration byte-identical: {same}")
    if not same:
        return 1

    summary = {"acceptance": chk,
               "temporal_leakage": {str(t): v for t, v in leak.items()},
               "per_fold": rows}
    sd = root / "docs" / "phase2"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "folds_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"  wrote {sd / 'folds_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
