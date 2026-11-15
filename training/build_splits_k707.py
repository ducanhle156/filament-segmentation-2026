"""Five grouped folds over the 707 Kaggle train images only.

Same generator and settings as the benchmark's `splits/folds_v1.json`
(`common/data/splits.build_folds`: obs_window groups, 7-day gap, seed 42,
val 0.15), restricted to the competition train stems. Image ids are the
project's seed-42 record ids, so the frozen tile cache can be filtered by id
without rebuilding. The 180 test stems never appear.

    .venv/bin/python kaggle/src/build_splits.py [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (KAGGLE_TRAIN_JSON, SPLITS, kaggle_test_stems,  # noqa: E402
                    kaggle_train_stems, our_record_ids, stem_of)
from common.data import splits  # noqa: E402  (project module, via sys.path ROOT)

GROUP_KEY, MIN_GAP, N_FOLDS, SEED, VAL_FRAC = "obs_window", 7, 5, 42, 0.15


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", type=Path, default=SPLITS)
    args = ap.parse_args()
    if args.out.exists() and not args.force:
        print(f"{args.out} exists; --force to regenerate")
        return 0

    train_stems = kaggle_train_stems()
    test_stems = set(kaggle_test_stems())
    rec = our_record_ids()
    missing = [s for s in train_stems if s not in rec]
    if missing:
        raise SystemExit(f"{len(missing)} Kaggle train stems not in our COCO: {missing[:5]}")
    # every chosen record must be a record the competition provides
    k = json.loads(KAGGLE_TRAIN_JSON.read_text())
    k_ids = {im["id"] for im in k["images"]}
    not_in_kaggle = [rec[s] for s in train_stems if rec[s] not in k_ids]
    if not_in_kaggle:
        raise SystemExit(f"{len(not_in_kaggle)} of our record ids are not Kaggle records")
    ids = sorted(rec[s] for s in train_stems)
    assert not (set(train_stems) & test_stems)

    folds = splits.build_folds([{"image_id": i} for i in ids], n_folds=N_FOLDS,
                               group_key=GROUP_KEY, seed=SEED,
                               val_fraction=VAL_FRAC, min_gap_days=MIN_GAP)
    doc = splits.folds_document(folds, ids, GROUP_KEY, SEED, N_FOLDS, VAL_FRAC,
                                MIN_GAP, version="k707_v1")
    doc["source"] = {"kaggle_train_json": str(KAGGLE_TRAIN_JSON),
                     "record_choice": "project seed-42 annotator (cleaned_magfilo_2026_SEED_42_coco.json)",
                     "n_kaggle_test_stems_excluded": len(test_stems)}
    chk = splits.check_folds(folds, ids, GROUP_KEY, MIN_GAP)
    print(f"{len(ids)} images, {chk['n_groups']} groups -> {N_FOLDS} folds; "
          f"test sizes {chk['fold_sizes']} spread {100 * chk['size_spread']:.1f}%; "
          f"passed={chk['passed']}")
    for kf in sorted(folds):
        f = folds[kf]
        leak = {stem_of(i) for s in ("train", "val", "test") for i in f[s]} & test_stems
        print(f"  fold {kf}: train {len(f['train'])} val {len(f['val'])} "
              f"test {len(f['test'])}  kaggle-test stems inside: {len(leak)}")
        assert not leak
    if not chk["passed"]:
        print("ACCEPTANCE FAILED", chk)
        return 1
    splits.write_folds(doc, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
