"""Solar filament instance segmentation: test image -> submission CSV.

    preprocess.py   JPEG -> limb-fitted, flattened, z-scored disk + analysis mask
    model.py        SAM 2.1 Hiera-B+ / LoRA / FPN network and its stored weights
    tta.py          flip / rotation test-time augmentation
    inference.py    tiled full-disk probability map, averaged over views and fold models
    treedecode.py   candidate components of the probability map and their features
    keep_rule.py    learned keep rule: which candidates become filaments
    rle.py          COCO run-length encoding of the masks
    scorer.py       the organisers' self-evaluation metric (PQ, SQ, RQ, Dice, match counts)
    run.py          the whole chain, as a function and as a command
"""

import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "third_party")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
