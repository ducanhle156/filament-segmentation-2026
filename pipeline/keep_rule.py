"""Step 3 - from the probability map to filament instances.

Candidates are the 8-connected components of {p >= 0.45} on the analysis
disk. The submitted baseline kept a candidate when it passed three global
thresholds (area, mean probability, seed level). Here a gradient-boosted
regressor looks at 22 features of the candidate (size, elongation, width,
confidence statistics, darkness against its surroundings, distance from disk
centre, how it splits at higher levels and what it joins at lower ones) and
predicts q = the IoU the candidate is expected to earn as a true positive
(0 if it would not match an annotation). A candidate is kept when q > tau.

Why a cut on expected IoU: PQ = sum IoU / (TP + FP/2 + FN/2). An extra
prediction adds its IoU to the numerator when it matches and 1/2 to the
denominator either way, so it raises PQ exactly when its expected IoU is
above PQ/2 (about 0.2). tau was tuned by nested cross-validation.

The regressor was fitted on out-of-fold probability maps of the 707
training disks. Kept components are pixel-disjoint by construction.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from . import settings, treedecode as td

ROOT = Path(__file__).resolve().parents[1]


class KeepRule:
    def __init__(self, path: Path | None = None):
        d = pickle.load(open(path or ROOT / settings.KEEP_FILE, "rb"))
        self.model, self.tau, self.variant, self.cv = d["model"], float(d["tau"]), d["variant"], d.get("cv")
        assert tuple(d["feats"]) == tuple(td.FEATS) and tuple(d["thr"]) == tuple(td.THR), "feature contract changed"
        assert self.variant == f"flat@{settings.KEEP_LEVEL}", self.variant
        self.level = td.THR.index(settings.KEEP_LEVEL)

    def decode(self, prob: np.ndarray, valid: np.ndarray, img: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
        """-> (label map int32, 0 = background and 1..n = filaments; n; predicted q of the kept filaments)."""
        tree = td.build_tree(prob, valid, img)
        if not len(tree["area"]):
            return np.zeros(prob.shape, dtype=np.int32), 0, np.zeros(0)
        q = np.clip(self.model.predict(tree["X"]), 0, 1)
        chosen = td.select(tree["parent"], q, self.tau, tree["level"] == self.level)
        lab, n = td.render(tree, chosen)
        return lab, n, q[chosen]
