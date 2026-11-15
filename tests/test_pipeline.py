"""Unit tests of the pipeline pieces that have an exact expected answer.

    python -m tests.test_pipeline          (from the repository root; no GPU, no data needed)
"""
from __future__ import annotations

import numpy as np

from pipeline import rle, scorer, treedecode as td, tta


def test_rle_round_trip():
    rng = np.random.default_rng(0)
    m = np.zeros((2048, 2048), dtype=bool)
    m[100:140, 300:900] = True
    m[rng.integers(0, 2048, 500), rng.integers(0, 2048, 500)] = True
    s = rle.encode(m)
    assert np.array_equal(rle.decode(s), m)
    # the pure-Python coder and pycocotools write the same string
    assert rle.runs_to_string(rle.runs_from_mask(m)) == s


def test_views_invert():
    a = np.random.default_rng(1).random((3, 8, 8))
    for v in tta.VIEWS:
        assert np.array_equal(tta.inverse_view(tta.forward_view(a, v), v), a)
    assert len({tta.forward_view(a, v).tobytes() for v in tta.VIEWS}) == 8      # eight distinct views


def _toy():
    """Two blobs joined by a faint bridge, plus a faint isolated blob."""
    p = np.zeros((2048, 2048), dtype=np.float32)
    p[500:520, 500:600] = 0.95
    p[500:520, 640:740] = 0.90
    p[505:515, 600:640] = 0.30           # bridge: one component below 0.30, two above
    p[900:915, 900:960] = 0.42           # never reaches the 0.45 level
    valid = np.ones_like(p, dtype=bool)
    img = -p * 3
    return p, valid, img


def test_tree_structure_and_selection():
    p, valid, img = _toy()
    t = td.build_tree(p, valid, img)
    lv = lambda x: td.THR.index(x)      # noqa: E731
    assert (t["level"] == lv(0.30)).sum() == 2 and (t["level"] == lv(0.45)).sum() == 2
    assert (t["level"] == lv(0.40)).sum() == 3            # the faint blob exists up to 0.40
    # every node lies inside its parent: areas shrink along the tree, levels rise by one
    for i, j in enumerate(t["parent"]):
        if j >= 0:
            assert t["area"][i] <= t["area"][j] and t["level"][i] == t["level"][j] + 1
    # selection never returns a node together with one of its ancestors, whatever the scores
    rng = np.random.default_rng(2)
    for _ in range(50):
        q = rng.random(len(t["area"]))
        ch = td.select(t["parent"], q, 0.5)
        anc = set()
        for i in ch:
            j = t["parent"][i]
            while j >= 0:
                anc.add(int(j))
                j = t["parent"][j]
        assert not anc & set(ch.tolist())
        lab, n = td.render(t, ch)
        assert n == len(ch) and np.array_equal(np.bincount(lab.ravel(), minlength=n + 1)[1:], t["area"][ch])
    # exhaustive check of the optimum on this small forest
    q = rng.random(len(t["area"]))
    best = td.select(t["parent"], q, 0.3)
    val = lambda ch: sum(q[i] - 0.3 for i in ch)          # noqa: E731
    n = len(q)
    top = -1.0
    for mask in range(1 << n) if n <= 16 else []:
        ch = [i for i in range(n) if mask >> i & 1]
        ok = True
        for i in ch:
            j = t["parent"][i]
            while j >= 0 and ok:
                ok = j not in ch
                j = t["parent"][j]
        if ok:
            top = max(top, val(ch))
    if n <= 16:
        assert abs(val(best) - top) < 1e-9


def test_scorer_counts():
    import pandas as pd
    a = np.zeros((2048, 2048), dtype=bool); a[10:50, 10:110] = True          # GT 1
    b = np.zeros_like(a); b[200:240, 10:110] = True                         # GT 2 (missed)
    p1 = np.zeros_like(a); p1[10:50, 10:90] = True                          # IoU 0.8 with GT 1
    p2 = np.zeros_like(a); p2[600:640, 600:700] = True                      # false positive
    gt = pd.DataFrame({"filament_id": ["0101-img_0", "0101-img_1"], "segmentation_rle": [rle.encode(a), rle.encode(b)]})
    pr = pd.DataFrame({"filament_id": ["img_0", "img_1"], "segmentation_rle": [rle.encode(p1), rle.encode(p2)]})
    s = scorer.score(gt, pr)
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 1)
    assert abs(s["pq"] - 0.8 / 2) < 1e-9 and abs(s["sq"] - 0.8) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
