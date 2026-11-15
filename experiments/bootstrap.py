"""Paired bootstrap over disks: is variant B better than variant A on local validation?

Reads the candidate file, the cross-fitted scores and the per-fold taus that
tree_experiment.py --stage fit stored; "baseline" is the submitted decode
rebuilt as a node selection (level-0.45 components that contain a level-0.50
component with area >= 400 px and mean probability >= 0.85) and must
reproduce 4858/3322/3341.

  python kaggle/try_error/bootstrap.py --pairs kag13/baseline:kag13/flat@0.45 kag13/flat@0.45:tta8/flat@0.45
(the tree variants of the first session were bootstrapped from fit_kag13_variants.json -> bootstrap_kag13.json)
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np

import lib
import treedecode as td
import tree_experiment as te


def baseline_nodes(r: dict) -> np.ndarray:
    l45, l50 = td.THR.index(0.45), td.THR.index(0.50)
    seed = (r["level"] == l50) & (r["area"] >= lib.BASE["min_area"]) & (r["X"][:, td.FEATS.index("mean_p")] >= lib.BASE["min_prob"])
    par = np.unique(r["parent"][seed])
    return par[(par >= 0)] if len(par) else par


def per_disk(disks, chooser) -> np.ndarray:
    """(n_disks, 4): sum IoU of hits, TP, FP, FN (pooled over the disk's records)."""
    out = np.zeros((len(disks), 4))
    for i, r in enumerate(disks):
        m = lib.score_rows(te.rows_for(r, chooser(r)))
        out[i] = (m["sq"] * m["tp"], m["tp"], m["fp"], m["fn"])
    return out


def pq(c: np.ndarray) -> np.ndarray:
    return c[..., 0] / (c[..., 1] + 0.5 * c[..., 2] + 0.5 * c[..., 3])


def counts_for(spec: str, cache: dict) -> np.ndarray:
    """spec = "<tag>/<variant>" (e.g. "kag13/flat@0.45", "tta8/flat@0.45", "kag13/baseline")."""
    tag, v = spec.split("/")
    if tag not in cache:
        cache[tag] = (sorted(pickle.load(open(te.RES / f"tree_{tag}.pkl", "rb")), key=lambda r: r["stem"]),
                      pickle.load(open(te.RES / f"q_oof_{tag}.pkl", "rb")),
                      json.loads((te.RES / f"fit_{tag}.json").read_text()))
    disks, q, fit = cache[tag]
    if v == "baseline":
        return per_disk(disks, baseline_nodes)
    tau = {int(k): x for k, x in fit[v]["tau"].items()}
    return per_disk(disks, lambda r: td.select(r["parent"], q[r["stem"]], tau[r["fold"]], te.allowed_mask(r, v)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="A:B with A, B = <tag>/<variant>; reports B - A")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", default="bootstrap_pairs.json")
    a = ap.parse_args()
    cache, counts = {}, {}
    for spec in sorted({x for p in a.pairs for x in p.split(":")}):
        counts[spec] = counts_for(spec, cache)
        t = counts[spec].sum(0)
        print(f"{spec:22s} PQ {pq(t):.4f}  TP/FP/FN {int(t[1])}/{int(t[2])}/{int(t[3])}")
    n = len(next(iter(counts.values())))
    idx = np.random.default_rng(0).integers(0, n, size=(a.n, n))
    out_path = te.RES / a.out
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    for p in a.pairs:
        x, y = p.split(":")
        d = pq(counts[y][idx].sum(1)) - pq(counts[x][idx].sum(1))
        res[p] = {"delta": float(pq(counts[y].sum(0)) - pq(counts[x].sum(0))), "sd": float(d.std()),
                  "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))], "p_gain": float((d > 0).mean())}
        print(f"{y} - {x}: {res[p]['delta']:+.4f}  SD {res[p]['sd']:.4f}  "
              f"95% CI [{res[p]['ci95'][0]:+.4f}, {res[p]['ci95'][1]:+.4f}]  P(gain>0) {res[p]['p_gain']:.3f}")
    out_path.write_text(json.dumps(res, indent=1))
