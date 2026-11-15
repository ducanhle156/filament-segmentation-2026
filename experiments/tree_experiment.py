"""Experiments 1-2: learned keep rule and component-tree decoding, on local validation.

  --stage extract   per inner-test disk: build the component tree of its
                    out-of-fold probability map, node features, and the
                    intersection of every node with every GT filament of
                    every annotator record -> results/tree_<tag>.pkl
  --stage fit       nested cross-fit. For held-out fold F: the regressor
                    is fitted on the other four folds; tau is chosen on
                    those four using predictions from an inner cross-fit
                    (so neither the regressor nor tau ever sees fold F).
                    Scores variants, appends rows to EXPERIMENTS.md, and
                    with --final fits the deployment model on all 707.

  python kaggle/try_error/tree_experiment.py --stage extract --workers 8
  python kaggle/try_error/tree_experiment.py --stage fit
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from multiprocessing import get_context

import numpy as np

import lib
import treedecode as td

RES = lib.HERE / "results"
TAUS = [round(0.10 + 0.01 * k, 2) for k in range(26)]       # 0.10 .. 0.35


def probs_for(fold: int, stem: str, arms: list[str]) -> np.ndarray:
    return np.mean([lib.load_prob(fold, stem, arm=a) for a in arms], axis=0)


def extract_one(job):
    fold, stem, arms, source = job
    d = lib.load_disk(stem, source)
    tree = td.build_tree(probs_for(fold, stem, arms), d["valid"], d["img"])
    n = len(tree["area"])
    recs = {}
    for rec, gts in lib.gt_records(stem).items():
        inter = np.zeros((len(gts), n), dtype=np.float32)
        pos = np.searchsorted(tree["idx0"], np.concatenate(gts) if gts else np.zeros(0, np.int64))
        # GT pixels below the lowest level touch no node
        cuts = np.cumsum([0] + [len(g) for g in gts])
        cl = np.minimum(pos, max(len(tree["idx0"]) - 1, 0))
        flat_ok = (tree["idx0"][cl] == np.concatenate(gts)) if len(pos) and len(tree["idx0"]) else np.zeros(len(pos), bool)
        for lv in np.unique(tree["level"]):
            nodes = np.flatnonzero(tree["level"] == lv)
            lut = np.full(int(tree["labs"][lv].max()) + 1, -1, dtype=np.int64)
            lut[tree["cid"][nodes]] = nodes
            for g in range(len(gts)):
                s = slice(cuts[g], cuts[g + 1])
                ids = lut[tree["labs"][lv][cl[s][flat_ok[s]]]]
                ids = ids[ids >= 0]
                if len(ids):
                    inter[g] += np.bincount(ids, minlength=n)
        recs[rec] = {"inter": inter, "gt_area": np.array([len(g) for g in gts], dtype=np.float64)}
    return {"fold": fold, "stem": stem, "X": tree["X"], "parent": tree["parent"], "level": tree["level"],
            "area": tree["area"], "recs": recs}


def stage_extract(a):
    t0 = time.time()
    jobs = [(f, s, a.arms, a.source) for f, s in lib.jobs()[: a.limit]]
    out = []
    with get_context("spawn").Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(extract_one, jobs, chunksize=2)):
            out.append(r)
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    out.sort(key=lambda r: r["stem"])
    RES.mkdir(exist_ok=True)
    pickle.dump(out, open(RES / f"tree_{a.tag}.pkl", "wb"), protocol=4)
    print(f"{len(out)} disks, {sum(len(r['area']) for r in out)} nodes -> tree_{a.tag}.pkl ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ fit / score

def target(r: dict) -> np.ndarray:
    """Mean over the disk's annotator records of IoU x [IoU > 0.5] (what a node earns)."""
    y = np.zeros(len(r["area"]))
    for v in r["recs"].values():
        if not len(v["gt_area"]):
            continue
        iou = v["inter"] / (v["gt_area"][:, None] + r["area"][None, :] - v["inter"])
        y += np.where(iou > lib.IOU_T, iou, 0).sum(0)
    return y / max(len(r["recs"]), 1)


def rows_for(r: dict, chosen: np.ndarray) -> list[dict]:
    rows = []
    for v in r["recs"].values():
        inter = v["inter"][:, chosen].astype(np.float64)
        ga, pa = v["gt_area"][:, None], r["area"][None, chosen]
        rows.append({"iou_matrix": inter / np.maximum(ga + pa - inter, 1),
                     "dice_matrix": 2 * inter / np.maximum(ga + pa, 1),
                     "n_gt": inter.shape[0], "n_pred": len(chosen)})
    return rows


def allowed_mask(r: dict, variant: str) -> np.ndarray | None:
    if variant == "tree":
        return None
    kind, spec = variant.split("@")
    if kind == "tree":                                        # "tree@0.4-0.6": levels in that range only
        lo, hi = (float(x) for x in spec.split("-"))
        t = np.asarray(td.THR)[r["level"]]
        return (t >= lo - 1e-9) & (t <= hi + 1e-9)
    return r["level"] == td.THR.index(float(spec))            # "flat@0.45": one level only


def score(disks: list[dict], q: dict, tau: float, variant: str) -> dict:
    rows = []
    for r in disks:
        rows += rows_for(r, td.select(r["parent"], q[r["stem"]], tau, allowed_mask(r, variant)))
    return lib.score_rows(rows)


def fit_model(disks: list[dict], seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = np.concatenate([r["X"] for r in disks])
    y = np.concatenate([target(r) for r in disks])
    w = np.concatenate([np.full(len(r["area"]), len(r["recs"])) for r in disks])   # the scorer pools records
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                      l2_regularization=1.0, random_state=seed)
    return m.fit(X, y, sample_weight=w)


def predict(m, disks):
    return {r["stem"]: np.clip(m.predict(r["X"]), 0, 1) if len(r["area"]) else np.zeros(0) for r in disks}


def best_tau(disks, q, variant):
    s = [(score(disks, q, t, variant)["pq"], t) for t in TAUS]
    return max(s)[1]


def stage_fit(a):
    t0 = time.time()
    disks = pickle.load(open(RES / f"tree_{a.tag}.pkl", "rb"))
    folds = sorted({r["fold"] for r in disks})
    by = {f: [r for r in disks if r["fold"] == f] for f in folds}
    variants = a.variants
    q_out, taus = {}, {v: {} for v in variants}
    for F in folds:
        outer = [f for f in folds if f != F]
        q_in = {}
        for G in outer:                                           # inner cross-fit: tau never sees F
            m = fit_model([r for f in outer if f != G for r in by[f]])
            q_in.update(predict(m, by[G]))
        tr = [r for f in outer for r in by[f]]
        for v in variants:
            taus[v][F] = best_tau(tr, q_in, v)
        q_out.update(predict(fit_model(tr), by[F]))
        print(f"fold {F}: tau {[taus[v][F] for v in variants]} ({time.time() - t0:.0f}s)", flush=True)
    out = {}
    for v in variants:
        rows = []
        for F in folds:
            for r in by[F]:
                rows += rows_for(r, td.select(r["parent"], q_out[r["stem"]], taus[v][F], allowed_mask(r, v)))
        out[v] = lib.score_rows(rows)
        out[v]["per_fold"] = {int(F): score(by[F], q_out, taus[v][F], v)["pq"] for F in folds}
        out[v]["tau"] = {int(k): float(x) for k, x in taus[v].items()}
        print(f"{v:10s} {lib.fmt(out[v])}  folds {[round(x, 3) for x in out[v]['per_fold'].values()]}", flush=True)
    # tau sensitivity of the tree with cross-fitted q (in-sample in tau only; for the doc)
    out["tau_curve"] = {v: {str(t): score(disks, q_out, t, v)["pq"] for t in TAUS} for v in variants}
    (RES / f"fit_{a.tag}{a.suffix}.json").write_text(json.dumps(out, indent=1))
    pickle.dump(q_out, open(RES / f"q_oof_{a.tag}.pkl", "wb"))
    if a.final:
        v = a.final
        m = fit_model(disks)
        tau = float(np.median(list(taus[v].values())))
        pickle.dump({"model": m, "tau": tau, "variant": v, "feats": td.FEATS, "thr": td.THR,
                     "min_area": td.MIN_AREA, "tag": a.tag, "cv": {k: out[v][k] for k in ("pq", "sq", "rq", "tp", "fp", "fn")}},
                    open(lib.HERE / "weights" / f"keep_{a.tag}.pkl", "wb"))
        print(f"deployment model ({v}): all {len(disks)} disks, tau {tau}")
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["extract", "fit"], required=True)
    ap.add_argument("--arms", nargs="+", default=[lib.ARM], help="arms whose inner-test maps are averaged")
    ap.add_argument("--tag", default="kag13")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--final", default=None, help="variant to deploy: fit on all disks, tau = median of the fold taus")
    ap.add_argument("--variants", nargs="+", default=["flat@0.45", "flat@0.5", "tree"])
    ap.add_argument("--suffix", default="")
    ap.add_argument("--source", choices=["cache", "kaggle"], default="cache", help="which preprocessed disks the maps belong to")
    a = ap.parse_args()
    (lib.HERE / "weights").mkdir(exist_ok=True)
    stage_extract(a) if a.stage == "extract" else stage_fit(a)
