"""Step 2 — diagnose the submitted pipeline before changing anything.

For every inner-test disk (707, grouped folds) decode the stored kag13
probability map with the submitted cell, score it with the official rule
and give every failure a class, per annotator record:

  GT filament without a hit (FN)
    missed      < 20 % of its pixels are covered by any prediction
    fragmented  >= 2 predictions lie mostly inside it and their union would hit (IoU > 0.5)
    merged      its best prediction also swallows another GT filament and would
                hit the union of the GT filaments it covers
    boundary    one partner, right place, IoU <= 0.5 (too thin / too wide / too short)
  prediction without a hit (FP)
    false_pos   < 20 % of its pixels lie on any GT filament of that record
    (otherwise it is the partner of a fragmented / merged / boundary GT)

Oracles answer "how much PQ is each class worth": the class is repaired
perfectly (repaired filaments enter as hits with the pooled mean IoU of
the real hits) and PQ is recomputed.

Outputs: results/diagnosis.json, docs/01_diagnosis.md (tables only; the
reading is written by hand below the marker), figures/failures/*.png.

  python kaggle/try_error/diagnose.py --workers 16
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

import lib

OUTD = lib.HERE
CLASSES = ["missed", "fragmented", "merged", "boundary"]


def classify(o: dict) -> tuple[list[str], list[str], list[dict]]:
    """-> class of every GT row ('hit' or a failure), of every pred column, case list."""
    inter, iou, ga, pa = o["inter"], o["iou_matrix"], o["gt_area"], o["pred_area"]
    n_gt, n_pred = o["n_gt"], o["n_pred"]
    hit = iou > lib.IOU_T
    g_cls = ["hit" if (n_pred and hit[g].any()) else "" for g in range(n_gt)]
    cases = []
    for g in range(n_gt):
        if g_cls[g]:
            continue
        row = inter[g] if n_pred else np.zeros(0)
        cov = row.sum() / max(ga[g], 1)
        info = {"g": g, "cov": float(cov), "best_iou": float(iou[g].max()) if n_pred else 0.0,
                "gt_area": float(ga[g])}
        if cov < 0.2:
            g_cls[g] = "missed"
        else:
            inside = np.flatnonzero(row >= 0.5 * pa)             # predictions mostly inside this GT
            u_iou = 0.0
            if len(inside) >= 2:
                i_u, a_u = row[inside].sum(), pa[inside].sum()
                u_iou = i_u / (ga[g] + a_u - i_u)
            j = int(row.argmax())
            others = np.flatnonzero(inter[:, j] >= 0.25 * ga)     # GT filaments this prediction covers
            m_iou = 0.0
            if len(others) >= 2 and g in others:
                i_m, a_m = inter[others, j].sum(), ga[others].sum()
                m_iou = i_m / (a_m + pa[j] - i_m)
            if u_iou > lib.IOU_T:
                g_cls[g] = "fragmented"
                info["n_frag"] = int(len(inside))
            elif m_iou > lib.IOU_T:
                g_cls[g] = "merged"
                info["n_merged"] = int(len(others))
            else:
                g_cls[g] = "boundary"
                info["pred_over_gt"] = float(pa[j] / max(ga[g], 1))
                info["pred_inside"] = float(row[j] / max(pa[j], 1))   # precision of the partner
                info["gt_covered"] = float(row[j] / max(ga[g], 1))    # recall of the partner
        info["cls"] = g_cls[g]
        cases.append(info)
    p_cls = []
    for j in range(n_pred):
        if n_gt and hit[:, j].any():
            p_cls.append("hit")
        elif n_gt == 0 or inter[:, j].sum() < 0.2 * pa[j]:
            p_cls.append("false_pos")
        else:
            p_cls.append("partner")
    return g_cls, p_cls, cases


def one(job):
    fold, stem = job
    d = lib.load_disk(stem)
    prob = lib.load_prob(fold, stem)
    lab, n = lib.baseline_labels(prob, d["valid"])
    rows, out = [], []
    for rec, gts in lib.gt_records(stem).items():
        o = lib.overlaps(lab, n, gts)
        g_cls, p_cls, cases = classify(o)
        fp_area = [float(o["pred_area"][j]) for j, c in enumerate(p_cls) if c == "false_pos"]
        fp_prob = [float(prob[lab == j + 1].mean()) for j, c in enumerate(p_cls) if c == "false_pos"]
        out.append({"stem": stem, "fold": fold, "record": rec, "g_cls": g_cls, "p_cls": p_cls,
                    "cases": cases, "fp_area": fp_area, "fp_prob": fp_prob,
                    "pixel_dice": lib.pixel_dice(lab, gts)})
        rows.append({k: o[k] for k in ("iou_matrix", "dice_matrix", "n_gt", "n_pred")})
    return rows, out


def oracle(base: dict, counts: dict, fix: str) -> float:
    """PQ if every failure of class `fix` were repaired into a hit at the mean IoU."""
    tp, fp, fn, sq = base["tp"], base["fp"], base["fn"], base["sq"]
    if fix == "false_pos":
        fp -= counts["false_pos"]
    else:
        k = counts[fix]
        tp, fn = tp + k, fn - k
        fp -= counts.get(f"{fix}_partners", 0)
    return sq * tp / (tp + 0.5 * fp + 0.5 * fn)


def panel(recs: list[dict], n_per: int = 4, seed: int = 0):
    """20 failure cases: 4 per class, drawn at random among the larger ones."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.measure import find_contours

    rng = np.random.default_rng(seed)
    pool = {c: [] for c in CLASSES + ["false_pos"]}
    for r in recs:
        for c in r["cases"]:
            if c["gt_area"] >= 600:
                pool[c["cls"]].append((r["stem"], r["fold"], r["record"], "g", c["g"], c))
        for j, c in enumerate(r["p_cls"]):
            if c == "false_pos":
                pool["false_pos"].append((r["stem"], r["fold"], r["record"], "p", j, {}))
    figdir = OUTD / "figures" / "failures"
    figdir.mkdir(parents=True, exist_ok=True)
    picked = []
    fig, axes = plt.subplots(5, n_per, figsize=(3.6 * n_per, 3.6 * 5))
    for ri, cls in enumerate(CLASSES + ["false_pos"]):
        idx = rng.choice(len(pool[cls]), size=min(n_per, len(pool[cls])), replace=False)
        for ci, k in enumerate(idx):
            stem, fold, rec, kind, i, info = pool[cls][int(k)]
            d = lib.load_disk(stem)
            lab, n = lib.baseline_labels(lib.load_prob(fold, stem), d["valid"])
            gts = lib.gt_records(stem)[rec]
            gmap = np.zeros(lib.H * lib.W, dtype=np.int32)
            for q, ix in enumerate(gts):
                gmap[ix] = q + 1
            gmap = gmap.reshape(lib.H, lib.W)
            focus = (gmap == i + 1) if kind == "g" else (lab == i + 1)
            ys, xs = np.nonzero(focus)
            cy, cx = int(ys.mean()), int(xs.mean())
            half = int(max(96, 0.75 * max(np.ptp(ys), np.ptp(xs)) + 48))
            y0, y1 = max(0, cy - half), min(lib.H, cy + half)
            x0, x1 = max(0, cx - half), min(lib.W, cx + half)
            ax = axes[ri, ci]
            ax.imshow(d["img"][y0:y1, x0:x1], cmap="gray")
            for m, col, lw in ((gmap[y0:y1, x0:x1] > 0, "#2ca02c", 1.2), (lab[y0:y1, x0:x1] > 0, "#d62728", 1.0)):
                # contour per instance so that touching instances stay visible
                src = gmap[y0:y1, x0:x1] if col == "#2ca02c" else lab[y0:y1, x0:x1]
                for v in np.unique(src[m]):
                    for c in find_contours((src == v).astype(float), 0.5):
                        ax.plot(c[:, 1], c[:, 0], color=col, lw=lw)
            extra = (f"best IoU {info.get('best_iou', 0):.2f}" if kind == "g" else "no GT in this record")
            ax.set_title(f"{cls} — {stem}\n{extra}", fontsize=8)
            ax.set_xticks([]), ax.set_yticks([])
            picked.append({"class": cls, "stem": stem, "record": rec, "kind": kind, "index": int(i), **info})
        for ci in range(len(idx), n_per):
            axes[ri, ci].axis("off")
    fig.suptitle("20 failure cases of the submitted decode (kag13, inner test). "
                 "Green = annotator record, red = prediction.", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"te01_failure_panel.{ext}", dpi=130)
    (figdir / "te01_failure_panel.json").write_text(json.dumps(picked, indent=1))
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()
    jobs = lib.jobs()[: a.limit]
    rows, recs = [], []
    with get_context("spawn").Pool(a.workers) as pool:
        for k, (r, o) in enumerate(pool.imap_unordered(one, jobs, chunksize=2)):
            rows += r
            recs += o
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    base = lib.score_rows(rows)
    print("baseline:", lib.fmt(base), flush=True)
    if a.limit is None:
        assert (base["tp"], base["fp"], base["fn"]) == (4858, 3322, 3341), "does not reproduce docs/02_results.md"
    counts = {c: 0 for c in CLASSES + ["false_pos", "partner"]}
    for r in recs:
        for c in r["g_cls"] + r["p_cls"]:
            if c in counts:
                counts[c] += 1
    # FP partners of each GT failure class, shared out in proportion (a partner
    # prediction is not tied to one class when several GT filaments touch it)
    non_missed = counts["fragmented"] + counts["merged"] + counts["boundary"]
    for c in ("fragmented", "merged", "boundary"):
        counts[f"{c}_partners"] = int(round(counts["partner"] * counts[c] / max(non_missed, 1)))
    orc = {c: oracle(base, counts, c) for c in CLASSES + ["false_pos"]}
    cases = [c for r in recs for c in r["cases"]]
    bnd = [c for c in cases if c["cls"] == "boundary"]
    q = lambda v: [float(x) for x in np.quantile(v, [0.1, 0.5, 0.9])] if len(v) else []  # noqa: E731
    fp_area = [x for r in recs for x in r["fp_area"]]
    fp_prob = [x for r in recs for x in r["fp_prob"]]
    hits_iou = np.concatenate([r["iou_matrix"][r["iou_matrix"] > lib.IOU_T] for r in rows if r["n_gt"] and r["n_pred"]])
    near = sum(1 for c in cases if 0.4 < c["best_iou"] <= 0.5)
    miss_area = [c["gt_area"] for c in cases if c["cls"] == "missed"]
    miss_small = float(np.mean([x < 400 for x in miss_area])) if miss_area else 0.0
    out = {"baseline": base, "counts": counts, "oracle_pq": orc,
           "pixel_dice_mean": float(np.mean([r["pixel_dice"] for r in recs])),
           "boundary": {"n": len(bnd), "pred_over_gt_q10_50_90": q([c["pred_over_gt"] for c in bnd]),
                        "partner_precision_q10_50_90": q([c["pred_inside"] for c in bnd]),
                        "gt_recall_q10_50_90": q([c["gt_covered"] for c in bnd]),
                        "best_iou_q10_50_90": q([c["best_iou"] for c in bnd])},
           "fn_best_iou_0.4_0.5": near,
           "missed": {"area_q10_50_90": q(miss_area), "share_under_400px": miss_small},
           "false_pos": {"n": len(fp_area), "area_q10_50_90": q(fp_area), "mean_prob_q10_50_90": q(fp_prob)},
           "hit_iou_q10_50_90": q(hits_iou), "hit_iou_below_0.6": float((hits_iou < 0.6).mean()),
           "sq_if_rq_perfect": base["sq"], "rq_if_sq_perfect": base["rq"]}
    (OUTD / "results").mkdir(exist_ok=True)
    (OUTD / "results" / "diagnosis.json").write_text(json.dumps(out, indent=1))
    picked = panel(recs) if a.limit is None or a.limit >= 50 else []

    fn = base["fn"]
    L = ["# Step 2 — diagnosis of the submitted pipeline", "",
         "_Tables generated by `kaggle/try_error/diagnose.py`; the reading below the marker is hand-written._", "",
         f"Local validation: {len(jobs)} inner-test disks of `kaggle/splits/folds_k707_v1.json` "
         f"(5 folds, grouped by 7-day observation window), {base['n_rows']} annotator records, "
         "official self-evaluation rule.", "",
         "| PQ | SQ | RQ (detection F1) | TP / FP / FN | Dice, matched pairs | Dice, overlapping pairs | pixel Dice per record | one-to-many | many-to-one |",
         "|---|---|---|---|---|---|---|---|---|",
         f"| **{base['pq']:.4f}** | {base['sq']:.3f} | {base['rq']:.3f} | {base['tp']}/{base['fp']}/{base['fn']} | "
         f"{base['dice_matched']:.3f} | {base['dice_overlapping']:.3f} | {out['pixel_dice_mean']:.3f} | "
         f"{base['one_to_many']} | {base['many_to_one']} |", "",
         "## Failure classes (per annotator record)", "",
         "| class | count | share of FN / FP | PQ if this class were repaired | gain |", "|---|---|---|---|---|"]
    for c in CLASSES:
        L.append(f"| {c} (GT without a hit) | {counts[c]} | {counts[c] / max(fn, 1):.1%} of FN | "
                 f"{orc[c]:.4f} | +{orc[c] - base['pq']:.4f} |")
    L.append(f"| false positive (prediction on no GT) | {counts['false_pos']} | "
             f"{counts['false_pos'] / max(base['fp'], 1):.1%} of FP | {orc['false_pos']:.4f} | "
             f"+{orc['false_pos'] - base['pq']:.4f} |")
    L.append(f"| partner (prediction on a GT it does not hit) | {counts['partner']} | "
             f"{counts['partner'] / max(base['fp'], 1):.1%} of FP | — | — |")
    b = out["boundary"]
    L += ["", "## Numbers behind the RQ-vs-SQ call", "",
          f"- Hits: IoU quantiles 10/50/90 % = {' / '.join(f'{x:.2f}' for x in out['hit_iou_q10_50_90'])}; "
          f"{out['hit_iou_below_0.6']:.0%} of hits sit between 0.5 and 0.6.",
          f"- FN whose best prediction has IoU in (0.4, 0.5]: {near} ({near / max(fn, 1):.1%} of FN).",
          f"- Boundary failures ({b['n']}): partner area / GT area 10/50/90 % = "
          f"{' / '.join(f'{x:.2f}' for x in b['pred_over_gt_q10_50_90'])}; share of the partner inside the GT "
          f"{' / '.join(f'{x:.2f}' for x in b['partner_precision_q10_50_90'])}; share of the GT covered "
          f"{' / '.join(f'{x:.2f}' for x in b['gt_recall_q10_50_90'])}.",
          f"- Missed filaments ({len(miss_area)}): GT area 10/50/90 % = "
          f"{' / '.join(f'{x:.0f}' for x in out['missed']['area_q10_50_90'])} px; {miss_small:.0%} are smaller than "
          "the 400 px seed minimum.",
          f"- Pure false positives ({out['false_pos']['n']}): area 10/50/90 % = "
          f"{' / '.join(f'{x:.0f}' for x in out['false_pos']['area_q10_50_90'])} px; mean probability "
          f"{' / '.join(f'{x:.2f}' for x in out['false_pos']['mean_prob_q10_50_90'])}.",
          f"- PQ = SQ x RQ = {base['sq']:.3f} x {base['rq']:.3f}. With every instance detected (RQ = 1) PQ would be "
          f"{base['sq']:.3f}; with perfect masks on today's hits (SQ = 1) it would be {base['rq']:.3f}.", "",
          "## Failure panel", "",
          "![20 failure cases](../figures/failures/te01_failure_panel.png)", "",
          "| # | class | disk | record | detail |", "|---|---|---|---|---|"]
    for k, p in enumerate(picked):
        det = ", ".join(f"{kk} {vv:.2f}" if isinstance(vv, float) else f"{kk} {vv}" for kk, vv in p.items()
                        if kk in ("best_iou", "cov", "n_frag", "n_merged", "pred_over_gt"))
        L.append(f"| {k + 1} | {p['class']} | {p['stem']} | {p['record'].split('-')[0]} | {det} |")
    doc = OUTD / "docs" / "01_diagnosis.md"
    doc.parent.mkdir(exist_ok=True)
    marker = "## Reading (hand-written)"
    tail = doc.read_text().split(marker, 1)[1] if doc.exists() and marker in doc.read_text() else "\n\n_pending_\n"
    doc.write_text("\n".join(L) + f"\n\n{marker}" + tail)
    print(f"wrote {doc} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
