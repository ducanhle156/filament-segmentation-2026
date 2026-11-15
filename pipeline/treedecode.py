"""Component-tree decoding: pick each filament at its own threshold.

The submitted decode cuts the probability map at one global level. The
diagnosis (docs/01_diagnosis.md) shows that most lost filaments are found
but drawn at the wrong extent (too short, too thin, split, or merged with a
neighbour) and that the best level differs from filament to filament.

1. `build_tree`: 8-connected components of {p >= t} for a ladder of
   thresholds. A component at a higher level lies inside exactly one
   component of the level below, so the components form a forest.
   Every node gets features from the probability map and the image only
   (size, shape, confidence, local darkness, position on the disk, how the
   node relates to its parent and children).
2. A regressor predicts q(node) = expected IoU the node would earn as a
   true positive (0 when it would not match), learned on out-of-fold maps.
3. `select`: choose a set of nodes with no node inside another (so masks
   are pixel-disjoint) maximising sum(q - tau) by dynamic programming:
        best(node) = max(q(node) - tau, sum of best(children)).
   Why q - tau: PQ = sum IoU / (TP + FP/2 + FN/2). Each extra prediction
   adds its IoU to the numerator if it matches and 1/2 to the denominator
   either way, so it raises PQ exactly when its expected IoU exceeds PQ/2.
   tau is therefore about PQ/2 (0.2); it is tuned by nested cross-fitting.

No file name, timestamp or station is used; inputs are the image, the
analysis-disk mask derived from it, and the model's probability map.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

STRUCT8 = np.ones((3, 3), dtype=bool)
THR = (0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90)
MIN_AREA = 30
FEATS = ["thr", "log_area", "mean_p", "std_p", "max_p", "mean_img", "contrast", "major", "minor", "elong",
         "width", "log_perim", "extent", "rdist", "parent_area_ratio", "parent_mean_p", "n_children",
         "child_area_ratio", "root_area_ratio", "n_at_050", "frac_at_050", "depth_span"]


def build_tree(prob: np.ndarray, valid: np.ndarray, img: np.ndarray, thr=THR, min_area: int = MIN_AREA) -> dict:
    """-> {"X": features (n, F), "parent": (n,), "level": (n,), "cid": (n,), "area": (n,),
           "idx0": flat indices of the pixels above the lowest level, "labs": [label of idx0 pixels per level]}"""
    h, w = prob.shape
    idx0 = np.flatnonzero(((prob >= thr[0]) & valid).ravel())
    p = prob.ravel()[idx0].astype(np.float64)
    im = img.ravel()[idx0].astype(np.float64)
    bg = (img - ndi.uniform_filter(img, 61)).ravel()[idx0].astype(np.float64)   # darkness against the surroundings
    yy, xx = np.divmod(idx0, w)
    yy, xx = yy.astype(np.float64), xx.astype(np.float64)
    cy, cx = ndi.center_of_mass(valid)
    rad = np.sqrt(valid.sum() / np.pi)
    cols = {k: [] for k in ("level", "cid", "area", "parent", "mean_p", "std_p", "max_p", "mean_img", "contrast",
                            "major", "minor", "perim", "extent", "rdist")}
    labs, prev_lab, prev_node, offset = [], None, None, 0
    n050 = f050 = 0
    for k, t in enumerate(thr):
        mask = (prob >= t) & valid
        lab, n = ndi.label(mask, structure=STRUCT8)
        lc = lab.ravel()[idx0]
        labs.append(lc)
        if n == 0:
            prev_lab, prev_node = lc, np.full(1, -1)
            continue
        bc = lambda wts=None: np.bincount(lc, weights=wts, minlength=n + 1)[1:]  # noqa: E731
        area = bc()
        if abs(t - 0.5) < 1e-9:
            n050, f050 = int((area >= 400).sum()), float(area.sum() / max(valid.sum(), 1))
        sp, sp2 = bc(p), bc(p * p)
        a = np.maximum(area, 1)
        mx, my = bc(xx) / a, bc(yy) / a
        vxx = bc(xx * xx) / a - mx * mx + 1 / 12
        vyy = bc(yy * yy) / a - my * my + 1 / 12
        vxy = bc(xx * yy) / a - mx * my
        tr, det = vxx + vyy, vxx * vyy - vxy * vxy
        disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
        major, minor = 4 * np.sqrt(np.maximum(tr / 2 + disc, 0)), 4 * np.sqrt(np.maximum(tr / 2 - disc, 1e-6))
        edge = mask & ~ndi.binary_erosion(mask)
        perim = np.bincount(lab.ravel()[np.flatnonzero(edge.ravel())], minlength=n + 1)[1:]
        sl = ndi.find_objects(lab)
        box = np.array([(s[0].stop - s[0].start) * (s[1].stop - s[1].start) for s in sl], dtype=np.float64)
        maxp = ndi.maximum(prob, lab, index=np.arange(1, n + 1))
        # one pixel of every component -> its component one level below
        first = np.zeros(n, dtype=np.int64)
        pos = np.flatnonzero(lc)
        first[lc[pos][::-1] - 1] = pos[::-1]
        par = prev_node[prev_lab[first]] if prev_lab is not None else np.full(n, -1)
        keep = area >= min_area
        node = np.full(n + 1, -1, dtype=np.int64)
        node[1:][keep] = offset + np.arange(int(keep.sum()))
        offset += int(keep.sum())
        for name, v in (("level", np.full(n, k)), ("cid", np.arange(1, n + 1)), ("area", area), ("parent", par),
                        ("mean_p", sp / a), ("std_p", np.sqrt(np.maximum(sp2 / a - (sp / a) ** 2, 0))),
                        ("max_p", maxp), ("mean_img", bc(im) / a), ("contrast", bc(bg) / a), ("major", major),
                        ("minor", minor), ("perim", perim), ("extent", area / np.maximum(box, 1)),
                        ("rdist", np.hypot(mx - cx, my - cy) / rad)):
            cols[name].append(np.asarray(v)[keep])
        prev_lab, prev_node = lc, node
    c = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    n = len(c["area"])
    parent = c["parent"].astype(np.int64)
    level = c["level"].astype(np.int64)
    area = c["area"]
    n_child, child_max, top = np.zeros(n), np.zeros(n), level.copy()
    root_area = area.copy()
    for i in range(n):                       # parents precede children
        if parent[i] >= 0:
            root_area[i] = root_area[parent[i]]
    for i in range(n - 1, -1, -1):
        j = parent[i]
        if j >= 0:
            n_child[j] += 1
            child_max[j] = max(child_max[j], area[i])
            top[j] = max(top[j], top[i])
    has_p = parent >= 0
    pa = np.where(has_p, area[np.maximum(parent, 0)], area)
    pm = np.where(has_p, c["mean_p"][np.maximum(parent, 0)], c["mean_p"]) if n else np.zeros(0)
    thr_a = np.asarray(thr)
    X = np.stack([thr_a[level], np.log(np.maximum(area, 1)), c["mean_p"], c["std_p"], c["max_p"], c["mean_img"],
                  c["contrast"], c["major"], c["minor"], c["major"] / np.maximum(c["minor"], 1e-3),
                  2 * area / np.maximum(c["perim"], 1), np.log(np.maximum(c["perim"], 1)), c["extent"], c["rdist"],
                  area / np.maximum(pa, 1), pm, n_child, child_max / np.maximum(area, 1),
                  area / np.maximum(root_area, 1), np.full(n, n050), np.full(n, f050),
                  thr_a[top] - thr_a[level]], axis=1).astype(np.float32) if n else np.zeros((0, len(FEATS)), np.float32)
    return {"X": X, "parent": parent, "level": level, "cid": c["cid"].astype(np.int64), "area": area,
            "idx0": idx0, "labs": labs, "shape": (h, w)}


def select(parent: np.ndarray, q: np.ndarray, tau: float, allowed: np.ndarray | None = None) -> np.ndarray:
    """Indices of the chosen nodes: no chosen node contains another, sum(q - tau) is maximal.
    `allowed` (bool per node) restricts which nodes may be chosen (e.g. one level only)."""
    n = len(q)
    own = q - tau
    if allowed is not None:
        own = np.where(allowed, own, -np.inf)
    kids = np.zeros(n)
    for i in range(n - 1, -1, -1):           # children before parents
        best = max(own[i], kids[i], 0.0)
        if parent[i] >= 0:
            kids[parent[i]] += best
    take = np.zeros(n, dtype=bool)
    blocked = np.zeros(n, dtype=bool)        # an ancestor was taken
    for i in range(n):
        if parent[i] >= 0 and (blocked[parent[i]] or take[parent[i]]):
            blocked[i] = True
            continue
        if own[i] > 0 and own[i] >= kids[i]:
            take[i] = True
    return np.flatnonzero(take)


def render(tree: dict, chosen: np.ndarray) -> tuple[np.ndarray, int]:
    """Label map (0 = background, 1..k = instances) of the chosen nodes."""
    h, w = tree["shape"]
    out = np.zeros(h * w, dtype=np.int32)
    for k, i in enumerate(chosen):
        out[tree["idx0"][tree["labs"][tree["level"][i]] == tree["cid"][i]]] = k + 1
    return out.reshape(h, w), len(chosen)
