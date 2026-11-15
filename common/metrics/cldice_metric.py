"""clDice (Phase 2, Step 12).

clDice = harmonic mean of topology precision and topology sensitivity:

    tprec = |skel(P) and T| / |skel(P)|      (centerline precision)
    tsens = |skel(T) and P| / |skel(T)|      (centerline recall)
    clDice = 2 * tprec * tsens / (tprec + tsens)

Two skeletonizations live here and they are not interchangeable:

`cldice`        morphological skeletons, the **reported metric**. Matches the
                Phase 0 inter-annotator ceiling (clDice 0.660 excluding
                both-empty pairs), so an arm's clDice can be read against it.
`soft_cldice`   the differentiable soft-skeleton of Shit et al., for the
                Phase 3+ training loss. Iterated min/max pooling, no skimage,
                stays on the GPU and keeps a gradient. Its values differ
                slightly from the morphological ones, so it must never be
                used for a reported number.
"""

from __future__ import annotations

import numpy as np

from common.metrics.centerline import skeletons, _shape_of
from common.metrics.common import BOTH_EMPTY, is_tensor, like, prepare, ratio


def cldice(pred, target, valid_mask=None):
    """Reported clDice, on morphological skeletons. Returns [B]."""
    p, t, _ = prepare(pred, target, valid_mask)
    shape = _shape_of(pred, valid_mask)
    sp, st = skeletons(p, shape), skeletons(t, shape)
    tprec = ratio(np.logical_and(sp, t).sum(1), sp.sum(1), 0.0)
    tsens = ratio(np.logical_and(st, p).sum(1), st.sum(1), 0.0)
    denom = tprec + tsens
    out = np.where(denom > 0, 2 * tprec * tsens / np.where(denom > 0, denom, 1.0), 0.0)
    p_any, t_any = p.sum(1) > 0, t.sum(1) > 0
    out = np.where(~p_any & ~t_any, BOTH_EMPTY, out)
    out = np.where(p_any ^ t_any, 0.0, out)
    return like(out, pred)


# ------------------------------------------------------------- soft skeleton

def _soft_erode(x):
    """Min over a cross-shaped neighbourhood, as two separable max-pools."""
    import torch
    import torch.nn.functional as F
    vertical = -F.max_pool2d(-x, (3, 1), (1, 1), (1, 0))
    horizontal = -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1))
    return torch.min(vertical, horizontal)


def _soft_dilate(x):
    import torch.nn.functional as F
    return F.max_pool2d(x, (3, 3), (1, 1), (1, 1))


def _soft_open(x):
    return _soft_dilate(_soft_erode(x))


def soft_skeleton(x, iters: int = 3):
    """Differentiable skeleton of a probability map [B,1,H,W] in [0,1].

    Shit et al., CVPR 2021. Repeated morphological opening peels the shape one
    layer at a time; what erosion removes but opening does not restore is
    skeleton. Keep `iters` at least the expected half-width of the structures
    — filaments here are ~10 px wide, so the Phase 3 loss should use ~5.
    """
    import torch
    skel = torch.relu(x - _soft_open(x))
    for _ in range(iters):
        x = _soft_erode(x)
        delta = torch.relu(x - _soft_open(x))
        skel = skel + delta - skel * delta
    return skel


def soft_cldice(pred_prob, target, iters: int = 5, eps: float = 1e-6):
    """Differentiable clDice on probabilities, for the training loss.

    `pred_prob` is [B,1,H,W] in [0,1] (post-sigmoid), `target` the binary
    ground truth in the same shape. Returns [B]; the loss is 1 - this.
    Never report this number — report `cldice`.
    """
    sp = soft_skeleton(pred_prob, iters)
    st = soft_skeleton(target.to(pred_prob.dtype), iters)
    dims = (1, 2, 3)
    tprec = ((sp * target).sum(dims) + eps) / (sp.sum(dims) + eps)
    tsens = ((st * pred_prob).sum(dims) + eps) / (st.sum(dims) + eps)
    return 2.0 * tprec * tsens / (tprec + tsens + eps)
