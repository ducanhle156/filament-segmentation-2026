"""Loss terms and their weighted combination (Phase 2, Step 13).

Phase 0 measured a foreground fraction of 0.65% per disk (1.04% inside a
foreground tile). Plain cross-entropy on that imbalance collapses to
all-background, which is why every arm's loss carries a Dice term, and why
clDice is available alongside it: filaments are thin, and a loss that only
counts area is happy to lose the centre line.

Every term takes logits `[B,1,H,W]`, a binary target of the same shape, and
the analysis-region mask, and returns a scalar. Off-region pixels are excluded
from every term — including them would let the model earn loss reductions on
the 45% of each frame it is not being asked to segment.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from common.metrics.cldice_metric import soft_skeleton

EPS = 1e-6


def _masked(x, valid):
    return x * valid


def dice_loss(logits, target, valid, eps: float = EPS):
    """Soft Dice on probabilities, per image then averaged."""
    p = _masked(torch.sigmoid(logits), valid)
    t = _masked(target, valid)
    dims = (1, 2, 3)
    inter = (p * t).sum(dims)
    total = p.sum(dims) + t.sum(dims)
    return (1.0 - (2 * inter + eps) / (total + eps)).mean()


def bce_loss(logits, target, valid, pos_weight: float | None = None):
    """Masked binary cross-entropy; the mean is over valid pixels only."""
    kw = {}
    if pos_weight is not None:
        kw["pos_weight"] = torch.as_tensor(pos_weight, device=logits.device,
                                           dtype=logits.dtype)
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none", **kw)
    return (raw * valid).sum() / valid.sum().clamp_min(1.0)


def cldice_loss(logits, target, valid, iters: int = 5, eps: float = EPS):
    """1 - soft clDice. Differentiable, so it uses the soft skeleton.

    `iters` should be at least the half-width of the structures: filaments are
    ~10 px across at full resolution, so 5 peels them to the centre line.
    """
    p = _masked(torch.sigmoid(logits), valid)
    t = _masked(target, valid)
    sp = soft_skeleton(p, iters)
    st = soft_skeleton(t, iters)
    dims = (1, 2, 3)
    tprec = ((sp * t).sum(dims) + eps) / (sp.sum(dims) + eps)
    tsens = ((st * p).sum(dims) + eps) / (st.sum(dims) + eps)
    return (1.0 - 2.0 * tprec * tsens / (tprec + tsens + eps)).mean()


def focal_loss(logits, target, valid, alpha: float = 0.25, gamma: float = 2.0):
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    w = (alpha * target + (1 - alpha) * (1 - target)) * (1 - pt).pow(gamma)
    return ((raw * w) * valid).sum() / valid.sum().clamp_min(1.0)


def component_loss(logits, target, valid, fp_w: float = 2.0,
                   fn_w: float = 2.0, miss_frac: float = 0.1):
    """Matching-aware surrogate (Phase 7-1, arm12): weighted BCE where the
    pixels of component-level mistakes carry extra weight — `fp_w` on
    predicted components with zero GT overlap (pushed toward background),
    `fn_w` on GT components the prediction covers by less than `miss_frac`
    (pushed toward foreground). PQ's RQ term charges the model per such
    component; Dice/clDice, being pixel/skeleton pooled, never see them.

    Components come from the DETACHED binarized prediction (CPU scipy label,
    ~ms per tile), so gradients flow through the pixel loss only.
    """
    import numpy as np
    from scipy import ndimage as ndi

    with torch.no_grad():
        pred_np = ((torch.sigmoid(logits) >= 0.5) & (valid > 0.5)
                   ).squeeze(1).cpu().numpy()
        t_np = ((target > 0.5) & (valid > 0.5)).squeeze(1).cpu().numpy()
        w_np = np.ones(pred_np.shape, dtype=np.float32)
        for b in range(pred_np.shape[0]):
            lab_p, n_p = ndi.label(pred_np[b])
            if n_p:
                overlap = np.bincount(
                    lab_p.ravel(), weights=t_np[b].ravel().astype(np.float64),
                    minlength=n_p + 1)[1:]
                fp_ids = np.nonzero(overlap == 0)[0] + 1
                if fp_ids.size:
                    w_np[b][np.isin(lab_p, fp_ids)] = fp_w
            lab_t, n_t = ndi.label(t_np[b])
            if n_t:
                area_t = np.bincount(lab_t.ravel(), minlength=n_t + 1)[1:]
                cov = np.bincount(
                    lab_t.ravel(),
                    weights=pred_np[b].ravel().astype(np.float64),
                    minlength=n_t + 1)[1:]
                miss_ids = np.nonzero(cov < miss_frac * area_t)[0] + 1
                if miss_ids.size:
                    w_np[b][np.isin(lab_t, miss_ids)] = fn_w
        w = torch.from_numpy(w_np).unsqueeze(1).to(logits.device)
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (raw * w * valid).sum() / valid.sum().clamp_min(1.0)


TERMS = {
    "dice": dice_loss,
    "bce": bce_loss,
    "cldice": cldice_loss,
    "focal": focal_loss,
    "component": component_loss,
}


def pil_bce_loss(logits, target, valid, extra=None, pil_channel: int = 2,
                 valid_channel: int = 3, near: float = 0.5,
                 fn_w: float = 2.0, fp_w: float = 2.0):
    """BCE whose pixel weights come from a magnetic prior (Phase 8, arm19).

    `extra` is the side-channel block of the input ([B,C,H,W]); channel
    `pil_channel` is a PIL proximity in [0, 1] and `valid_channel` says where
    the magnetogram exists. Where it exists, a filament pixel *near* an
    inversion line (proximity >= `near`) that the model misses costs `fn_w`
    times more, and a background pixel *far* from any line that the model
    fires on costs `fp_w` times more. Everywhere else, and when `extra` is
    None (single-channel arms, or an image without a magnetogram), it is
    plain masked BCE. The prior shapes training only: nothing magnetic is
    needed at inference."""
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    w = valid
    if extra is not None:
        mv = extra[:, valid_channel:valid_channel + 1]
        is_near = (extra[:, pil_channel:pil_channel + 1] >= near).to(raw.dtype)
        w = valid * (1.0 + mv * ((fn_w - 1.0) * target * is_near
                                 + (fp_w - 1.0) * (1.0 - target) * (1.0 - is_near)))
    return (raw * w).sum() / w.sum().clamp_min(1.0)


TERMS["pil_bce"] = pil_bce_loss
EXTRA_TERMS = {"pil_bce"}            # terms that take the side-channel block


class ComboLoss(torch.nn.Module):
    """Weighted sum of named terms, built straight from the arm config.

        loss:
          terms:
            - {name: dice,   weight: 1.0}
            - {name: cldice, weight: 0.5, iters: 5}
    """

    def __init__(self, terms):
        super().__init__()
        self.terms = []
        for t in terms:
            spec = dict(t)
            name = spec.pop("name")
            weight = float(spec.pop("weight", 1.0))
            if name not in TERMS:
                raise ValueError(f"unknown loss term {name!r}; "
                                 f"have {sorted(TERMS)}")
            self.terms.append((name, weight, spec))

    def forward(self, logits, target, valid, extra=None):
        """`extra`: the input's side channels ([B,C,H,W]) or None; only the
        terms in EXTRA_TERMS see it."""
        total = logits.new_zeros(())
        parts = {}
        for name, weight, kw in self.terms:
            if name in EXTRA_TERMS:
                v = TERMS[name](logits, target, valid, extra=extra, **kw)
            else:
                v = TERMS[name](logits, target, valid, **kw)
            parts[name] = float(v.detach())
            total = total + weight * v
        return total, parts

    def describe(self):
        return [{"name": n, "weight": w, **kw} for n, w, kw in self.terms]


def build_loss(cfg: dict) -> ComboLoss:
    return ComboLoss(cfg["loss"]["terms"])
