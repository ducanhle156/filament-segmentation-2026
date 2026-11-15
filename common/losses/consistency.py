"""Unsupervised consistency loss for Mean Teacher (Phase 9, arm21).

The teacher's probabilities on the weak view are the *soft* target for the
student's logits on the strong view. Only pixels where the teacher is
confident count: p >= tau_pos ("filament") or p <= tau_neg ("background").
Two thresholds because foreground is ~0.7 % of pixels — one symmetric band
would make almost every confident pixel a background pixel.

The confidence weight is passed to the loss terms through their `valid`
argument and nowhere else. Round-1 self-training (kaggle_ssl, 2026-10-26)
removed uncertain pixels from the *dataset* `valid`, which also zeroes the
input image (common/data/dataset.py:189); every student collapsed. Here the
student always sees the full tile; only the loss is masked.

Only `bce` and `dice` are allowed as unsupervised terms: `component`
binarises the target and labels components on the CPU, `cldice` skeletonises
it — neither is meaningful on a soft target.
"""
from __future__ import annotations

import torch

from common.losses import combo

UNSUP_TERMS = {"bce": combo.bce_loss, "dice": combo.dice_loss}


def confidence_targets(p_t: torch.Tensor, valid: torch.Tensor,
                       tau_pos: float, tau_neg: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(soft target, loss weight). Weight is 1 on valid pixels where the
    teacher is confident either way, 0 elsewhere."""
    conf = (p_t >= tau_pos) | (p_t <= tau_neg)
    weight = valid * conf.to(valid.dtype)
    return p_t, weight


def unsup_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,
               terms) -> tuple[torch.Tensor, dict]:
    """Weighted sum of the allowed terms; `weight` plays the role of `valid`."""
    total = logits.new_zeros(())
    parts = {}
    for t in terms:
        spec = dict(t)
        name = spec.pop("name")
        w = float(spec.pop("weight", 1.0))
        if name not in UNSUP_TERMS:
            raise ValueError(f"unsupervised term {name!r} not allowed; "
                             f"have {sorted(UNSUP_TERMS)}")
        v = UNSUP_TERMS[name](logits, target, weight, **spec)
        parts[name] = float(v.detach())
        total = total + w * v
    return total, parts


@torch.no_grad()
def unsup_stats(p_t: torch.Tensor, weight: torch.Tensor, valid: torch.Tensor,
                tau_pos: float) -> dict:
    """What the teacher is saying, for the epoch log / collapse diagnostics."""
    n_valid = valid.sum().clamp_min(1.0)
    pos = ((p_t >= tau_pos).to(valid.dtype) * valid).sum()
    return {"conf_frac": float(weight.sum() / n_valid),
            "conf_frac_pos": float(pos / n_valid),
            "teacher_fg_frac": float(((p_t >= 0.5).to(valid.dtype) * valid).sum() / n_valid),
            "teacher_mean_p": float((p_t * valid).sum() / n_valid)}
