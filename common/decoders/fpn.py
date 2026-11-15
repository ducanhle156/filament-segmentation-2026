"""FPN decoder (Phase 5, Step 21) — shared by the DINOv3 and SAM 2 arms.

    model:
      decoder: {name: fpn, channels: 256}

Consumes the registry's encoder contract: a list of taps (fine -> coarse),
`out_channels`, `out_strides`. Two kinds of encoder feed it and the decoder
is the same module for both — that is the point of arm08 vs arm09:

* a ViT (DINOv3): every tap has stride 16, so the pyramid has to be built
  first. Each tap is resampled to its target stride, per the Step 21 table:

      | tap      | resample                   | stride |
      | block 3  | transposed conv, 4x up     | 4      |
      | block 6  | transposed conv, 2x up     | 8      |
      | block 9  | identity                   | 16     |
      | block 12 | strided conv, 2x down      | 32     |

  The resampling is decided from `in_strides` versus `pyramid_strides`
  (default [4, 8, 16, 32], assigned fine -> coarse), not from block indices,
  so the Step 22 tap-set ablation ([12] alone, [9..12], [2,4,6,9]) is a
  config change and nothing else. Resamples keep the tap's channel count:
  a 4x up is two 2x2/2 transposed convs, a 2x down is one 2x2/2 conv.

* a hierarchical backbone (SAM 2's Hiera): the strides already are
  [4, 8, 16, 32], every resample is the identity, and the pyramid is the
  encoder's own. That asymmetry is architectural and is noted in the paper
  (`models/sam2/MODEL_SPEC.md`); it is not something the decoder adds.

Then a standard FPN: 1x1 lateral projections to `channels`, top-down
upsample-and-add, one 3x3 smoothing conv per level, and a head that
resamples every level to the finest stride, concatenates, fuses with a 3x3
conv and predicts the logit. The logits are returned at input resolution
(bilinear from the finest stride), per the decoder contract.

Budget. Step 21 asks for 10-20M parameters: with four 768-channel ViT-B taps
and channels 256 this module has ~15.5M (resamples 8.3M, laterals 0.8M,
smoothing 2.4M, head 3.7M), and 3.7M on Hiera-B+'s [112, 224, 448, 896]
pyramid, which needs no resamples. A heavier head is where overfitting on
958 images would appear first, and it would confound arm07 -> arm08.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.engine.registry import register_decoder


def _conv_gn_gelu(ci: int, co: int, k: int) -> nn.Sequential:
    # GroupNorm rather than BatchNorm: batch 8 of 512-tiles with a 0.65 %
    # foreground fraction gives noisy batch statistics, and GN keeps the
    # decoder's behaviour identical between train and eval.
    return nn.Sequential(
        nn.Conv2d(ci, co, k, padding=k // 2, bias=False),
        nn.GroupNorm(32 if co % 32 == 0 else 1, co),
        nn.GELU(),
    )


def make_resample(channels: int, in_stride: int, out_stride: int) -> nn.Module:
    """Map a stride-`in_stride` tap to stride `out_stride`, channels kept.

    Up by 2^k = k transposed 2x2/2 convs (GN+GELU between them, as ViTDet's
    simple feature pyramid does); down by 2^k = k strided 2x2/2 convs;
    equal = identity.
    """
    if in_stride == out_stride:
        return nn.Identity()
    ratio = in_stride / out_stride
    steps = math.log2(ratio) if ratio > 1 else math.log2(1 / ratio)
    if abs(steps - round(steps)) > 1e-9:
        raise ValueError(f"stride {in_stride} -> {out_stride} is not a power of two")
    layers: list[nn.Module] = []
    for i in range(int(round(steps))):
        if ratio > 1:                           # up
            layers.append(nn.ConvTranspose2d(channels, channels, 2, stride=2))
        else:                                   # down
            layers.append(nn.Conv2d(channels, channels, 2, stride=2))
        if i + 1 < int(round(steps)):
            layers += [nn.GroupNorm(32 if channels % 32 == 0 else 1, channels),
                       nn.GELU()]
    return nn.Sequential(*layers)


@register_decoder("fpn")
class FPNDecoder(nn.Module):
    def __init__(self, in_channels: list[int], in_strides: list[int],
                 num_classes: int = 1, channels: int = 256,
                 pyramid_strides: list[int] | None = None,
                 dropout: float = 0.1):
        super().__init__()
        in_channels = [int(c) for c in in_channels]
        in_strides = [int(s) for s in in_strides]
        n = len(in_channels)
        if n < 1:
            raise ValueError("fpn needs at least one tap")
        if pyramid_strides is None:
            pyramid_strides = [4, 8, 16, 32]
        if n > len(pyramid_strides):
            raise ValueError(f"{n} taps but only {len(pyramid_strides)} pyramid "
                             f"strides {pyramid_strides}")
        # taps are fine -> coarse; the pyramid is assigned in the same order
        self.in_strides = in_strides
        self.strides = [int(s) for s in pyramid_strides[:n]]
        self.channels = int(channels)

        self.resamples = nn.ModuleList(
            make_resample(c, si, so)
            for c, si, so in zip(in_channels, in_strides, self.strides))
        self.laterals = nn.ModuleList(
            nn.Conv2d(c, self.channels, 1) for c in in_channels)
        self.smooth = nn.ModuleList(
            _conv_gn_gelu(self.channels, self.channels, 3) for _ in in_channels)
        self.fuse = _conv_gn_gelu(self.channels * n, self.channels, 3)
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.head = nn.Conv2d(self.channels, num_classes, 1)
        nn.init.zeros_(self.head.bias)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        if len(feats) != len(self.laterals):
            raise ValueError(f"expected {len(self.laterals)} taps, got {len(feats)}")
        # 1. pyramid
        pyr = [rs(f) for rs, f in zip(self.resamples, feats)]
        # 2. laterals + top-down
        lat = [l(p) for l, p in zip(self.laterals, pyr)]
        for i in range(len(lat) - 1, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(
                lat[i], size=lat[i - 1].shape[-2:], mode="bilinear",
                align_corners=False)
        outs = [s(x) for s, x in zip(self.smooth, lat)]
        # 3. fuse at the finest level
        size = outs[0].shape[-2:]
        outs = [outs[0]] + [F.interpolate(o, size=size, mode="bilinear",
                                          align_corners=False)
                            for o in outs[1:]]
        logits = self.head(self.dropout(self.fuse(torch.cat(outs, dim=1))))
        # 4. input resolution (the finest pyramid stride)
        return F.interpolate(logits, scale_factor=self.strides[0],
                             mode="bilinear", align_corners=False)
