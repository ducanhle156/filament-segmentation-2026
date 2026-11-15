"""Linear probe head (Phase 4, Step 18).

One 1x1 convolution from patch features to the class logit, then bilinear
upsampling to the input resolution. With a ViT-B/16 at 512 that is a 32x32
grid of 768-d tokens -> 32x32 logits -> 512x512; 769 trainable parameters.
The head is deliberately incapable of learning anything spatial: whatever it
scores is a property of the frozen features, which is the whole point of the
probe (`05_phase4_probes.md`, interpretation table).

Several taps are concatenated along channels before the 1x1, so the same head
also serves a multi-tap probe. Every tap must share one stride.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.engine.registry import register_decoder


@register_decoder("linear")
class LinearDecoder(nn.Module):
    def __init__(self, in_channels, in_strides, num_classes: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        strides = list(in_strides)
        if len(set(strides)) != 1:
            raise ValueError(f"linear head wants one stride, got {strides}")
        self.stride = int(strides[0])
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Conv2d(int(sum(in_channels)), num_classes, 1)
        nn.init.zeros_(self.proj.bias)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, feats) -> torch.Tensor:
        x = feats[0] if len(feats) == 1 else torch.cat(list(feats), dim=1)
        x = self.proj(self.drop(x))
        return F.interpolate(x, scale_factor=self.stride, mode="bilinear",
                             align_corners=False)
