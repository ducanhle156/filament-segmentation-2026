"""UPerNet decoder — a pyramid pooling module on top of an FPN.

Written in `common/` rather than under one family because two phases need it:
Step 16's from-scratch ConvNeXt arm, and any later arm that wants a heavier
head than the plain FPN of Step 21. It consumes the registry's encoder
contract (a list of taps, `out_channels`, `out_strides`) so the same decoder
sits on a ConvNeXt pyramid or a ViT's reshaped taps without changes.

`fpn_dim` defaults to 256, the same lateral width Step 21 fixes for the FPN
decoder. Keeping the two decoders at one width is deliberate: when a later arm
beats this one, the difference should be the backbone, not a wider head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.engine.registry import register_decoder


def _conv_bn_relu(ci: int, co: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(ci, co, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(co),
        nn.ReLU(inplace=True),
    )


class PyramidPoolingModule(nn.Module):
    """PSPNet's PPM: global context at four pooled scales, concatenated."""

    def __init__(self, in_channels: int, out_channels: int,
                 scales: tuple[int, ...] = (1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList(
            nn.Sequential(nn.AdaptiveAvgPool2d(s),
                          _conv_bn_relu(in_channels, out_channels, 1))
            for s in scales)
        self.bottleneck = _conv_bn_relu(
            in_channels + len(scales) * out_channels, out_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        pooled = [F.interpolate(stage(x), size=size, mode="bilinear",
                                align_corners=False) for stage in self.stages]
        return self.bottleneck(torch.cat([x] + pooled, dim=1))


@register_decoder("upernet")
class UPerNetDecoder(nn.Module):
    """Taps (fine -> coarse) in, one logit map at the finest tap's stride out.

    The caller is responsible for the final upsample to input resolution; the
    decoder does not know the input size, only the strides it was built with.
    """

    def __init__(self, in_channels: list[int], in_strides: list[int],
                 num_classes: int = 1, fpn_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        if len(in_channels) < 2:
            raise ValueError("UPerNet needs at least two taps")
        self.in_strides = list(in_strides)

        # coarsest tap goes through the PPM; the rest get 1x1 laterals
        self.ppm = PyramidPoolingModule(in_channels[-1], fpn_dim)
        self.laterals = nn.ModuleList(
            _conv_bn_relu(c, fpn_dim, 1) for c in in_channels[:-1])
        # smooths the aliasing that top-down nearest/bilinear addition creates
        self.fpn_convs = nn.ModuleList(
            _conv_bn_relu(fpn_dim, fpn_dim, 3) for _ in in_channels[:-1])

        self.fuse = _conv_bn_relu(fpn_dim * len(in_channels), fpn_dim, 3)
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.head = nn.Conv2d(fpn_dim, num_classes, 1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        laterals = [lat(f) for lat, f in zip(self.laterals, feats[:-1])]
        laterals.append(self.ppm(feats[-1]))

        # top-down: each level adds the coarser level resampled onto it
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:],
                mode="bilinear", align_corners=False)

        outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals[:-1])]
        outs.append(laterals[-1])

        size = outs[0].shape[-2:]
        outs = [outs[0]] + [F.interpolate(o, size=size, mode="bilinear",
                                          align_corners=False)
                            for o in outs[1:]]
        return self.head(self.dropout(self.fuse(torch.cat(outs, dim=1))))
