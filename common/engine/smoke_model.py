"""Harness fixtures — tiny models used to exercise the training path.

These are **not** benchmark arms. They exist so Step 14's smoke test can run
end to end before any real backbone exists, and so Step 13's acceptance ("two
different arm configs run through the same code path with no `if arm ==`
branches") can be demonstrated rather than asserted: `tiny_cnn` is monolithic
and `tiny_unet` is a two-scale encoder-decoder, and swapping them is a config
edit.

Real arms live under `models/<family>/src/`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common.engine.registry import register_model


@register_model("tiny_cnn")
class TinyCNN(nn.Module):
    """The 2-layer model the spec's --smoke calls for."""

    def __init__(self, in_channels: int = 1, width: int = 16,
                 num_classes: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, num_classes, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model("tiny_unet")
class TinyUNet(nn.Module):
    """Two-scale encoder-decoder, to prove the loop is architecture-agnostic."""

    def __init__(self, in_channels: int = 1, width: int = 16,
                 num_classes: int = 1):
        super().__init__()

        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co),
                nn.ReLU(inplace=True))

        self.enc1 = block(in_channels, width)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = block(width, width * 2)
        self.up = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.dec = block(width * 2, width)
        self.head = nn.Conv2d(width, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d = self.dec(torch.cat([self.up(e2), e1], dim=1))
        return self.head(d)
