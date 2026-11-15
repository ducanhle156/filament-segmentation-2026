"""SAM 2 image encoder (Hiera-B+ trunk) for the registry (Phase 5, Step 24).

    model:
      type: encoder_decoder
      encoder: {name: sam2_hiera_bplus, checkpoint: weights/sam2/sam2.1_hiera_base_plus.pt,
                taps: [1, 2, 3, 4], frozen: false,
                peft: {method: lora, rank: 8, alpha: 16, targets: [qkv, proj]}}
      decoder: {name: fpn, channels: 256}

Image encoder only. The prompt encoder, mask decoder, memory attention and
SAM 2's own FPN neck are never built: this module instantiates the Hiera
trunk (`sam2.modeling.backbones.hieradet.Hiera`) and loads only the
`image_encoder.trunk.*` tensors of the checkpoint. No points, boxes or text
exist anywhere in this arm.

Taps. Hiera is hierarchical: the trunk returns one map per stage at strides
4, 8, 16, 32 with 112, 224, 448, 896 channels (Hiera-B+). `taps: [k]` is the
1-based stage index; `[1, 2, 3, 4]` is the native pyramid, which the shared
FPN decoder consumes without any resampling. The paper notes this asymmetry
against the ViT's resampled pyramid: it is a real architectural difference
between the two backbones, not a decoder confound.

Input. One z-scored channel replicated x3, fed as is — the same rule as
`models/dinov3/src/encoder.py` and for the same reason (zero-mean,
unit-scale already; no trainable adapter in front of the encoder). SAM 2's
own pipeline would subtract ImageNet mean/std from a [0,1] image; that
convention is deliberately not reproduced so both backbones of the arm08
vs arm09 comparison see the identical tensor. Hiera's absolute position
embedding is interpolated to the token grid (128x128 at 512), and the
windowed embedding tiles it, so 512 needs no other change.

Frozen means frozen: as for DINOv3, an encoder with no trainable parameter
stays in eval mode through `model.train()`.

LoRA targets (`common/engine/lora.py`): `qkv` -> `blocks.N.attn.qkv`,
`proj` -> `blocks.N.attn.proj`. The stage-boundary `blocks.N.proj`
(dimension change, not attention) is excluded on purpose.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from common.engine.registry import register_encoder

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = "weights/sam2/sam2.1_hiera_base_plus.pt"
# Hiera-B+ as in sam2/configs/sam2.1/sam2.1_hiera_b+.yaml (the rest of the
# trunk's arguments are the class defaults, which that yaml does not touch).
HIERA_BPLUS = {"embed_dim": 112, "num_heads": 2}
TRUNK_PREFIX = "image_encoder.trunk."


class SAM2HieraEncoder(nn.Module):
    LORA_TARGETS = {
        "qkv": [r".*\.blocks\.\d+\.attn\.qkv"],
        "proj": [r".*\.blocks\.\d+\.attn\.proj"],
        "mlp": [r".*\.blocks\.\d+\.mlp\.layers\.\d+"],
    }

    def __init__(self, checkpoint: str | None = DEFAULT_CHECKPOINT,
                 taps=(1, 2, 3, 4), in_channels: int = 1,
                 config: dict | None = None):
        """`config` (tests only) builds a random-init Hiera from kwargs and
        skips the checkpoint."""
        super().__init__()
        from sam2.modeling.backbones.hieradet import Hiera

        kw = dict(HIERA_BPLUS if config is None else config)
        kw.setdefault("return_interm_layers", True)
        self.trunk = Hiera(**kw)
        self.checkpoint = None
        if config is None and checkpoint:
            path = ROOT / checkpoint
            if not path.exists():
                raise FileNotFoundError(f"{path}: run scripts/fetch_weights.py first")
            sd = torch.load(path, map_location="cpu", weights_only=True)
            sd = sd.get("model", sd)
            trunk_sd = {k[len(TRUNK_PREFIX):]: v for k, v in sd.items()
                        if k.startswith(TRUNK_PREFIX)}
            missing, unexpected = self.trunk.load_state_dict(trunk_sd, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"trunk state mismatch: missing {missing}, "
                                   f"unexpected {unexpected}")
            self.checkpoint = str(checkpoint)

        n_stages = len(self.trunk.stage_ends)
        self.taps = [int(t) for t in taps]
        if any(t < 1 or t > n_stages for t in self.taps):
            raise ValueError(f"taps {self.taps} outside 1..{n_stages}")
        # Hiera.channel_list is coarse -> fine; the contract is fine -> coarse
        chans = list(self.trunk.channel_list)[::-1]
        stride0 = 4                                  # PatchEmbed: 7x7 / stride 4
        q = int(self.trunk.q_stride[0])
        strides = [stride0 * q ** i for i in range(n_stages)]
        self.in_channels = int(in_channels)
        self.out_channels = [chans[t - 1] for t in self.taps]
        self.out_strides = [strides[t - 1] for t in self.taps]

    def train(self, mode: bool = True):
        frozen = not any(p.requires_grad for p in self.parameters())
        return super().train(mode and not frozen)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.shape[1] == 1 and self.in_channels == 1:
            x = x.expand(-1, 3, -1, -1)
        feats = self.trunk(x)                         # one per stage, fine -> coarse
        return [feats[t - 1] for t in self.taps]


@register_encoder("sam2_hiera_bplus")
def sam2_hiera_bplus(**kw) -> SAM2HieraEncoder:
    return SAM2HieraEncoder(**kw)
