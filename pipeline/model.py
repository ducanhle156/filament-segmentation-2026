"""Model construction and weight loading.

Architecture (configs/kag13_sam2_pqsel.yaml): the SAM 2.1 Hiera-B+ image
encoder (hierarchical ViT, 68.6 M weights, kept frozen) with LoRA rank-4
adapters on the attention qkv / proj layers and trainable LayerNorms, feeding
a 4-level FPN decoder (5.2 M weights) that outputs one filament-probability
channel. Input: 512 x 512 tiles of the preprocessed disk, one channel.

Weights in the repository
  weights/sam2/sam2.1_hiera_base_plus.trunk.pt.part*   the public SAM 2.1 Hiera-B+ encoder trunk
        (Apache-2.0, facebookresearch/sam2), split into parts under 100 MB so a plain `git clone` gets them
  weights/kag13_fold<F>.pt                           what training changed for fold model F: LoRA
        adapters, LayerNorms, FPN decoder (22 MB each)
`load_fold_model` joins the parts once, builds the network, loads the fold file on top and checks that
every tensor not in the fold file is one of the frozen encoder tensors.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "kag13_sam2_pqsel.yaml"
BASE = ROOT / "weights" / "sam2" / "sam2.1_hiera_base_plus.pt"
FOLDS = (0, 1, 2, 3, 4)


def ensure_base() -> Path:
    """Join the encoder parts into the file the encoder module reads (done once)."""
    if not BASE.exists():
        parts = sorted(BASE.parent.glob("sam2.1_hiera_base_plus.trunk.pt.part*"))
        if not parts:
            raise FileNotFoundError(f"no encoder parts in {BASE.parent}")
        with open(BASE, "wb") as out:
            for p in parts:
                out.write(p.read_bytes())
    return BASE


def load_config() -> dict:
    from common.utils.config import load_config as _load
    return _load(CONFIG, ROOT)


def load_fold_model(fold: int, device=None):
    """-> (model in eval mode on `device`, resolved config)."""
    import torch
    from common.engine.registry import build_model
    ensure_base()
    cfg = load_config()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"])
    ck = torch.load(ROOT / "weights" / f"kag13_fold{fold}.pt", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    if unexpected or sorted(missing) != sorted(ck["frozen_keys"]):
        raise RuntimeError(f"fold {fold}: weight file does not match the architecture "
                           f"({len(unexpected)} unexpected, {len(missing)} missing vs {len(ck['frozen_keys'])} frozen)")
    return model.to(device).eval(), cfg
