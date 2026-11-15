"""Model registry (Phase 2, Step 13).

Rule 4's gate: swapping U-Net for ConvNeXt must be a config edit, so model
construction is a lookup by string key and there is no `if arm ==` anywhere in
the training path. Adding a backbone means one `@register_encoder` line.

Two model shapes are supported, matching `00_conventions.md`:

`monolithic`      one registered module that maps [B,1,H,W] -> logits
                  [B,1,H,W]. U-Net, EdgeAttNet, ConvNeXt-UPerNet.
`encoder_decoder` a registered encoder exposing `out_channels` / `out_strides`
                  and returning a list of taps, plus a registered decoder that
                  consumes them. DINOv3 and SAM 2 arms.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

ENCODERS: dict[str, Callable] = {}
DECODERS: dict[str, Callable] = {}
MODELS: dict[str, Callable] = {}
# A family that cannot run under the shared loop (a detector with its own
# trainer, e.g. YOLO) registers a trainer under its `model.type`; the harness
# looks it up by that key and falls back to `train_fold` for everything else.
TRAINERS: dict[str, Callable] = {}

_ROOT = Path(__file__).resolve().parents[2]
_discovered = False
_import_errors: dict[str, str] = {}


def discover_families(root: Path | None = None) -> None:
    """Import every `models/<family>/src/*.py` so its decorators run.

    Registration has to happen before a config is resolved, and the config
    names only a registry key -- so the keys have to come to us. Scanning is
    what keeps rule 4's gate honest: adding a backbone is one
    `@register_model` line in a new file, with nothing to wire up and no arm
    list to edit anywhere.

    A family whose dependencies are not installed (SAM 2, DINOv3) must not
    take the run down with it -- U-Net does not need them. Those import
    failures are recorded and reported by name in the KeyError of any build
    that actually wanted a missing key, so a real breakage still surfaces
    loudly instead of looking like a typo.
    """
    global _discovered
    root = Path(root) if root is not None else _ROOT
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:                                    # the shared heads in common/decoders
        import common.decoders  # noqa: F401
    except Exception as exc:
        _import_errors["common/decoders"] = f"{type(exc).__name__}: {exc}"

    for path in sorted((root / "models").glob("*/src/*.py")):
        if path.name.startswith("_"):
            continue
        mod_name = f"_fmfam_{path.parent.parent.name}_{path.stem}"
        if mod_name in sys.modules:
            continue
        try:
            spec = importlib.util.spec_from_file_location(mod_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
        except Exception as exc:                     # missing family deps
            sys.modules.pop(mod_name, None)
            _import_errors[str(path.relative_to(root))] = (
                f"{type(exc).__name__}: {exc}")
    _discovered = True


def _register(table: dict, name: str):
    def deco(fn):
        if name in table:
            raise ValueError(f"{name!r} already registered")
        table[name] = fn
        return fn
    return deco


def register_encoder(name: str):
    return _register(ENCODERS, name)


def register_decoder(name: str):
    return _register(DECODERS, name)


def register_model(name: str):
    return _register(MODELS, name)


def register_trainer(model_type: str):
    return _register(TRAINERS, model_type)


def get_trainer(model_type: str):
    """The training function for a `model.type`, after family discovery."""
    if model_type not in TRAINERS and not _discovered:
        discover_families()
    return TRAINERS.get(model_type)


def _build(table: dict, kind: str, spec: dict, **extra):
    spec = dict(spec)
    name = spec.pop("name", None)
    if name is None:
        raise ValueError(f"{kind} config has no 'name'")
    if name not in table and not _discovered:
        discover_families()
    if name not in table:
        msg = f"unknown {kind} {name!r}; registered: {sorted(table)}"
        if _import_errors:
            skipped = "; ".join(f"{k} ({v})" for k, v in _import_errors.items())
            msg += f"; family modules that failed to import: {skipped}"
        raise KeyError(msg)
    return table[name](**spec, **extra)


def build_encoder(spec: dict):
    return _build(ENCODERS, "encoder", spec)


def build_decoder(spec: dict, in_channels, in_strides, num_classes: int = 1):
    return _build(DECODERS, "decoder", spec, in_channels=in_channels,
                  in_strides=in_strides, num_classes=num_classes)


def build_model(model_cfg: dict):
    """Build whatever the config describes. The only construction path."""
    import torch.nn as nn

    kind = model_cfg.get("type", "monolithic")
    if kind == "monolithic" or kind in TRAINERS or (
            not _discovered and kind not in ("encoder_decoder",)):
        # a family with its own trainer still builds through the registry
        return _build(MODELS, "model", model_cfg.get("arch", model_cfg))

    if kind != "encoder_decoder":
        raise ValueError(f"unknown model type {kind!r}")

    enc_cfg = dict(model_cfg["encoder"])
    peft = enc_cfg.pop("peft", None)
    frozen = enc_cfg.pop("frozen", False)
    encoder = build_encoder(enc_cfg)
    decoder = build_decoder(model_cfg["decoder"], encoder.out_channels,
                            encoder.out_strides,
                            model_cfg.get("num_classes", 1))

    method = (peft or {}).get("method", "none") if peft else "none"
    if method not in ("none", "lora", "full"):
        raise ValueError(f"unknown peft.method {method!r}; none | lora | full")
    if method != "none" and frozen:
        raise ValueError(f"encoder.frozen: true contradicts peft.method: {method}")
    if method == "lora":
        from common.engine.lora import apply_lora
        apply_lora(encoder, peft)
    elif method == "full":                        # Step 25: every weight trains
        for p in encoder.parameters():
            p.requires_grad_(True)
    elif frozen:
        for p in encoder.parameters():
            p.requires_grad_(False)
    # Randomly initialised modules *inside* a pretrained encoder (Phase 8:
    # the magnetic adapter beside the Hiera trunk) are listed by the encoder
    # in FRESH_MODULES. They have nothing pretrained to protect, so PEFT /
    # frozen never switches them off, and the training loop trains them at
    # the head's learning rate (`train.lr`), not `lr_encoder`.
    for name, p in encoder.named_parameters():
        if is_fresh(encoder, name):
            p.requires_grad_(True)

    class EncoderDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return EncoderDecoder()


def is_fresh(encoder, param_name: str) -> bool:
    """True when `param_name` (inside `encoder`) belongs to a FRESH_MODULES
    submodule: a randomly initialised part of an otherwise pretrained encoder."""
    prefixes = tuple(getattr(encoder, "FRESH_MODULES", ()) or ())
    return any(param_name == pf or param_name.startswith(pf + ".") for pf in prefixes)


def parameter_counts(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out = {"total_parameters": int(total),
           "trainable_parameters": int(trainable),
           "trainable_fraction": float(trainable / total) if total else 0.0}
    encoder = getattr(model, "encoder", None)
    if encoder is not None:                      # Step 23: trainable % of encoder
        from common.engine.lora import summary
        out.update(summary(encoder))
        fresh = sum(p.numel() for n, p in encoder.named_parameters()
                    if is_fresh(encoder, n))
        if fresh:
            out["encoder_fresh_parameters"] = int(fresh)
    return out
