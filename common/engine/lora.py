"""LoRA on a registry encoder (Phase 5, Step 23), through `peft`.

    model:
      encoder:
        frozen: false
        peft: {method: lora, rank: 8, alpha: 16,
               targets: [qkv, proj], train_layernorms: true}

Adapters are injected in place (`peft.inject_adapter_in_model`), so the
encoder keeps its class, its `train()` override and its tap contract, and
the harness sees the same `EncoderDecoder` it sees for a frozen arm; only
`requires_grad` changes. Checkpoints hold the whole model state (base
weights + adapters), exactly as for every other arm.

Targets are *logical*. `qkv` means the attention input projections and
`proj` the attention output projection, whatever a backbone calls them —
DINOv3 in `transformers` has `q_proj/k_proj/v_proj/o_proj`, SAM 2's Hiera
has one fused `qkv` and `proj`. Each encoder class lists the regexes for
its own modules in `LORA_TARGETS`; that keeps a stage-boundary `proj` in
Hiera (a dimension change, not attention) out of the adapter set. `mlp`
exists for the "only if rank 8 plateaus" variant the spec allows and is
never a default.

`train_layernorms: true` unfreezes every LayerNorm (and GroupNorm)
parameter in the encoder as well — nearly free, and normalisation
statistics are what breaks first under this domain shift.

`summary(encoder)` gives trainable adapter / norm counts and the trainable
fraction of the encoder, the number the paper reports.
"""

from __future__ import annotations

import re

import torch.nn as nn

# fallback regexes when an encoder does not declare LORA_TARGETS
GENERIC_TARGETS = {
    "qkv": [r".*\.(q_proj|k_proj|v_proj|qkv|query|key|value)$"],
    "proj": [r".*\.(attention|attn)\.(o_proj|proj|out_proj|dense)$"],
    "mlp": [r".*\.mlp\.(up_proj|down_proj|fc1|fc2|lin1|lin2)$"],
}
NORM_TYPES = (nn.LayerNorm, nn.GroupNorm)


def resolve_targets(encoder: nn.Module, targets) -> list[str]:
    """Module names (inside `encoder`) matched by the logical target list."""
    table = dict(GENERIC_TARGETS)
    table.update(getattr(encoder, "LORA_TARGETS", {}) or {})
    patterns = []
    for t in targets:
        if t not in table:
            raise ValueError(f"unknown LoRA target {t!r}; have {sorted(table)}")
        patterns += table[t]
    regs = [re.compile(p) for p in patterns]
    names = [n for n, m in encoder.named_modules()
             if isinstance(m, nn.Linear) and any(r.fullmatch(n) for r in regs)]
    if not names:
        raise ValueError(f"no Linear module of {type(encoder).__name__} matches "
                         f"targets {list(targets)}")
    return names


def apply_lora(encoder: nn.Module, peft_cfg: dict) -> nn.Module:
    from peft import LoraConfig, inject_adapter_in_model

    rank = int(peft_cfg.get("rank", 8))
    alpha = float(peft_cfg.get("alpha", 2 * rank))
    targets = list(peft_cfg.get("targets", ["qkv", "proj"]))
    dropout = float(peft_cfg.get("dropout", 0.0))
    names = resolve_targets(encoder, targets)

    for p in encoder.parameters():                 # base weights: frozen
        p.requires_grad_(False)
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=names, bias="none", init_lora_weights=True)
    inject_adapter_in_model(cfg, encoder, adapter_name="default")
    for n, p in encoder.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
    if peft_cfg.get("train_layernorms", True):
        for m in encoder.modules():
            if isinstance(m, NORM_TYPES):
                for p in m.parameters():
                    p.requires_grad_(True)
    encoder.lora_targets = names
    encoder.lora_config = {"rank": rank, "alpha": alpha, "targets": targets,
                           "dropout": dropout,
                           "train_layernorms": bool(peft_cfg.get("train_layernorms", True))}
    return encoder


def summary(encoder: nn.Module) -> dict:
    """Trainable adapter / norm / total parameters of an encoder."""
    lora = norm = total = trainable = 0
    norm_ids = set()
    for m in encoder.modules():
        if isinstance(m, NORM_TYPES):
            norm_ids.update(id(p) for p in m.parameters())
    for n, p in encoder.named_parameters():
        k = p.numel()
        total += k
        if p.requires_grad:
            trainable += k
            if "lora_" in n:
                lora += k
            elif id(p) in norm_ids:
                norm += k
    return {"encoder_parameters": int(total),
            "encoder_trainable_parameters": int(trainable),
            "encoder_lora_parameters": int(lora),
            "encoder_norm_parameters_trainable": int(norm),
            "encoder_trainable_pct": 100.0 * trainable / total if total else 0.0}
