"""Arm configuration loading (rule 4: an arm is a YAML file, never a script).

`common/configs/base.yaml` holds the defaults every arm inherits; an arm file
overrides only what differs. Merging is a deep merge, so an arm can set
`model.encoder.peft.rank` without restating the rest of the encoder block.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import yaml

BASE = "common/configs/base.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; `override` wins at the leaves."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(arm_path, root: Path, overrides: dict | None = None) -> dict:
    """base.yaml + the arm file + explicit overrides, in that order."""
    root = Path(root)
    base = yaml.safe_load((root / BASE).read_text()) or {}
    arm = yaml.safe_load(Path(arm_path).read_text()) or {} if arm_path else {}
    cfg = deep_merge(base, arm)
    cfg = deep_merge(cfg, overrides or {})
    cfg["_arm_config_file"] = str(arm_path) if arm_path else None
    cfg["_config_sha256"] = config_hash(cfg)
    return cfg


# Keys that change how fast or where a run executes, never what it computes.
# They are left out of the hash (and of `differing_keys`) so that moving a job
# to another GPU class or giving the loader more workers does not turn one
# experiment into "a different config" — and does not invalidate a `last.pt`
# written before the change (2026-08-25, compute-strategy review).
NON_SCIENTIFIC_KEYS = frozenset({
    "compute",
    "train.num_workers", "train.torch_threads",
    "train.resume", "train.save_checkpoint",
    "eval.save_predictions",
})


def _strip(cfg: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        key = f"{prefix}{k}"
        if key in NON_SCIENTIFIC_KEYS:
            continue
        out[k] = _strip(v, f"{key}.") if isinstance(v, dict) else v
    return out


def config_hash(cfg: dict) -> str:
    """Hash of the resolved config, ignoring bookkeeping and runtime keys.

    Two runs with the same hash saw the same settings — that is what makes
    "one variable per comparison" checkable after the fact rather than
    asserted in a commit message. Runtime knobs (`NON_SCIENTIFIC_KEYS`) are
    excluded: they cannot change a result, only its wall clock.
    """
    payload = yaml.safe_dump(_strip(cfg), sort_keys=True,
                             default_flow_style=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def get(cfg: dict, dotted: str, default=None):
    """`get(cfg, "model.encoder.name")`."""
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def differing_keys(a: dict, b: dict, prefix: str = "") -> list[str]:
    """Dotted keys where two resolved configs differ.

    The gate for rule 1: two arms in a comparison must differ in exactly one
    of backbone, decoder, adaptation method or loss — and this is how you
    check that claim instead of trusting it.
    """
    out = []
    for k in sorted(set(a) | set(b)):
        if k.startswith("_"):
            continue
        key = f"{prefix}{k}"
        if key in NON_SCIENTIFIC_KEYS:
            continue
        av, bv = a.get(k), b.get(k)
        if isinstance(av, dict) and isinstance(bv, dict):
            out.extend(differing_keys(av, bv, f"{key}."))
        elif av != bv:
            out.append(key)
    return out
