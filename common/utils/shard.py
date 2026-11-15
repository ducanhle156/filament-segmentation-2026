"""Split an embarrassingly-parallel image list across PBS jobs.

Inference-only stages — zero-shot SAM 2 / SAM 3 (Step 20), full-disk
galleries, any re-evaluation of a checkpoint — have no state between images,
so they scale by running N single-GPU jobs over disjoint image subsets and
merging their per-image rows (`scripts/merge_shards.py`). No DDP needed.

    ids = shard_ids(all_ids, "2/4")     # the third of four shards
"""

from __future__ import annotations


def parse_shard(spec: str | None) -> tuple[int, int]:
    """'I/N' -> (I, N) with 0 <= I < N; None -> (0, 1)."""
    if not spec:
        return 0, 1
    i, _, n = str(spec).partition("/")
    i, n = int(i), int(n)
    if n < 1 or not 0 <= i < n:
        raise ValueError(f"bad shard spec {spec!r}: want I/N with 0 <= I < N")
    return i, n


def shard_ids(ids, spec: str | None) -> list:
    """Deterministic, order-preserving strided split of a sorted id list."""
    i, n = parse_shard(spec)
    ordered = sorted(ids)
    return ordered[i::n]


def shard_suffix(spec: str | None) -> str:
    """File suffix for a shard's output: '' for the whole set, '_shard2of4'."""
    i, n = parse_shard(spec)
    return "" if n == 1 else f"_shard{i}of{n}"
