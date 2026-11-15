"""Process-level runtime settings (Phase 2, Step 13).

Thread oversubscription is a real, measured trap on this cluster and not a
micro-optimisation. A PBS chunk here grants 32 CPUs, so torch defaults to 32
intra-op threads; on the small tensors this benchmark uses (512x512 single
channel, ~1 MB) the synchronisation cost swamps the arithmetic:

    torch.set_num_threads(1)   cldice_loss on one tile   0.21 s
    torch.set_num_threads(4)                             0.59 s
    torch.set_num_threads(32)                            27.1 s     <- default

That is a 129x slowdown on the default setting, and it compounds: every
dataloader worker spawns its own thread pool, so N workers x 32 threads fight
over the same cores. Entry points call `configure_threads` before building
anything.
"""

from __future__ import annotations

import os

DEFAULT_THREADS = 4


def configure_threads(n_threads: int | None = None) -> int:
    """Pin torch intra-op threads. Returns the value used.

    Precedence: explicit argument, then OMP_NUM_THREADS if the scheduler set
    it, then a small default. Never the full core count.
    """
    if n_threads is None:
        env = os.environ.get("OMP_NUM_THREADS")
        n_threads = int(env) if env and env.isdigit() else DEFAULT_THREADS
    n_threads = max(1, int(n_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    try:
        import torch
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(1) if _interop_unset() else None
    except ImportError:
        pass
    return n_threads


def _interop_unset() -> bool:
    """set_num_interop_threads throws once parallel work has started."""
    try:
        import torch
        return torch.get_num_interop_threads() > 1
    except Exception:
        return False
