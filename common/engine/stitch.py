"""Full-disk inference by tiling and stitching (Phase 2, Step 13).

Evaluation is per *image*, not per tile: the metrics contract returns
per-image values and the paper reports per-image spread. So a test disk is
covered by the **complete** tile grid — not the retention-filtered training
subset — the tiles are run through the model, and the overlapping predictions
are averaged back into one full-disk map.

Averaging (rather than taking the last writer) is what removes seams: with
25% overlap every interior pixel is predicted by two or four tiles, and a
discontinuity at a tile edge would otherwise show up as a straight line in
the output and a spurious filament boundary in the metrics.
"""

from __future__ import annotations

import numpy as np

from common.data.tiling import stitch, tile_grid


def predict_full_disk(model, img: np.ndarray, valid: np.ndarray, tile: int,
                      overlap: float, device, batch_size: int = 8,
                      amp: bool = True) -> np.ndarray:
    """Return a full-disk probability map.

    `img` is [H,W] float32, or [C,H,W] when the arm reads side channels
    (`common/data/channels.py`): the first channel is the H-alpha disk and
    every channel is cropped by the same tile window."""
    import torch

    shape = img.shape[-2:]
    grid = tile_grid(shape, tile, overlap)
    patches = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(grid), batch_size):
            chunk = grid[i:i + batch_size]
            batch = np.stack([img[..., y:y + tile, x:x + tile] for _, _, y, x in chunk])
            t = torch.from_numpy(np.ascontiguousarray(batch))
            if t.ndim == 3:
                t = t.unsqueeze(1)
            t = t.to(device)
            if amp and device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(t)
            else:
                logits = model(t)
            probs = torch.sigmoid(logits.float()).squeeze(1).cpu().numpy()
            for (_, _, y, x), p in zip(chunk, probs):
                patches.append((y, x, p))

    prob = stitch(patches, shape, reduce="mean")
    return np.where(valid, prob, 0.0).astype(np.float32)


def binarize(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return prob >= threshold
