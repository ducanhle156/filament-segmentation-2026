"""Step 1 - preprocessing. Uses only the image.

1. Fit the solar limb (centre and radius) from the image itself.
2. Remove limb darkening: divide by the radial median brightness profile, so
   quiet Sun has the same level at disk centre and near the limb.
3. z-score over the analysis disk (r <= 0.95 R). Outside it the image is
   set to a constant and nothing is predicted: filaments at the extreme limb
   are foreshortened and sit under the chromospheric rim.

The parameters are the frozen contract `common/configs/preprocess_v1.yaml`,
the same one the training images went through.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from common.data import preprocess as _pp
from .settings import H, W

ROOT = Path(__file__).resolve().parents[1]
_CFG = yaml.safe_load((ROOT / "common" / "configs" / "preprocess_v1.yaml").read_text())


def preprocess_jpeg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """-> (image float32 [2048, 2048] in z-score units, analysis-disk mask bool [2048, 2048])."""
    lim = _CFG["limb"]
    r = _pp.preprocess_image(Path(path), [], H, W, limb_method=lim["method"], n_bins=lim["n_bins"],
                             smooth_bins=lim["smooth_bins"], floor_frac=lim["floor_frac"],
                             flat_bins=lim["flat_bins"], analysis_scale=lim["disk"]["analysis_scale"])
    return r["img"].astype(np.float32), r["valid"].astype(bool)
