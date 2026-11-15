"""COCO compressed RLE, the submission's mask format.

`encode(mask)` -> the `counts` string the CSV cell holds (no quotes, size is
implicitly 2048 x 2048); `decode(counts, h, w)` -> bool mask. pycocotools is
used when importable; the pure-Python pair below implements the same
scheme (column-major runs, 5-bit groups with a continuation bit, deltas
from the run two back) and is what the unit test checks it against.
"""
from __future__ import annotations

import numpy as np

try:
    from pycocotools import mask as _mu
except ImportError:          # pragma: no cover
    _mu = None


def runs_from_mask(mask: np.ndarray) -> list[int]:
    """Column-major run lengths, starting with a (possibly empty) run of 0s."""
    flat = np.asfortranarray(mask.astype(np.uint8)).ravel(order="F")
    if flat.size == 0:
        return [0]
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(bounds).tolist()
    if flat[0] == 1:
        runs = [0] + runs
    return runs


def mask_from_runs(runs: list[int], h: int, w: int) -> np.ndarray:
    vals = np.zeros(sum(runs), dtype=np.uint8)
    pos, v = 0, 0
    for r in runs:
        if v:
            vals[pos:pos + r] = 1
        pos += r
        v ^= 1
    return vals.reshape((h, w), order="F").astype(bool)


def runs_to_string(runs: list[int]) -> str:
    out = []
    for i, x in enumerate(runs):
        if i > 2:
            x -= runs[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def string_to_runs(s: str) -> list[int]:
    runs: list[int] = []
    p = 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(runs) > 2:
            x += runs[-2]
        runs.append(x)
    return runs


def encode(mask: np.ndarray) -> str:
    if _mu is not None:
        r = _mu.encode(np.asfortranarray(mask.astype(np.uint8)))
        return r["counts"].decode("ascii")
    return runs_to_string(runs_from_mask(mask))


def decode(counts: str, h: int = 2048, w: int = 2048) -> np.ndarray:
    if _mu is not None:
        return _mu.decode({"size": [h, w], "counts": counts.encode("ascii")}).astype(bool)
    return mask_from_runs(string_to_runs(counts), h, w)


def polygons_to_mask(polys: list[list[float]], h: int = 2048, w: int = 2048) -> np.ndarray:
    """COCO polygon list -> bool mask, the way pycocotools rasterizes GT."""
    if _mu is not None:
        rles = _mu.frPyObjects([p for p in polys if len(p) >= 6], h, w)
        return _mu.decode(_mu.merge(rles)).astype(bool)
    from PIL import Image, ImageDraw                      # fallback: PIL fill
    im = Image.new("1", (w, h), 0)
    d = ImageDraw.Draw(im)
    for p in polys:
        if len(p) >= 6:
            d.polygon(list(zip(p[0::2], p[1::2])), fill=1, outline=1)
    return np.array(im, dtype=bool)
