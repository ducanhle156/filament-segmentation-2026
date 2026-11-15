"""Phase 1 driver: Steps 6-10 of files_FM_specs/02_phase1_preprocessing.md.

    python3 scripts/preprocess.py --config common/configs/preprocess_v1.yaml \
        [--steps 6 7 9 10 fig] [--workers 32] [--limit N]

Steps 7 and 8 are inseparable per image, so they run together under "7".
Step 9 (tiling) runs in the same pass to avoid re-reading the interim cache;
pass --steps 6 alone to only probe the source.

Writes:
    docs/phase1/source_format.md, source_format.json   (step 6)
    data/interim/<stem>.npz                            (steps 7, 8)
    docs/phase1/limb/<stem>.png                        (step fig)
    data/processed/v1/shard_XXXXX.npz, tiles_index.json (step 9)
    data/processed/v1/manifest.json                    (step 10)
    docs/phase1/image_stats.json                       (per-image diagnostics)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.data import limb, preprocess, tiling  # noqa: E402

CFG: dict = {}
IMAGES: list[dict] = []
ANNS: dict = {}


# --------------------------------------------------------------------- helpers

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _init_worker(cfg: dict, images: list[dict], anns: dict) -> None:
    global CFG, IMAGES, ANNS
    CFG, IMAGES, ANNS = cfg, images, anns
    # each worker is one image at a time; BLAS threads would only fight
    os.environ.setdefault("OMP_NUM_THREADS", "1")


# ------------------------------------------------------------- steps 7, 8, 9

def _process_shard(shard_id: int) -> dict:
    """Preprocess and tile one contiguous group of images; write both caches."""
    cfg = CFG
    per = cfg["output"]["images_per_shard"]
    group = IMAGES[shard_id * per:(shard_id + 1) * per]
    if not group:
        return {"shard_id": shard_id, "n_tiles": 0, "records": [], "stats": []}

    root = Path(cfg["_root"])
    raw_dir = root / cfg["source"]["raw_dir"]
    interim = root / cfg["output"]["interim_dir"]
    processed = root / cfg["output"]["processed_dir"]
    t = cfg["tiling"]

    imgs, masks, valids, records, stats = [], [], [], [], []
    for im in group:
        r = preprocess.preprocess_image(
            raw_dir / im["file_name"], ANNS.get(im["id"], []),
            im["height"], im["width"],
            limb_method=cfg["limb"]["method"],
            n_bins=cfg["limb"]["n_bins"],
            smooth_bins=cfg["limb"]["smooth_bins"],
            floor_frac=cfg["limb"]["floor_frac"],
            flat_bins=cfg["limb"]["flat_bins"],
            analysis_scale=cfg["limb"]["disk"]["analysis_scale"])
        stem = Path(im["file_name"]).stem
        st = dict(r["stats"], image_id=im["id"], file_name=im["file_name"])
        preprocess.save_interim(interim / f"{stem}.npz", r["img"], r["valid"],
                                r["mask"], st)

        rng = tiling.image_rng(cfg["seed"], im["id"])
        tiles = tiling.tile_image(
            r["img"], r["mask"], r["valid"], image_id=im["id"],
            tile=t["tile"], overlap=t["overlap"], keep_ratio=t["keep_ratio"],
            bg_keep_rate=t["bg_keep_rate"], rng=rng)
        st["n_tiles_kept"] = len(tiles)
        st["n_tiles_grid"] = len(tiling.tile_grid((im["height"], im["width"]),
                                                  t["tile"], t["overlap"]))
        stats.append(st)
        for k, tl in enumerate(tiles):
            imgs.append(tl["img"].astype(np.float16))
            masks.append(np.packbits(tl["mask"]))
            valids.append(np.packbits(tl["valid"]))
            records.append({"shard": shard_id, "index": len(records),
                            **{f: tl[f] for f in ("image_id", "row", "col",
                                                  "y0", "x0", "valid_frac",
                                                  "fg_pixels", "kind")},
                            "file_name": im["file_name"]})

    processed.mkdir(parents=True, exist_ok=True)
    out = processed / f"shard_{shard_id:05d}.npz"
    if imgs:
        np.savez(out,
                 img=np.stack(imgs), mask=np.stack(masks),
                 valid=np.stack(valids),
                 image_id=np.asarray([r["image_id"] for r in records]),
                 row=np.asarray([r["row"] for r in records], np.int16),
                 col=np.asarray([r["col"] for r in records], np.int16),
                 y0=np.asarray([r["y0"] for r in records], np.int32),
                 x0=np.asarray([r["x0"] for r in records], np.int32),
                 tile=np.asarray(t["tile"], np.int32))
    else:  # a shard can legitimately keep nothing only if it has no images
        np.savez(out, img=np.zeros((0, t["tile"], t["tile"]), np.float16))
    for i, r in enumerate(records):
        r["index"] = i
    return {"shard_id": shard_id, "n_tiles": len(records),
            "records": records, "stats": stats}


# ----------------------------------------------------------------- step "fig"

def limb_figures(root: Path, cfg: dict, image_ids: list[str], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    meta = {im["id"]: im for im in IMAGES}
    raw_dir = root / cfg["source"]["raw_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for iid in image_ids:
        im = meta[iid]
        with Image.open(raw_dir / im["file_name"]) as pil:
            gray = limb.to_gray(np.asarray(pil.convert("RGB")))
        d = limb.fit_disk(gray)
        geom = (d["cx"], d["cy"], d["r"])
        mask = limb.disk_mask(gray.shape, *geom)
        flat = limb.remove_limb_darkening(gray, mask, cfg["limb"]["method"],
                                          cfg["limb"]["n_bins"],
                                          cfg["limb"]["smooth_bins"],
                                          cfg["limb"]["floor_frac"],
                                          geometry=geom)
        before = limb.profile_flatness(gray, mask, 128, geom)
        after = limb.profile_flatness(flat, mask, 128, geom)

        show_b = np.where(mask, gray, np.nan)
        show_a = np.where(mask, flat, np.nan)
        fig, ax = plt.subplots(1, 3, figsize=(16, 5.4), dpi=130)
        for a, im_arr, title, ref in (
                (ax[0], show_b, "before", before), (ax[1], show_a, "after", after)):
            lo, hi = np.nanpercentile(im_arr, [1, 99])
            a.imshow(im_arr, cmap="gray", vmin=lo, vmax=hi)
            a.set_title(f"{title}  (radial rms dev {ref['rms_dev_full']:.4f})",
                        fontsize=10)
            a.axis("off")
        x = (np.arange(128) + 0.5) / 128
        ax[2].plot(x, before["profile"], label="before")
        ax[2].plot(x, after["profile"], label=f"after ({cfg['limb']['method']})")
        ax[2].axhline(1.0, color="k", lw=0.6, ls=":")
        scale = cfg["limb"]["disk"]["analysis_scale"]
        if scale < 1.0:
            ax[2].axvspan(scale, 1.0, color="0.85", zorder=0)
            ax[2].text(scale, 1.28, f" excluded (r/R > {scale})", fontsize=8,
                       va="top")
        ax[2].set_xlabel("r / R"); ax[2].set_ylabel("radial median / inner level")
        ax[2].set_ylim(0, 1.35); ax[2].legend(fontsize=9)
        ax[2].set_title(f"R = {d['r']:.1f} px, fit resid p95 "
                        f"{d['resid_p95_px']:.2f} px", fontsize=10)
        fig.suptitle(im["file_name"], fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"{Path(im['file_name']).stem}.png",
                    bbox_inches="tight")
        plt.close(fig)


# ------------------------------------------------------------------ step 6 doc

def write_source_format(probe: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "source_format.json").write_text(json.dumps(probe, indent=1) + "\n")
    b = probe["block_artifact"]
    g = probe["disk_geometry"]
    cap = probe["burned_in_captions"]
    verdict = ("JPEG, 8-bit, already compressed" if not probe["is_fits"]
               else "FITS")
    severity = ("negligible" if b["ratio_mean"] < 1.05 else
                "mild" if b["ratio_mean"] < 1.25 else
                "moderate" if b["ratio_mean"] < 2.0 else "severe")
    md = f"""# Step 6 — Source format verdict

Probe: `common/data/preprocess.probe_source`, {probe['n_sampled']} images
sampled at seed {probe['seed']} from {probe['n_files']} in `data/`.
Raw numbers: `docs/phase1/source_format.json`.

## Verdict

**{verdict}.** Format {'/'.join(probe['format'])}, PIL mode
{'/'.join(probe['pil_mode'])}, {'/'.join(probe['size'])}, dtype
`{probe['dtype']}` ({probe['bit_depth']} bit), value range
{probe['value_range'][0]}–{probe['value_range'][1]}.

There is no FITS anywhere in the dataset, so **no header is available**: the
disk centre and radius are fitted from the pixels
(`common/data/limb.fit_disk`), and there is no photometric calibration, no
exposure keyword and no observation-time metadata beyond the filename stem
and the caption burned into the frame.

The three channels are a greyscale rendering (channel spread reaches
{probe['max_channel_spread']} DN only in the coloured NISP logo); preprocessing
averages them, which also suppresses JPEG chroma noise.

## Block-artifact severity at barb scale — the number

High-frequency energy on the 8×8 DCT seams versus block interiors, measured
on on-disk pixels only (1.00 = no block signature):

| | ratio |
|---|---|
| horizontal | {b['horizontal_mean']:.3f} |
| vertical | {b['vertical_mean']:.3f} |
| **mean over {probe['n_sampled']} images** | **{b['ratio_mean']:.3f}** |
| range across images | {b['ratio_min']:.3f} – {b['ratio_max']:.3f} |

**Severity: {severity}.**

Why this matters: the Phase 0 audit measured a median filament width of ~10 px
at 2048×2048, and barbs are thinner still — the same order as the 8 px DCT
block. A strong block signature would mean barb-width detail is already
destroyed in the raw data, capping every model in the benchmark regardless of
capacity. This ratio must be stated in the paper as an upper bound on
achievable boundary accuracy.

## Disk geometry across the sample

| | min | mean | max |
|---|---|---|---|
| centre x (px) | {g['cx'][0]:.2f} | {g['cx'][1]:.2f} | {g['cx'][2]:.2f} |
| centre y (px) | {g['cy'][0]:.2f} | {g['cy'][1]:.2f} | {g['cy'][2]:.2f} |
| radius R (px) | {g['r'][0]:.2f} | {g['r'][1]:.2f} | {g['r'][2]:.2f} |

Worst circle-fit residual (p95 over rim points) in the sample:
{g['fit_resid_p95_px_max']:.2f} px.

## Burned-in captions

Every frame carries the same four burned-in captions (observatory name, site
and UT, "GONG H-Alpha / 6562.8 A", and the coloured NISP logo):
{cap['n_pixels']} pixels are saturated in *every* sampled image, spanning
rows {cap.get('bbox_y')} and columns {cap.get('bbox_x')}. The closest of them
sits at r/R = {cap.get('min_r_over_R', float('nan')):.3f}, and
{cap.get('n_inside_disk', 0)} of them fall inside the disk — so the on-disk
mask excludes the captions entirely and no text reaches the model.
"""
    (out_dir / "source_format.md").write_text(md)


# ------------------------------------------------------------------ main

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path,
                   default=ROOT / "common/configs/preprocess_v1.yaml")
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--steps", nargs="+", default=["6", "7", "9", "10", "fig"])
    p.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    p.add_argument("--limit", type=int, default=0,
                   help="process only the first N images (smoke test)")
    args = p.parse_args()

    root = args.root
    cfg = yaml.safe_load(args.config.read_text())
    cfg["_root"] = str(root)
    steps = set(args.steps)
    docs = root / "docs" / "phase1"
    processed = root / cfg["output"]["processed_dir"]

    coco = json.loads((root / cfg["source"]["coco"]).read_text())
    images = sorted(coco["images"], key=lambda im: im["file_name"])
    if args.limit:
        images = images[:args.limit]
    anns = preprocess.annotations_by_image(coco)
    anns = {k: v for k, v in anns.items()}
    global IMAGES, ANNS, CFG
    CFG, IMAGES, ANNS = cfg, images, anns
    print(f"{len(images)} images, config {args.config.name}, "
          f"{args.workers} workers", flush=True)

    if "6" in steps:
        t0 = time.time()
        probe = preprocess.probe_source(root / cfg["source"]["raw_dir"],
                                        seed=cfg["seed"])
        write_source_format(probe, docs)
        print(f"step 6: JPEG block ratio {probe['block_artifact']['ratio_mean']:.3f} "
              f"({time.time() - t0:.0f}s) -> {docs / 'source_format.md'}", flush=True)

    if "7" in steps or "9" in steps:
        per = cfg["output"]["images_per_shard"]
        n_shards = (len(images) + per - 1) // per
        t0 = time.time()
        results = []
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_worker,
                                 initargs=(cfg, images, anns)) as ex:
            for i, r in enumerate(ex.map(_process_shard, range(n_shards)), 1):
                results.append(r)
                if i % 20 == 0 or i == n_shards:
                    print(f"  shard {i}/{n_shards} "
                          f"({time.time() - t0:.0f}s)", flush=True)
        results.sort(key=lambda r: r["shard_id"])
        records = [r for res in results for r in res["records"]]
        stats = [s for res in results for s in res["stats"]]
        (processed / "tiles_index.json").write_text(
            json.dumps({"tile": cfg["tiling"]["tile"],
                        "overlap": cfg["tiling"]["overlap"],
                        "n_tiles": len(records), "tiles": records}, indent=0) + "\n")
        docs.mkdir(parents=True, exist_ok=True)
        (docs / "image_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
        print(f"steps 7-9: {len(records)} tiles from {len(stats)} images "
              f"({time.time() - t0:.0f}s)", flush=True)

        # dataset-level geometry sanity check: every frame is the same
        # instrument at the same scale, so R must be tightly clustered. A
        # frame that drifts off it is a fit failure, not an unusual Sun.
        R = np.array([s["disk"]["r"] for s in stats])
        med = float(np.median(R))
        off = [s["file_name"] for s in stats
               if abs(s["disk"]["r"] - med) > 0.02 * med]
        unconv = [s["file_name"] for s in stats
                  if not s["disk"].get("converged", True)]
        print(f"  disk fit: R median {med:.2f} px, range "
              f"{R.min():.2f}-{R.max():.2f}; {len(off)} frames >2% off median, "
              f"{len(unconv)} not converged", flush=True)
        for f in (off + unconv)[:10]:
            print(f"    suspect: {f}", flush=True)

    if "fig" in steps:
        rng = np.random.default_rng(cfg["seed"])
        ids = [images[i]["id"] for i in
               sorted(rng.choice(len(images), size=min(10, len(images)),
                                 replace=False))]
        limb_figures(root, cfg, ids, docs / "limb")
        (docs / "limb" / "selection.json").write_text(
            json.dumps({"seed": cfg["seed"], "image_ids": ids}, indent=1) + "\n")
        print(f"step fig: {len(ids)} before/after figures -> {docs / 'limb'}",
              flush=True)

    if "10" in steps:
        shards = sorted(processed.glob("shard_*.npz"))
        digests = {s.name: sha256_file(s) for s in shards}
        index = processed / "tiles_index.json"
        digests[index.name] = sha256_file(index)
        listing = "".join(f"{k}  {v}\n" for k, v in sorted(digests.items()))
        checksum = hashlib.sha256(listing.encode()).hexdigest()
        idx = json.loads(index.read_text())
        stats = json.loads((docs / "image_stats.json").read_text())
        manifest = {
            "version": cfg["version"],
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(root),
            "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "n_images": len(stats),
            "n_tiles": idx["n_tiles"],
            "n_tiles_foreground": sum(1 for t in idx["tiles"] if t["kind"] == "foreground"),
            "n_tiles_background": sum(1 for t in idx["tiles"] if t["kind"] == "background"),
            "n_shards": len(shards),
            "limb_method": cfg["limb"]["method"],
            "tile": cfg["tiling"]["tile"],
            "overlap": cfg["tiling"]["overlap"],
            "seed": cfg["seed"],
            "files": digests,
            "checksum": f"sha256:{checksum}",
        }
        (processed / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"step 10: {manifest['n_tiles']} tiles in {len(shards)} shards, "
              f"checksum {checksum[:16]}... -> {processed / 'manifest.json'}",
              flush=True)


if __name__ == "__main__":
    main()
