"""Write the deployable weights of the five kag13 fold models into the submission repository.

A training checkpoint (283 MB) holds the whole network, but training only
changed the LoRA adapters, the LayerNorms and the FPN decoder; the Hiera
encoder trunk is the public SAM 2.1 checkpoint, bit for bit. So the
repository stores the trunk once (split into parts under 100 MB) and, per
fold, only the tensors that differ from a freshly built model.

Checks: (1) every tensor left out equals the base-initialised tensor;
(2) a model rebuilt by the package loader equals the checkpoint tensor for tensor.

    python kaggle/try_error/export_weights.py --out kaggle/submission_repo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

import lib  # noqa: F401  (sets sys.path for the project imports)
from paths import ROOT, RUNS
from common.engine.registry import build_model
from common.utils.config import load_config

PART = 95 * 1024 * 1024
PREFIX = "image_encoder.trunk."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "kaggle" / "submission_repo")
    a = ap.parse_args()
    wdir = a.out / "weights"
    (wdir / "sam2").mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "kaggle/configs/kag13_sam2_pqsel.yaml", ROOT)
    # encoder trunk only, same key names as the public checkpoint so the encoder module reads it unchanged
    full = torch.load(ROOT / cfg["model"]["encoder"]["checkpoint"], map_location="cpu", weights_only=True)
    full = full.get("model", full)
    trunk = {"model": {k: v for k, v in full.items() if k.startswith(PREFIX)}}
    tmp = wdir / "sam2" / "trunk.tmp"
    torch.save(trunk, tmp)
    blob = tmp.read_bytes()
    tmp.unlink()
    for old in (wdir / "sam2").glob("*.part*"):
        old.unlink()
    n_parts = 0
    for i in range(0, len(blob), PART):
        (wdir / "sam2" / f"sam2.1_hiera_base_plus.trunk.pt.part{n_parts}").write_bytes(blob[i:i + PART])
        n_parts += 1
    manifest = {"trunk_sha256": hashlib.sha256(blob).hexdigest(), "trunk_parts": n_parts, "folds": {}}
    base = {k: v.clone() for k, v in build_model(cfg["model"]).state_dict().items()}
    for f in range(5):
        run = RUNS / "kag13_sam2_pqsel" / "seed0" / f"fold{f}"
        sd = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)["state_dict"]
        assert sd.keys() == base.keys()
        frozen = [k for k in sd if torch.equal(sd[k], base[k]) and k.startswith("encoder.")]
        delta = {k: v for k, v in sd.items() if k not in frozen}
        assert all(k.startswith("encoder.trunk") for k in frozen)
        out = wdir / f"kag13_fold{f}.pt"
        torch.save({"state_dict": delta, "frozen_keys": frozen, "fold": f, "arm": "kag13_sam2_pqsel"}, out)
        m = json.loads((run / "metrics.json").read_text())
        manifest["folds"][str(f)] = {"file": out.name, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                                     "tensors": len(delta), "params": int(sum(v.numel() for v in delta.values())),
                                     "frozen_tensors": len(frozen), "best_val_pq": m.get("best_val_pq"),
                                     "best_epoch": m.get("best_epoch"),
                                     "checkpoint_sha256": hashlib.sha256((run / "checkpoint.pt").read_bytes()).hexdigest()}
        print(f"fold {f}: {len(delta)} tensors, {manifest['folds'][str(f)]['params'] / 1e6:.2f} M params, "
              f"{out.stat().st_size / 1e6:.1f} MB; {len(frozen)} frozen encoder tensors left to the trunk file")
    (wdir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    # round trip through the package loader, in a clean interpreter started inside the package
    base_file = wdir / "sam2" / "sam2.1_hiera_base_plus.pt"
    if base_file.exists():
        base_file.unlink()
    code = (
        "import sys, torch\n"
        "from pipeline import model as pm\n"
        "for f in range(5):\n"
        "    mdl, _ = pm.load_fold_model(f, torch.device('cpu'))\n"
        "    sd = torch.load(f'{sys.argv[1]}/fold{f}/checkpoint.pt', map_location='cpu', weights_only=False)['state_dict']\n"
        "    got = mdl.state_dict()\n"
        "    assert got.keys() == sd.keys() and all(torch.equal(got[k], sd[k]) for k in sd), f'fold {f} differs'\n"
        "print('round trip: all five rebuilt models equal their training checkpoints, tensor for tensor')\n")
    subprocess.run([sys.executable, "-c", code, str(RUNS / "kag13_sam2_pqsel" / "seed0")], cwd=a.out, check=True)


if __name__ == "__main__":
    main()
