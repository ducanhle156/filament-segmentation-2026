"""Backbone-agnostic training loop (Phase 2, Step 13).

There is no arm-specific branching anywhere below: the model comes from the
registry, the loss from the config, the augmentation from one shared pipeline.
Swapping U-Net for ConvNeXt is a config edit, which is the Phase 2 gate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from common.data.augment import Augmentation
from common.data.channels import ChannelSource
from common.data.dataset import EpochSampler, TileDataset, build_dataset
from common.data.unlabeled_pairs import build_unlabeled_loader
from common.engine.mean_teacher import MeanTeacher
from common.data.splits import load_folds
from common.engine.evaluate import (evaluate_images, validation_pq,
                                    validation_score)
from common.engine.registry import build_model, parameter_counts
from common.losses.combo import build_loss
from common.utils.config import config_hash
from common.utils.seed import seed_everything, worker_init_fn


def _param_groups(model, cfg):
    """Encoder and decoder get separate learning rates.

    A pretrained backbone needs a much smaller step than a randomly
    initialised head, or the first few batches destroy the representation the
    arm is supposed to be measuring.
    """
    lr = float(cfg["train"]["lr"])
    # `lr_encoder` is for a *pretrained* encoder, i.e. the `encoder_decoder`
    # model type. A monolithic from-scratch arm that happens to name its
    # backbone `encoder` (ConvNeXt-UPerNet) trains every parameter at `lr`.
    # (Bug found 2026-08-25: the arm03 LR trials had trained their encoder at
    # 1e-5 whatever `train.lr` said; see models/convnext_uper/MODEL_SPEC.md.)
    pretrained = cfg.get("model", {}).get("type") == "encoder_decoder"
    lr_enc = float(cfg["train"].get("lr_encoder", lr)) if pretrained else lr
    # a FRESH_MODULES part of the encoder (random init, Phase 8 magnetic
    # adapter) is a head, not a pretrained weight: it trains at `lr`
    from common.engine.registry import is_fresh
    encoder = getattr(model, "encoder", None)
    enc, rest = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_enc = (pretrained and name.startswith("encoder.")
                  and not (encoder is not None
                           and is_fresh(encoder, name[len("encoder."):])))
        (enc if is_enc else rest).append(p)
    groups = []
    if enc:
        groups.append({"params": enc, "lr": lr_enc, "name": "encoder"})
    if rest:
        groups.append({"params": rest, "lr": lr, "name": "decoder"})
    return groups


def _build_optimizer(model, cfg):
    import torch

    groups = _param_groups(model, cfg)
    if not groups:
        raise ValueError("model has no trainable parameters")
    name = cfg["train"].get("optimizer", "adamw").lower()
    wd = float(cfg["train"].get("weight_decay", 0.01))
    if name == "adamw":
        return torch.optim.AdamW(groups, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, weight_decay=wd)
    raise ValueError(f"unknown optimizer {name!r}")


def _build_scheduler(opt, cfg, epochs):
    """Per-epoch schedule with an optional linear warmup.

    Warmup exists because of the first LR trials (2026-08-25): at 3e-3 the
    U-Net collapsed to all-background in epoch 1 and never recovered, and
    EdgeAttNet's validation clDice swung by 0.1 between consecutive epochs.
    A randomly initialised BN network fed 1%-foreground tiles has huge early
    gradients; ramping the LR over the first `warmup_epochs` epochs is the
    standard remedy and is applied identically to every arm.
    """
    import torch

    name = str(cfg["train"].get("scheduler", "cosine")).lower()
    warmup = int(cfg["train"].get("warmup_epochs", 0) or 0)
    if name in ("none", ""):
        main = None
    elif name == "cosine":
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(epochs - warmup, 1))
    elif name == "step":
        main = torch.optim.lr_scheduler.StepLR(
            opt, step_size=max((epochs - warmup) // 3, 1))
    else:
        raise ValueError(f"unknown scheduler {name!r}")
    if warmup <= 0:
        return main
    ramp = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0 / (warmup + 1), end_factor=1.0,
        total_iters=warmup)
    if main is None:
        return ramp
    return torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[ramp, main], milestones=[warmup])


def _ema_update(prev, value: float, a: float) -> float:
    """score = a * prev + (1 - a) * value; a == 0 (or no prev) is the raw value."""
    if a <= 0 or prev is None:
        return float(value)
    return a * float(prev) + (1.0 - a) * float(value)


def _ema_replay(history: list[dict], a: float):
    """Rebuild the EMA from a resumed history, so no checkpoint field is needed."""
    ema = None
    for row in history:
        ema = _ema_update(ema, row.get("val_score", row["val_cldice"]), a)
    return ema


def _anneal_scheduler(opt, cfg, epochs: int, anneal_n: int):
    """A fresh schedule fast-forwarded to the last `anneal_n` epochs.

    The tail of the cosine is where the LR is low; running it from the best
    state gives the run the anneal phase the early stop denied it. Stepping a
    new scheduler is exact and cheap (per-epoch schedules); it also resets
    the optimizer's param-group LRs to the schedule's value at that point.
    """
    sched = _build_scheduler(opt, cfg, epochs)
    if sched is None:
        return None
    # never start the tail inside the warm-up ramp (only reachable with a toy
    # budget where anneal_epochs + warmup exceeds epochs)
    warmup = int(cfg["train"].get("warmup_epochs", 0) or 0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # "step before optimizer.step"
        for _ in range(max(epochs - anneal_n - 1, warmup, 0)):
            sched.step()
    return sched


def _save_last(path, model, optimizer, scheduler, history, best, best_state,
               cfg, phase, stop_epoch, end_epoch, extra: dict | None = None) -> None:
    import torch

    tmp = path.with_suffix(".tmp")
    ck = {"model": model.state_dict(),
          "optimizer": optimizer.state_dict(),
          "scheduler": scheduler.state_dict() if scheduler else None,
          "history": history, "best": best, "best_state": best_state,
          "phase": phase, "stop_epoch": stop_epoch,
          "end_epoch": end_epoch,
          "config_sha256": config_hash(cfg)}
    if extra is not None:                   # Mean Teacher state (Phase 9); absent otherwise
        ck["ssl"] = extra
    torch.save(ck, tmp)
    tmp.replace(path)                       # never a half-written last.pt


def _mean_teacher_step(mt, model, criterion, img, mask, valid, extra, batch_u,
                       device, amp, t_epochs):
    """One joint step (Phase 9, arm21): supervised loss on the labeled tiles
    plus lambda(t) x consistency loss on the unlabeled tiles. The teacher
    scores the weak view under no_grad; the student sees labeled and strong
    tiles in ONE forward (no BatchNorm anywhere, so this equals two passes).
    The confidence weight reaches the loss through its `valid` argument only;
    the student's input is never masked by it."""
    import torch
    if extra is not None:
        raise NotImplementedError("Mean Teacher with side channels is not supported")
    img_w, img_s, valid_u = (t.to(device, non_blocking=True) for t in batch_u)
    p_t = mt.teacher_probs(img_w, amp)
    target_u, w_u = mt.targets(p_t, valid_u)
    b = img.shape[0]
    x = torch.cat([img, img_s], 0)
    if amp:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
    else:
        logits = model(x)
    logits = logits.float()
    # unsupervised term first: it is GPU-only, so the component loss's CPU
    # labelling below overlaps with queued GPU work
    loss_u, _ = mt.unsup_loss(logits[b:], target_u, w_u)
    loss_s, _ = criterion(logits[:b], mask, valid, None)
    lam = mt.lambda_at(t_epochs)
    loss = loss_s + lam * loss_u
    return loss, {"sup": float(loss_s.detach()), "unsup": float(loss_u.detach()),
                  "lambda": lam, **mt.stats(p_t, w_u, valid_u)}


def warm_start(model, path):
    """Load EVERY model weight from an existing run's checkpoint (strict).

    Used by `train.init_from` (Phase 7-1): fine-tune arms start from a
    converged parent instead of pretrained-encoder + fresh decoder. Accepts
    the run checkpoint layout ({"state_dict": ...}), the DAPT trunk layout
    ({"model": ...}), or a bare state dict.
    """
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        ck = ck.get("state_dict", ck.get("model", ck))
    model.load_state_dict(ck)


def train_fold(cfg: dict, fold: int, seed: int, root: Path,
               smoke: bool = False) -> dict:
    """Train one (arm, seed, fold) and evaluate on its test fold.

    Writes `models/<family>/results/<arm_id>/seed<N>/fold<K>/metrics.json`.
    """
    import torch
    from torch.utils.data import DataLoader

    root = Path(root)
    t_start = time.time()
    seed_everything(seed, cfg["train"].get("deterministic", True))

    folds = load_folds(root / cfg["data"]["splits_file"])
    if fold not in folds:
        raise KeyError(f"fold {fold} not in {sorted(folds)}")
    train_ids = folds[fold]["train"]
    val_ids = folds[fold]["val"]
    test_ids = folds[fold]["test"]

    epochs = int(cfg["train"]["epochs"])
    max_steps = None
    if smoke:
        train_ids = train_ids[:cfg.get("smoke", {}).get("n_images", 20)]
        val_ids = val_ids[:cfg.get("smoke", {}).get("n_images", 20)]
        test_ids = test_ids[:cfg.get("smoke", {}).get("n_images", 20)]
        epochs = int(cfg.get("smoke", {}).get("epochs", 2))
        max_steps = int(cfg.get("smoke", {}).get("iterations", 50))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)
    # Full-model warm start (Phase 7-1, arm12): load every weight from an
    # existing run's checkpoint before training. "{fold}" in the path
    # resolves to this fold, so each fold fine-tunes its own parent.
    # `train.init_from` is a scientific key (hashed), so a warm-started
    # config can never silently resume another config's last.pt.
    # (skipped in smoke runs: --smoke swaps the model for the TinyCNN
    # fixture, whose state dict cannot take the parent's weights)
    init_from = cfg["train"].get("init_from")
    if init_from and not smoke:
        ip = root / str(init_from).format(fold=fold)
        warm_start(model, ip)
        print(f"  warm start: loaded all weights from {ip}", flush=True)
    counts = parameter_counts(model)
    criterion = build_loss(cfg).to(device)
    # Mean Teacher (Phase 9, arm21): default-off. Without `train.ssl` no line
    # guarded by `mt is not None` runs, and every existing arm is unchanged
    # (tests/test_mean_teacher.py pins arm13's config hash and a golden run).
    ssl_cfg = cfg["train"].get("ssl")
    mt = MeanTeacher.from_config(cfg, model) if ssl_cfg else None
    if mt is not None:
        print(f"  mean teacher: ema {mt.alpha} over {len(mt.names)} trainable tensors, "
              f"lambda_u {mt.lambda_u} (ramp {mt.ramp_epochs} ep), "
              f"tau {mt.tau_pos}/{mt.tau_neg}, evaluating the {mt.eval_model}",
              flush=True)
    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg, epochs)
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"

    workers = int(cfg["train"].get("num_workers", 4))
    batch = int(cfg["train"]["batch_size"])
    loader_kw = dict(batch_size=batch, num_workers=workers,
                     pin_memory=device.type == "cuda",
                     worker_init_fn=worker_init_fn,
                     persistent_workers=workers > 0)
    # Side channels (Phase 8): None for every arm without `data.channels`.
    # The eval-side loaders need their own source (one disk cached each).
    channels = ChannelSource.from_config(cfg, root)
    channel_coverage = None
    if channels is not None:
        stems = [i.partition("-")[2] for i in train_ids + val_ids + test_ids]
        channel_coverage = {**channels.describe(), **channels.coverage(stems),
                            "train": channels.coverage(
                                i.partition("-")[2] for i in train_ids),
                            "val": channels.coverage(
                                i.partition("-")[2] for i in val_ids),
                            "test": channels.coverage(
                                i.partition("-")[2] for i in test_ids)}
        print(f"  side channels {channels.names} from {channels.dir}: "
              f"{channel_coverage['n_with_channels']}/{channel_coverage['n_images']} "
              f"disks have a file (missing: {channels.missing}"
              f"{', ZEROED' if channels.zero else ''})", flush=True)
    val_loader = DataLoader(
        TileDataset(root / cfg["data"]["processed_dir"], val_ids,
                    cfg["data"]["tile_size"],
                    channels=ChannelSource.from_config(cfg, root)),
        shuffle=False, **loader_kw)

    # What the stop / checkpoint signal watches (Phase 7-2, arm13):
    #   cldice  tile-level val clDice (every arm through 7-1)
    #   pq      full-disk pooled val PQ at `select_decode` (t, min-area,
    #           min mean prob) -- the benchmark decode, so the checkpoint is
    #           chosen on the number the benchmark reports
    # `best["score"]` is the watched value (EMA-smoothed when val_ema > 0);
    # `best["cldice"]` stays the raw val clDice at that epoch so every
    # existing table keeps its column.
    select = str(cfg["train"].get("select_metric", "cldice")).lower()
    if select not in ("cldice", "pq"):
        raise ValueError(f"train.select_metric must be cldice|pq, got {select!r}")
    select_decode = tuple(cfg["train"].get("select_decode", [0.5, 400, 0.85]))
    best = {"score": -1.0, "cldice": -1.0, "epoch": -1, "metric": select}
    best_state = None
    out_dir = root / cfg["results_dir"] / cfg["arm_id"] / f"seed{seed}" / f"fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path = out_dir / "last.pt"
    start_epoch = 0
    patience = int(cfg["train"].get("early_stop_patience", 15))
    # no early stop before this epoch: the val clDice curve is noisy early on
    # and the 15-epoch patience was firing on a transient dip
    min_epochs = int(cfg["train"].get("min_epochs", 0) or 0)
    # the stop signal: an EMA of val clDice when val_ema > 0, raw otherwise
    ema_a = float(cfg["train"].get("val_ema", 0.0) or 0.0)
    # anneal-on-stop (compute-strategy review, 2026-08-25): a pretrained
    # encoder plateaus by epoch ~5, so patience fires with the cosine barely
    # started and the low-LR tail never runs. When enabled, the stop reloads
    # the best state and runs the last `anneal_epochs` of the schedule.
    anneal_on = bool(cfg["train"].get("anneal_on_stop", False)) and not smoke
    anneal_n = int(cfg["train"].get("anneal_epochs", 10) or 0) if anneal_on else 0
    phase = "train"                     # "train" | "anneal"
    stop_epoch = None                   # epoch at which patience fired
    end_epoch = epochs                  # exclusive; shrinks when annealing
    history = []

    # Resumable: `last.pt` is rewritten every epoch (model, optimizer,
    # scheduler, history, best-so-far) so a walltime kill or node failure
    # costs one epoch, not the run. Same seed + same epoch index -> the same
    # augmentation stream and the same order (EpochSampler), so a resumed run
    # follows the original's data exactly.
    if cfg["train"].get("resume", True) and last_path.exists() and not smoke:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if ck.get("config_sha256") == config_hash(cfg):
            model.load_state_dict(ck["model"])
            if mt is not None:
                if "ssl" not in ck:
                    raise RuntimeError(f"{last_path} has no teacher state for a "
                                       "Mean Teacher run; delete it to restart")
                mt.load_state_dict(ck["ssl"])
            optimizer.load_state_dict(ck["optimizer"])
            if scheduler is not None and ck.get("scheduler") is not None:
                scheduler.load_state_dict(ck["scheduler"])
            history = ck["history"]
            best = ck["best"]
            best.setdefault("score", best["cldice"])
            best.setdefault("metric", select)
            best_state = ck.get("best_state")
            phase = ck.get("phase", "train")
            stop_epoch = ck.get("stop_epoch")
            end_epoch = int(ck.get("end_epoch", epochs))
            start_epoch = len(history)
            print(f"  resumed from {last_path} at epoch {start_epoch} "
                  f"(best {best['score']:.4f} @ {best['epoch']}, "
                  f"phase {phase})", flush=True)
        else:
            print("  last.pt belongs to a different config -- ignored", flush=True)

    # One dataset and one loader for the whole run. The epoch travels inside
    # the sampler's indices, so persistent workers see it without being
    # re-spawned (which cost a worker start-up per epoch before).
    train_ds = build_dataset(cfg, root, train_ids, train=True, seed=seed,
                             fold=fold)
    if hasattr(train_ds, "n_extra"):
        print(f"  extra_train: {train_ds.n_lab} labeled + {train_ds.n_extra} "
              f"pseudo tiles per epoch (pool {len(train_ds.extra)})", flush=True)
    sampler = EpochSampler(len(train_ds), seed)
    train_loader = DataLoader(train_ds, sampler=sampler, drop_last=False,
                              **loader_kw)
    steps_per_epoch = len(train_loader) if not max_steps else min(len(train_loader), max_steps)
    unl_loader = None
    if mt is not None:
        unl_loader = build_unlabeled_loader(cfg, root, seed, steps_per_epoch, loader_kw)
        if unl_loader is not None:
            print(f"  unlabeled: {len(unl_loader.dataset)} two-view tiles in the pool "
                  f"({ssl_cfg.get('pair_mode', 'temporal')}), "
                  f"{len(unl_loader.sampler)} per epoch, batch {unl_loader.batch_size}",
                  flush=True)
        else:
            print("  unlabeled: none (batch_unlabeled 0: EMA teacher only)", flush=True)
    collapse_max_fg = float(ssl_cfg.get("collapse_max_fg", 0.2)) if ssl_cfg else None
    ema = _ema_replay(history, ema_a)

    epoch = start_epoch
    while epoch < end_epoch:
        sampler.set_epoch(epoch)
        train_ds.set_epoch(epoch)       # for num_workers == 0
        model.train()
        run_loss, n_steps = 0.0, 0
        unl_it, mt_sums = None, {}
        if mt is not None:
            mt.train_mode()
            if unl_loader is not None:
                unl_loader.sampler.set_epoch(epoch)
                unl_loader.dataset.set_epoch(epoch)
                unl_it = iter(unl_loader)
        for img, mask, valid in train_loader:
            img, mask, valid = (t.to(device, non_blocking=True)
                                for t in (img, mask, valid))
            optimizer.zero_grad(set_to_none=True)
            # side channels (if any) also go to the loss: a prior term such
            # as pil_bce reads them; every other term ignores them
            extra = img[:, 1:].float() if img.shape[1] > 1 else None
            if unl_it is not None:
                loss, st = _mean_teacher_step(
                    mt, model, criterion, img, mask, valid, extra, next(unl_it),
                    device, amp, epoch + n_steps / max(steps_per_epoch, 1))
                for k, v in st.items():
                    mt_sums[k] = mt_sums.get(k, 0.0) + v
            elif amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(img)
                loss, _ = criterion(logits.float(), mask, valid, extra)
            else:
                loss, _ = criterion(model(img).float(), mask, valid, extra)
            loss.backward()
            clip = cfg["train"].get("grad_clip")
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
            optimizer.step()
            if mt is not None:
                mt.update(model)
            run_loss += float(loss.detach())
            n_steps += 1
            if max_steps and n_steps >= max_steps:
                break

        lr_used = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        # the evaluated model: the EMA teacher for a Mean Teacher arm with
        # `eval_model: teacher`, the student (== model) everywhere else
        eval_model = mt.model_for_eval(model) if mt is not None else model
        val = validation_score(eval_model, val_loader, device, cfg)
        row = {"epoch": epoch, "train_loss": run_loss / max(n_steps, 1),
               "val_cldice": val["cldice"], "val_loss": val["loss"],
               "lr": lr_used, "phase": phase}
        if select == "pq":
            vpq = validation_pq(eval_model, val_ids, root / cfg["data"]["interim_dir"],
                                cfg, device, select_decode,
                                limit=len(val_ids) if not smoke else 2,
                                channels=channels)
            row.update({"val_pq": vpq["pq"], "val_tp": vpq["tp"],
                        "val_fp": vpq["fp"], "val_fn": vpq["fn"]})
            score = vpq["pq"]
        else:
            score = val["cldice"]
        if mt is not None:
            n = max(n_steps, 1)
            row.update({"lambda_u": mt.lambda_at(epoch), "ema_steps": mt.step,
                        "val_fg_frac": val["fg_frac"],
                        **{f"mt_{k}": v / n for k, v in mt_sums.items()}})
            if ssl_cfg.get("eval_student_too") and select == "pq" and mt.eval_model == "teacher":
                spq = validation_pq(model, val_ids, root / cfg["data"]["interim_dir"],
                                    cfg, device, select_decode,
                                    limit=len(val_ids) if not smoke else 2,
                                    channels=channels)
                row["val_pq_student"] = spq["pq"]
            if val["fg_frac"] > collapse_max_fg:
                raise RuntimeError(f"student collapse: val foreground fraction "
                                   f"{val['fg_frac']:.3f} > {collapse_max_fg} at epoch {epoch}")
        ema = _ema_update(ema, score, ema_a)
        row["val_score"] = score
        row["val_score_ema"] = ema
        row["val_cldice_ema"] = ema if select == "cldice" else val["cldice"]
        history.append(row)
        print(f"  epoch {epoch:3d}  train {row['train_loss']:.4f}  "
              f"val clDice {val['cldice']:.4f}"
              + (f"  val PQ {row['val_pq']:.4f} "
                 f"({row['val_tp']}/{row['val_fp']}/{row['val_fn']})"
                 if select == "pq" else "")
              + (f" (ema {ema:.4f})" if ema_a > 0 else "")
              + f"  val loss {val['loss']:.4f}"
              + (f"  [mt lambda {row['lambda_u']:.2f} unsup {row.get('mt_unsup', 0.0):.4f} "
                 f"conf+ {row.get('mt_conf_frac_pos', 0.0):.4f} fg {row['val_fg_frac']:.4f}]"
                 if mt is not None else "")
              + ("  [anneal]" if phase == "anneal" else ""), flush=True)

        # early stopping watches the selected score (smoothed if val_ema > 0),
        # never the loss
        if ema > best["score"]:
            best = {"score": ema, "cldice": val["cldice"], "epoch": epoch,
                    "raw_score": score, "raw_cldice": val["cldice"],
                    "metric": select}
            best_state = {k: v.detach().cpu().clone()
                          for k, v in eval_model.state_dict().items()}
        elif (phase == "train" and epoch - best["epoch"] >= patience
              and epoch + 1 >= min_epochs):
            stop_epoch = epoch
            if anneal_n > 0 and epoch + 1 < epochs and best_state is not None:
                # do not exit: finish the schedule's tail from the best state
                phase = "anneal"
                end_epoch = min(epochs, epoch + 1 + anneal_n)
                model.load_state_dict(best_state)
                if mt is not None:
                    mt.reset_from(model)    # teacher := the reloaded best state
                scheduler = _anneal_scheduler(optimizer, cfg, epochs, anneal_n)
                print(f"  early stop at epoch {epoch} "
                      f"(best {best['score']:.4f} @ {best['epoch']}) -> "
                      f"annealing {end_epoch - epoch - 1} epochs from the best "
                      f"state at lr {optimizer.param_groups[0]['lr']:.2e}",
                      flush=True)
            else:
                print(f"  early stop at epoch {epoch} "
                      f"(best {best['score']:.4f} @ {best['epoch']})", flush=True)
                if not smoke:
                    _save_last(last_path, model, optimizer, scheduler, history,
                               best, best_state, cfg, phase, stop_epoch, end_epoch,
                               extra=mt.state_dict() if mt is not None else None)
                break

        if not smoke:
            _save_last(last_path, model, optimizer, scheduler, history, best,
                       best_state, cfg, phase, stop_epoch, end_epoch,
                       extra=mt.state_dict() if mt is not None else None)
        epoch += 1

    if best_state is not None:
        model.load_state_dict(best_state)

    if cfg["train"].get("save_checkpoint", True) and best_state is not None:
        ck = {"state_dict": best_state, "cfg": cfg, "fold": fold,
              "seed": seed, "best": best}
        if mt is not None:
            ck["ssl"] = mt.describe()
        torch.save(ck, out_dir / "checkpoint.pt")

    per_image = evaluate_images(
        model, test_ids, root / cfg["data"]["interim_dir"], cfg, device,
        threshold=float(cfg["eval"].get("threshold", 0.5)),
        save_predictions=out_dir / "predictions"
        if cfg["eval"].get("save_predictions", False) else None,
        channels=channels)

    peak_mem = (torch.cuda.max_memory_allocated() / 2 ** 30
                if device.type == "cuda" else 0.0)
    metrics = {
        "arm_id": cfg["arm_id"], "fold": fold, "seed": seed,
        "config_sha256": config_hash(cfg),
        "arm_config_file": cfg.get("_arm_config_file"),
        "cache_version": cfg["data"]["processed_dir"],
        "splits_file": cfg["data"]["splits_file"],
        "git_commit": cfg.get("_git_commit"),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "smoke": smoke,
        "n_train_images": len(train_ids), "n_val_images": len(val_ids),
        "n_test_images": len(test_ids),
        "best_epoch": best["epoch"],
        # unchanged meaning for clDice-selected arms (the smoothed value the
        # stop watched); for pq-selected arms the raw val clDice at the
        # chosen epoch (the smoothed PQ is best_val_score)
        "best_val_cldice": best["score"] if select == "cldice" else best["cldice"],
        "best_val_cldice_raw": best.get("raw_cldice", best["cldice"]),
        "select_metric": select,
        "select_decode": list(select_decode) if select == "pq" else None,
        "best_val_score": best["score"],
        "best_val_pq": best.get("raw_score") if select == "pq" else None,
        "val_ema": ema_a,
        "epochs_run": len(history),
        "epochs_budget": epochs,
        "stop_epoch": stop_epoch,
        "anneal_epochs": sum(1 for r in history if r.get("phase") == "anneal"),
        "wall_clock_s": time.time() - t_start,
        "peak_gpu_memory_gib": peak_mem,
        **counts,
        "loss_terms": criterion.describe(),
        "augmentation": (Augmentation(cfg["augment"]).describe()
                         if cfg["data"].get("augment", True) else None),
        "side_channels": channel_coverage,
        "history": history,
        "per_image": per_image,
    }
    if mt is not None:
        from common.data.augment import StrongAugmentation
        metrics["ssl"] = {**mt.describe(),
                          "strong_augmentation": StrongAugmentation(ssl_cfg.get("strong")).describe(),
                          "pool_manifest": ssl_cfg.get("pool_manifest"),
                          "n_unlabeled_tiles": len(unl_loader.dataset) if unl_loader else 0,
                          "unlabeled_per_epoch": len(unl_loader.sampler) if unl_loader else 0}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    print(f"  wrote {out_dir / 'metrics.json'}", flush=True)
    if last_path.exists():
        last_path.unlink()                  # the run is complete; best is in checkpoint.pt
    return metrics
