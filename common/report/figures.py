"""Figures for reports and gates (`common/report/`).

Two families live here:

* **Qualitative** -- what the model actually drew on a disk, against the
  annotation. `stitched_prediction_figure` covers one run; `prediction_grid_figure`
  puts several arms on the *same* images side by side, which is the only honest
  way to look at two models at once.
* **Quantitative** -- training curves, per-image distributions, arm-vs-arm
  comparisons. Every one of these reads a `metrics.json` and nothing else, so
  they can be regenerated at any time without touching a GPU.

House rule for all of them: per-image spread is shown, never hidden behind a
mean. The Phase 3 gate turns seed spread into a significance threshold, so a
bar chart that omits it would be reporting a difference the project has already
decided not to believe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.report.style import (FALSE_POS, GRID, GT_FILL, HIT, INK_2,
                                 INK_MUTED, MISSED, OUTCOME_LABELS, PROB_CMAP,
                                 annotate_provenance, series_color,
                                 use_house_style)


def _prep(out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    use_house_style()
    return out_path


def _disk(ax, img: np.ndarray, valid: np.ndarray):
    """The normalized disk as a grayscale backdrop, clipped to the analysis
    region. Percentile limits, not min/max: one hot pixel would flatten the
    granulation and the filaments with it."""
    show = np.where(valid, img, np.nan)
    lo, hi = np.nanpercentile(show, [1, 99])
    ax.imshow(show, cmap="gray", vmin=lo, vmax=hi)
    ax.axis("off")
    return show


def _outcome_overlay(target: np.ndarray, pred: np.ndarray, valid: np.ndarray):
    gt = target & valid
    ov = np.zeros((*target.shape, 4))
    ov[gt & ~pred] = MISSED
    ov[pred & ~gt] = FALSE_POS
    ov[pred & gt] = HIT
    return ov


def _header(fig_h: float, inches: float = 0.6) -> float:
    """Top of the plotting area, reserving a fixed *physical* strip for the
    title and legend. A fractional reserve leaves a growing band of white space
    as a grid gets taller -- the strip's contents are the same size either way.
    """
    return 1.0 - inches / fig_h


def _outcome_legend(fig, y: float = 1.0):
    """Figure-level, never inside an axes: the panels are full-bleed images, so
    an in-axes legend lands on top of the disk it is explaining."""
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=c, label=n) for n, c in OUTCOME_LABELS],
               loc="upper right", bbox_to_anchor=(0.995, y), ncols=3,
               fontsize=8, labelcolor=INK_2)


# ------------------------------------------------------------- qualitative

def stitched_prediction_figure(image_id: str, img: np.ndarray, target: np.ndarray,
                               prob: np.ndarray, valid: np.ndarray,
                               out_path: Path, threshold: float = 0.5,
                               metrics: dict | None = None,
                               provenance: str = "") -> Path:
    """One run, one disk: input, ground truth, stitched probability, outcome.

    The probability panel is the one that shows tiling bugs: a seam or a gap in
    the tile grid appears as a straight edge at a multiple of the stride.
    """
    import matplotlib.pyplot as plt

    out_path = _prep(out_path)
    pred = prob >= threshold
    gt = target & valid

    fig_h = 5.2
    fig, ax = plt.subplots(1, 4, figsize=(18, fig_h))
    _disk(ax[0], img, valid)
    ax[0].set_title("input (normalized, analysis region)")

    _disk(ax[1], img, valid)
    layer = np.zeros((*target.shape, 4))
    layer[gt] = GT_FILL
    ax[1].imshow(layer)
    ax[1].set_title(f"ground truth — {int(gt.sum()):,} px")

    im = ax[2].imshow(np.where(valid, prob, np.nan), cmap=PROB_CMAP, vmin=0, vmax=1)
    ax[2].axis("off")
    ax[2].set_title("stitched probability")
    cb = fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, color=GRID)

    _disk(ax[3], img, valid)
    ax[3].imshow(_outcome_overlay(target, pred, valid))
    ax[3].set_title(f"prediction vs truth @ {threshold:g}")

    title = image_id
    if metrics:
        items = [f"{k} {v:.3f}" for k, v in metrics.items()]
        # at most six per line: the legend sits at the top right and a long
        # single line runs underneath it
        chunks = [items[i:i + 6] for i in range(0, len(items), 6)]
        title += "\n" + "\n".join("   ".join(c) for c in chunks)
    fig.suptitle(title, fontsize=9, x=0.01, ha="left", va="top", y=0.995)
    _outcome_legend(fig, y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, _header(fig_h)))
    if provenance:
        annotate_provenance(fig, provenance)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def prediction_grid_figure(rows: list[dict], model_labels: list[str],
                           out_path: Path, threshold: float = 0.5,
                           title: str = "", metric_key: str = "cldice",
                           provenance: str = "") -> Path:
    """Several arms on the same disks: images down, models across.

    `rows` is one dict per image with keys `image_id`, `img`, `valid`,
    `target`, and `probs` mapping each model label to its full-disk
    probability map; optional `metrics` maps a label to that image's metric
    row. Columns are input, ground truth, then one outcome overlay per model.

    Holding the *images* fixed is the point. Comparing an arm's best disk with
    another arm's best disk compares two different problems; this compares one.
    """
    import matplotlib.pyplot as plt

    out_path = _prep(out_path)
    n_r, n_c = len(rows), 2 + len(model_labels)
    fig_h = 3.35 * n_r
    fig, ax = plt.subplots(n_r, n_c, figsize=(3.1 * n_c, fig_h), squeeze=False)

    for r, row in enumerate(rows):
        valid, target = row["valid"], row["target"]
        gt = target & valid
        _disk(ax[r][0], row["img"], valid)
        _disk(ax[r][1], row["img"], valid)
        layer = np.zeros((*target.shape, 4))
        layer[gt] = GT_FILL
        ax[r][1].imshow(layer)

        for c, label in enumerate(model_labels, start=2):
            a = ax[r][c]
            prob = row["probs"].get(label)
            if prob is None:
                a.axis("off")
                a.text(0.5, 0.5, "no run", ha="center", va="center",
                       color=INK_MUTED, fontsize=9, transform=a.transAxes)
                continue
            _disk(a, row["img"], valid)
            a.imshow(_outcome_overlay(target, prob >= threshold, valid))
            v = (row.get("metrics", {}).get(label) or {}).get(metric_key)
            if v is not None:
                a.text(0.02, 0.02, f"{metric_key} {v:.3f}", transform=a.transAxes,
                       fontsize=8, color=INK_2, ha="left", va="bottom")

        # row identity on the left edge, so the disks stay uncluttered
        ax[r][0].text(-0.04, 0.5, row["image_id"], transform=ax[r][0].transAxes,
                      rotation=90, va="center", ha="right", fontsize=7.5,
                      color=INK_2)
        ax[r][1].text(0.02, 0.02, f"{int(gt.sum()):,} px", fontsize=8,
                      color=INK_2, transform=ax[r][1].transAxes,
                      ha="left", va="bottom")

    for c, name in enumerate(["input", "ground truth", *model_labels]):
        ax[0][c].set_title(name, fontsize=10)

    if title:
        fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    _outcome_legend(fig, y=1.0)
    fig.tight_layout(rect=(0, 0.015, 1, _header(fig_h)))
    if provenance:
        annotate_provenance(fig, provenance)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------------------------------------ quantitative

def training_curves_figure(runs: list[dict], out_path: Path, title: str = "",
                           provenance: str = "") -> Path:
    """Loss and validation clDice against epoch, one line per run.

    Two stacked axes rather than one axis with two scales: loss and clDice do
    not share units, and a twin y-axis lets the reader infer a crossing that
    is an artefact of where the two scales were pinned.

    `runs` is a list of `{"label": str, "history": [...]}` -- `history` rows as
    written by `train_fold`.
    """
    import matplotlib.pyplot as plt

    out_path = _prep(out_path)
    fig, ax = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True,
                           height_ratios=(1, 1))

    for i, run in enumerate(runs):
        h = run["history"]
        if not h:
            continue
        c = series_color(i)
        ep = [r["epoch"] for r in h]
        ax[0].plot(ep, [r["train_loss"] for r in h], color=c, label=run["label"])
        if any("val_loss" in r for r in h):
            ax[0].plot(ep, [r.get("val_loss", np.nan) for r in h], color=c,
                       linestyle="--", linewidth=1.4, alpha=0.85)
        cld = [r.get("val_cldice", np.nan) for r in h]
        ax[1].plot(ep, cld, color=c, label=run["label"])
        if np.all(np.isnan(cld)):  # e.g. YOLO trials: no per-epoch clDice
            continue
        best = int(np.nanargmax(cld))
        ax[1].plot([ep[best]], [cld[best]], marker="o", color=c,
                   markersize=8, markeredgecolor="#fcfcfb", markeredgewidth=2,
                   zorder=5)
        ax[1].annotate(f"{cld[best]:.3f} @ ep {ep[best]}",
                       (ep[best], cld[best]), textcoords="offset points",
                       xytext=(8, -2), fontsize=8, color=INK_2)

    ax[0].set_title("loss — solid train, dashed validation")
    ax[0].set_ylabel("loss")
    ax[1].set_title("validation clDice — the selection metric")
    ax[1].set_ylabel("clDice")
    ax[1].set_xlabel("epoch")
    if len(runs) > 1:
        ax[0].legend(loc="upper right", labelcolor=INK_2)

    if title:
        fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97 if title else 1))
    if provenance:
        annotate_provenance(fig, provenance)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def metric_distribution_figure(groups: list[dict], metrics: list[str],
                               out_path: Path, title: str = "",
                               provenance: str = "") -> Path:
    """Per-image distributions, one facet per metric, one box per group.

    `groups` is `[{"label": str, "rows": [per_image dicts]}]`. Every image is
    drawn as a point behind the box: the image-to-image spread on this dataset
    is far larger than any arm-to-arm gap, and a figure that showed only means
    would make gaps look decisive that are not.
    """
    import matplotlib.pyplot as plt

    out_path = _prep(out_path)
    rng = np.random.default_rng(0)          # jitter must not move between runs
    n = len(metrics)
    fig, ax = plt.subplots(1, n, figsize=(2.9 * n + 1.2, 4.4), squeeze=False)
    ax = ax[0]

    for j, metric in enumerate(metrics):
        a = ax[j]
        data, labels, colors = [], [], []
        for i, g in enumerate(groups):
            vals = [r[metric] for r in g["rows"] if metric in r]
            if not vals:
                continue
            data.append(np.asarray(vals, dtype=float))
            labels.append(g["label"])
            colors.append(series_color(i))
        if not data:
            a.axis("off")
            continue
        for k, (vals, c) in enumerate(zip(data, colors)):
            x = k + 1 + rng.uniform(-0.16, 0.16, size=len(vals))
            a.scatter(x, vals, s=7, color=c, alpha=0.28, linewidths=0, zorder=1)
        bp = a.boxplot(data, widths=0.42, showfliers=False, whis=(5, 95),
                       tick_labels=labels, zorder=3)
        for k, c in enumerate(colors):
            bp["boxes"][k].set(color=c, linewidth=1.8)
            bp["medians"][k].set(color=c, linewidth=2.6)
            for part in ("whiskers", "caps"):
                for e in bp[part][2 * k:2 * k + 2]:
                    e.set(color=c, linewidth=1.2)
            a.text(k + 1, 1.005, f"{np.mean(data[k]):.3f}",
                   transform=a.get_xaxis_transform(), ha="center",
                   fontsize=8, color=INK_2)
        # pad clears the mean row, which sits just above the axes
        a.set_title(metric, pad=18)
        a.set_ylim(bottom=0)
        if len(labels) == 1:
            # one group: the title already names the run, so repeating it under
            # every facet is five copies of the same label
            a.set_xticks([])
        else:
            a.tick_params(axis="x", labelrotation=20 if len(labels) > 2 else 0)

    ax[0].set_ylabel("per-image value")
    sub = ("box: median, IQR, 5-95% whiskers · one dot per test image · "
           "number above each box: mean")
    if title:
        fig.suptitle(title, fontsize=11, x=0.01, ha="left")
        fig.text(0.01, 0.925, sub, fontsize=8, color=INK_MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.91 if title else 1))
    if provenance:
        annotate_provenance(fig, provenance)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def arm_comparison_figure(summaries: dict, metrics: list[str], out_path: Path,
                          gate: dict | None = None, title: str = "",
                          provenance: str = "") -> Path:
    """Mean +/- std-across-seeds per arm, with the significance gate drawn in.

    `summaries` maps an arm label to `aggregate()` output. `gate` maps a metric
    to the seed spread that a gap must exceed to count; it is drawn as a bar of
    that height on the leading arm so the reader can see, without arithmetic,
    whether a visible gap is larger than the noise the project already measured.
    """
    import matplotlib.pyplot as plt

    out_path = _prep(out_path)
    labels = list(summaries)
    n_m, n_a = len(metrics), len(labels)
    fig, ax = plt.subplots(figsize=(2.0 * n_m + 2.4, 4.6))
    width = 0.8 / max(n_a, 1)

    for i, label in enumerate(labels):
        s = summaries[label]["all"]
        xs, ys, es = [], [], []
        for j, m in enumerate(metrics):
            a = s.get(m)
            if not a:
                continue
            xs.append(j - 0.4 + width * (i + 0.5))
            ys.append(a["mean"])
            es.append(a.get("std_across_seeds") or 0.0)
        c = series_color(i)
        ax.bar(xs, ys, width=width * 0.92, color=c, label=label, zorder=2)
        ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=INK_2, elinewidth=1.2,
                    capsize=3, zorder=3)
        for x, y, e in zip(xs, ys, es):
            ax.text(x, y + e + 0.012, f"{y:.3f}", ha="center", fontsize=7.5,
                    color=INK_2)

    if gate:
        for j, m in enumerate(metrics):
            g = gate.get(m)
            if not g:
                continue
            ax.annotate("", xy=(j + 0.44, g), xytext=(j + 0.44, 0),
                        arrowprops=dict(arrowstyle="<->", color=INK_MUTED,
                                        linewidth=1.0))
            ax.text(j + 0.47, g / 2, f"gate\n{g:.3f}", fontsize=6.5,
                    color=INK_MUTED, va="center")

    ax.set_xticks(range(n_m), metrics)
    ax.set_ylabel("mean over images  (bars: ± std across seeds)")
    ax.set_ylim(bottom=0)
    if n_a > 1:
        ax.legend(loc="upper left", ncols=min(n_a, 3), labelcolor=INK_2)
    if title:
        ax.set_title(title)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    if provenance:
        annotate_provenance(fig, provenance)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def paired_scatter_figure(a_rows: list[dict], b_rows: list[dict],
                          labels: tuple[str, str], metric: str, out_path: Path,
                          title: str = "", provenance: str = "") -> Path:
    """The same image scored by two arms, one point per image, against y = x.

    A mean tells you which arm is ahead; this tells you whether it is ahead
    everywhere or ahead on a handful of disks and behind on the rest -- the
    difference between a better model and a different failure mode.
    """
    import matplotlib.pyplot as plt

    out_path = _prep(out_path)
    a_by = {r["image_id"]: r for r in a_rows}
    b_by = {r["image_id"]: r for r in b_rows}
    shared = sorted(set(a_by) & set(b_by))
    xs = np.array([a_by[i][metric] for i in shared if metric in a_by[i]])
    ys = np.array([b_by[i][metric] for i in shared if metric in b_by[i]])

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    hi = float(max(xs.max(initial=0), ys.max(initial=0))) * 1.06 or 1.0
    ax.plot([0, hi], [0, hi], color=INK_MUTED, linewidth=1.0, linestyle="--",
            zorder=1)
    ax.scatter(xs, ys, s=16, color=series_color(0), alpha=0.55, linewidths=0,
               zorder=2)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal")
    ax.grid(True, axis="both")
    ax.set_xlabel(f"{labels[0]} — {metric}")
    ax.set_ylabel(f"{labels[1]} — {metric}")
    if len(xs):
        won = int((ys > xs).sum())
        ax.set_title(title or f"{labels[1]} vs {labels[0]} — {metric}")
        ax.text(0.03, 0.97, f"{labels[1]} higher on {won}/{len(xs)} images\n"
                            f"mean Δ {np.mean(ys - xs):+.4f}",
                transform=ax.transAxes, va="top", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    if provenance:
        annotate_provenance(fig, provenance)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
