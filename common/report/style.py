"""House style for every figure in the project (`common/report/`).

One palette, one set of rcParams, so a figure from Phase 3 and a figure from
Phase 6 can sit on the same page without looking like they came from different
papers.

The categorical hues are slots 1-3 of the validated reference theme. Only the
first three clear the all-pairs colour-vision-deficiency floors, which is the
gate that matters here: arms appear together in scatter plots, where every pair
is adjacent. A fourth arm does not get a fourth hue -- it gets its own facet.
"""

from __future__ import annotations

# ---------------------------------------------------------------- surfaces
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3df"

# ------------------------------------------------------- categorical slots
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#8e5bd6")   # blue, orange, aqua, violet
# Slot 4 was added for the YOLO-seg arm (2026-08-25). It separates from the
# first three by hue and by lightness; keep facets for anything beyond four.
MAX_SERIES = len(SERIES)

# --------------------------------------------------------- segmentation
# Three outcome classes drawn over a grayscale disk. Deliberately NOT the
# usual red/green: those are the one pair a deuteranope cannot separate, and
# missed-vs-hit is exactly the distinction the panel exists to show. Blue /
# yellow / red separate by hue for normal vision and by lightness under CVD,
# and every panel also carries a labelled legend so identity is never colour
# alone.
GT_FILL = (0.16, 0.47, 0.84, 0.55)      # ground truth, blue
HIT = (0.93, 0.63, 0.00, 0.85)          # true positive, yellow
MISSED = (0.16, 0.47, 0.84, 0.65)       # false negative, blue
FALSE_POS = (0.89, 0.29, 0.28, 0.65)    # false positive, red
OUTCOME_LABELS = (("hit", HIT), ("missed", MISSED), ("false positive", FALSE_POS))

PROB_CMAP = "magma"                     # sequential, single-hue-ish, dark->bright


def series_color(i: int) -> str:
    """Colour for series `i`. Raises past slot 3 rather than cycling: a cycled
    hue silently repaints an entity, which is the one thing colour must not do.
    """
    if i >= MAX_SERIES:
        raise IndexError(
            f"series {i}: only {MAX_SERIES} validated categorical slots. "
            "Facet into small multiples instead of reusing a hue.")
    return SERIES[i]


def use_house_style() -> None:
    """Apply the shared rcParams. Call once before building a figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "figure.dpi": 130,
    })


def annotate_provenance(fig, text: str) -> None:
    """Stamp the run identity along the bottom. A figure that leaves the repo
    without its arm/fold/seed/commit cannot be traced back to a run.
    """
    fig.text(0.005, 0.004, text, fontsize=6.5, color=INK_MUTED,
             ha="left", va="bottom")
