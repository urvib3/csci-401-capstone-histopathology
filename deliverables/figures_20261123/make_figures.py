# Figures for deliverables/deliverable_7.md (November 23). Numbers are copied from the
# run summaries cited in the report; rerun after updating them.
#
#   python deliverables/figures_20261123/make_figures.py

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SET_RULE, LOCAL, NEG = "#2a78d6", "#a9a8a2", "#eb6834"    # palette slot 1, neutral, slot 2

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
})


# One row per method family, same rows for both datasets (None = not run).
# PanNuke: organ_wcsp/breast/evaluation.json (branch organ-wcsp-breast), Breast test, 169 patches.
# BCSS: runs/*/dev/summary.csv, dev, 185 tiles (lambda = 8): the only split with every family.
FAMILIES = [  # (label, PanNuke Breast test, BCSS dev, BCSS val, is presence); None = not run yet
    ("Base model", 0.432348, 0.706720, 0.648852, None),
    ("Pixel WCSP, handwritten table", None, 0.709734, 0.650712, False),
    ("Pixel WCSP, data-learned table", 0.432412, 0.708998, 0.650602, False),
    ("Hierarchical (multi-scale blocks)", 0.432328, 0.710750, 0.651698, False),
    ("Hierarchical + decay", 0.432305, 0.710783, None, False),
    ("Cluster (nucleus / SLIC superpixel)", 0.434374, 0.716253, None, False),
    ("Presence", None, 0.722877, 0.654962, True),
    ("Presence + cluster", 0.448441, 0.724218, 0.656387, True),
]


def family_panel(ax, title, col, base):
    """Horizontal bars on the raw mIoU scale; the axis starts just below the lowest value."""
    labels = [f[0] for f in FAMILIES]
    ys = list(range(len(FAMILIES)))[::-1]
    vals = [f[col] for f in FAMILIES if f[col] is not None]
    lo = round(min(vals) - 0.006, 2)
    for y, f in zip(ys, FAMILIES):
        v, rule = f[col], f[4]
        if v is None:
            ax.text(lo + 0.0008, y, "—", va="center", color=MUTED, fontsize=9)
            continue
        color = MUTED if rule is None else (SET_RULE if rule else LOCAL)
        ax.barh(y, v - lo, left=lo, color=color, height=0.62, edgecolor=SURFACE, linewidth=2)
        d = v - base
        text = f"{v:.4f}" if rule is None else f"{v:.4f}  ({'0.0000' if abs(d) < 5e-5 else f'{d:+.4f}'})"
        ax.text(v + 0.0007, y, text, va="center", color=INK, fontsize=9)
    ax.axvline(base, color=INK, linewidth=1, linestyle="--", zorder=3)
    ax.set_yticks(ys, labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(lo, max(vals) + 0.012)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"mIoU (axis from {lo:.2f})")
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")


def fig_gains():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True, gridspec_kw={"wspace": 0.1})
    family_panel(axes[0], "PanNuke · Breast test", 1, 0.432348)
    family_panel(axes[1], "BCSS · dev", 2, 0.706720)
    family_panel(axes[2], "BCSS · val", 3, 0.648852)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (MUTED, LOCAL, SET_RULE)]
    fig.legend(handles, ["Base model (HoVer-Net / U-Net)", "Smoothing / local structure", "Presence"],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.55, -0.12))
    fig.savefig(OUT / "gains_both_datasets.png", dpi=200, bbox_inches="tight")


def fig_val():
    rows = [("Pixel WCSP", 0.18, (0.11, 0.24), False),
            ("WCSP + class presence", 0.61, (-0.11, 1.20), True),
            ("Cluster WCSP + class presence", 0.75, (-0.07, 1.39), True)]
    fig, ax = plt.subplots(figsize=(6.2, 2.3))
    for i, (label, v, (lo, hi), rule) in enumerate(rows[::-1]):
        color = SET_RULE if rule else LOCAL
        ax.plot([lo, hi], [i, i], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(v, i, "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(hi + 0.06, i, f"{v:+.2f}  [{lo:+.2f}, {hi:+.2f}]", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.4, 2.3)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("mIoU points over the U-Net (95% ROI bootstrap)")
    ax.set_title("BCSS val: 2,124 tiles, 45 held-out ROIs", loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.savefig(OUT / "bcss_val.png", dpi=200, bbox_inches="tight")


def fig_classes():
    classes = ["Tumor", "Inflammatory", "Stroma", "Necrosis", "Other"]
    delta = [0.4, -0.6, 0.1, 2.5, 1.4]       # val: U-Net -> cluster WCSP + presence, IoU points
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    colors = [SET_RULE if d >= 0 else NEG for d in delta]
    bars = ax.bar(classes, delta, color=colors, width=0.6, edgecolor=SURFACE, linewidth=2)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", length=0, colors=INK)
    for bar, d in zip(bars, delta):
        ax.text(bar.get_x() + bar.get_width() / 2, d + (0.12 if d >= 0 else -0.12), f"{d:+.1f}",
                ha="center", va="bottom" if d >= 0 else "top", color=INK, fontsize=9)
    ax.set_ylim(-1.3, 3.1)
    ax.set_ylabel("IoU points")
    ax.set_title("BCSS val: where the gain comes from", loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.savefig(OUT / "bcss_val_classes.png", dpi=200, bbox_inches="tight")


def fig_rules():
    rules = ["Rare class set in tile", "'Other' in tile", "Necrosis in tile",
             "'Other' present but < 2%", "Inflammatory not in stroma", "Necrosis not touching tumor"]
    truth = [2, 11, 11, 21, 4, 60]
    unet = [10, 20, 15, 43, 19, 36]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ys = range(len(rules))[::-1]
    for y, t, u in zip(ys, truth, unet):
        ax.plot([t, u], [y, y], color=GRID, linewidth=2, zorder=1)
    ax.scatter(truth, list(ys), s=60, color=LOCAL, edgecolor=SURFACE, linewidth=2, zorder=2, label="Ground truth")
    ax.scatter(unet, list(ys), s=60, color=NEG, edgecolor=SURFACE, linewidth=2, zorder=3, label="U-Net")
    ax.set_yticks(list(ys), rules)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("% of tiles or regions (dev)")
    ax.set_xlim(0, 65)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("BCSS rules: how often each is broken", loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.savefig(OUT / "bcss_rule_check.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    fig_gains()
    fig_val()
    fig_classes()
    fig_rules()
    print("written to", OUT)
