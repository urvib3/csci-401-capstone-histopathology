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


def miou_panel(ax, title, base_label, base, rows):
    """Dot plot on the raw mIoU scale; the base model is the first dot and a dashed line."""
    rows = [(base_label, base, None)] + rows
    ys = list(range(len(rows)))[::-1]
    ax.axvline(base, color=MUTED, linewidth=1, linestyle="--", zorder=1)
    for y, (label, v, rule) in zip(ys, rows):
        color = MUTED if rule is None else (SET_RULE if rule else LOCAL)
        ax.plot([base, v], [y, y], color=GRID, linewidth=2, zorder=1)
        ax.plot(v, y, "o", color=color, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        d = v - base
        text = f"{v:.4f}" if rule is None else f"{v:.4f}  ({'0.0000' if abs(d) < 5e-5 else f'{d:+.4f}'})"
        ax.text(max(v, base) + 0.0012, y, text, va="center", color=INK, fontsize=9)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    lo, hi = min(r[1] for r in rows), max(r[1] for r in rows)
    ax.set_xlim(lo - 0.004, hi + 0.012)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("mIoU")
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")


def fig_gains():
    # exact values: BCSS runs/*/val/summary.csv; PanNuke organ_wcsp/breast/evaluation.json (branch organ-wcsp-breast)
    bcss = [("Pixel WCSP", 0.650602, False),
            ("WCSP + presence", 0.654962, True),
            ("Cluster WCSP + presence", 0.656387, True)]
    pannuke = [("Pixel learned table", 0.432412, False),
               ("Hierarchy", 0.432328, False),
               ("Scaling / siblings", 0.432258, False),
               ("Decay", 0.432305, False),
               ("Nucleus pairwise", 0.433482, False),
               ("Type counts", 0.430543, False),
               ("Raw TP components", 0.435567, False),
               ("Patch-wide type set", 0.448441, True)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"wspace": 0.85, "width_ratios": [1.1, 1]})
    miou_panel(axes[0], "PanNuke Breast test (169 patches)", "HoVer-Net", 0.432348, pannuke)
    miou_panel(axes[1], "BCSS val (2,124 tiles)", "U-Net", 0.648852, bcss)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=9) for c in (MUTED, LOCAL, SET_RULE)]
    fig.legend(handles, ["Base model", "Smoothing / local structure", "Presence (class-set rule)"],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.12))
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
