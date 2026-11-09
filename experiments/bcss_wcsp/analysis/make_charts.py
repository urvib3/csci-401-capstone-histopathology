# Charts for Deliverable 6 (static PNGs for the markdown report).
#   val_gains.png       val mIoU gain over the U-Net per method, with ROI-bootstrap 95% CIs
#   error_location.png  where the U-Net's wrong pixels are vs. what the WCSP changes
#   rule_check.png      tile-level regularities: ground truth vs. U-Net vs. WCSP
#
#   python experiments/bcss_wcsp/analysis/make_charts.py

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = Path(os.path.expandvars("$WCSP_OUT/runs"))
OUT = Path("experiments/bcss_wcsp/figures")
ANALYSIS = Path("experiments/bcss_wcsp/analysis")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HIGHLIGHT, BASE, SECOND = "#2a78d6", "#a9a8a2", "#eb6834"

# (label, run name); the first is the reference everyone is compared against
METHODS = [
    ("Current WCSP (pixel neighbors)", "valfull_data_lam8"),
    ("Current WCSP, run to convergence", "valfull_data_lam8_conv50"),
    ("Multi-scale blocks", "valfull_msd_mu0.5"),
]
EXTRA = json.loads(os.environ.get("CHART_EXTRA_METHODS", "[]"))   # [[label, run], ...]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def val_gains():
    rows = []
    for label, run in METHODS + [tuple(m) for m in EXTRA]:
        path = RUNS / run / "val" / "bootstrap.json"
        if path.exists():
            b = json.loads(path.read_text())
            rows.append((label, b["delta_mIoU"], b["ci_low"], b["ci_high"]))
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(rows) + 1.2), facecolor=SURFACE)
    style(ax)
    best = max(r[1] for r in rows)
    for y, (label, d, lo, hi) in enumerate(rows[::-1]):
        color = HIGHLIGHT if d == best else BASE
        ax.barh(y, d, height=0.5, color=color)
        ax.plot([lo, hi], [y, y], color=INK, lw=1.2)
        ax.text(hi + 0.0001, y, f"+{d:.4f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=9, color=INK)
    ax.set_xlabel("Val mIoU gain over the U-Net (2,124 tiles; bars = 95% CI)", fontsize=9, color=MUTED)
    ax.set_xlim(0, max(r[3] for r in rows) * 1.25)
    fig.tight_layout()
    fig.savefig(OUT / "val_gains.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def error_location():
    e = json.loads((ANALYSIS / "error_structure_val_data.json").read_text())
    bins = e["errors_by_boundary_distance"]
    labels = list(bins)
    share = [bins[k]["share_of_errors"] * 100 for k in labels]
    changed = [bins[k]["wcsp_fixed"] + bins[k]["wcsp_broken"] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor=SURFACE)
    style(ax)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    colors = [HIGHLIGHT if c > 0 else BASE for c in changed]
    ax.bar(range(len(labels)), share, width=0.6, color=colors)
    for x, (s, c) in enumerate(zip(share, changed)):
        note = f"{s:.0f}%" + (f"\nWCSP changed {c / 1e3:.0f}k px" if c else "\nWCSP never reaches")
        ax.text(x, s + 0.8, note, ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(labels)), [k.replace("px", "px from a border") for k in labels], fontsize=8.5)
    ax.set_ylabel("Share of the U-Net's wrong pixels", fontsize=9, color=MUTED)
    ax.set_ylim(0, max(share) * 1.35)
    ax.set_title("Most U-Net errors are far from any border, where neighbor rules can't reach",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "error_location.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def rule_check():
    r = json.loads((ANALYSIS / "region_stats.json").read_text())
    checks = [
        ("Inflammatory in tile | necrosis", "p_b_given_a", "Necrosis", "Inflammatory"),
        ("Other in tile | necrosis", "p_b_given_a", "Necrosis", "Other"),
        ("Other in tile | tumor", "p_b_given_a", "Tumor", "Other"),
        ("Other's border touching tumor", "border_share", "Other", "Tumor"),
        ("Necrosis border touching other", "border_share", "Necrosis", "Other"),
    ]
    series = [("Ground truth (fit tiles)", "truth_fit", INK, "o"), ("U-Net (val)", "unet_val", BASE, "s"),
              ("After current WCSP (val)", "wcsp_val", SECOND, "D")]
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    style(ax)
    for name, key, color, marker in series:
        xs = [r[key][kind][a][b] for _, kind, a, b in checks]
        ax.scatter(xs, range(len(checks)), s=60, color=color, marker=marker, label=name, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
    for y, (_, kind, a, b) in enumerate(checks):
        xs = [r[key][kind][a][b] for _, key, _, _ in series]
        ax.plot([min(xs), max(xs)], [y, y], color=GRID, lw=2, zorder=1)
    ax.set_yticks(range(len(checks)), [c[0] for c in checks], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Probability / share", fontsize=9, color=MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=3)
    ax.set_title("The WCSP keeps class combinations that the ground truth almost never has",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "rule_check.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    val_gains()
    error_location()
    rule_check()
    print("wrote val_gains.png, error_location.png, rule_check.png")
