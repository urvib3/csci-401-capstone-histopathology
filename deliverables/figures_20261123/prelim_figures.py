# Preliminary-test figures for deliverables/deliverable_7.md, from Ryan's 185 frozen full-res
# dev inputs and the reports of rules/fullres_presence_dev.py, oracle_tests.py, grouping_tests.py.
#
#   python deliverables/figures_20261123/prelim_figures.py

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.bcss_wcsp.rules.oracle_tests import region_oracle  # noqa: E402

OUT = Path(__file__).resolve().parent
INPUTS = Path("data/fullres_multiscale_20261115/inputs")
RUNS = Path("/scratch1/ubhuwani/wcsp/runs")
NAMES = ["Tumor", "Inflammatory", "Stroma", "Necrosis", "Other"]
IGNORE, K = 255, 5
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a9a8a2"
# tissue map colours (categorical, one per class) + light gray for unlabeled
TISSUE = ["#d64541", "#2f9e44", "#5c7cfa", "#f2c037", "#c56cf0"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
})


def title(ax, text):
    ax.set_title(text, loc="left", color=INK, fontsize=11, fontweight="bold")


def per_class_split(paths):
    """Wrong pixels per true class, split into grouping vs labeling errors."""
    group, label = np.zeros(K), np.zeros(K)
    for p in paths:
        z = np.load(p)
        target, pred = z["target"], z["probs"].astype(np.float32).argmax(0)
        valid = target != IGNORE
        for c in range(K):
            comp, n = ndimage.label((target == c) & valid, structure=np.ones((3, 3)))
            for r in range(1, n + 1):
                region = comp == r
                wrong = (pred[region] != c).sum()
                if np.bincount(pred[region], minlength=K).argmax() == c:
                    group[c] += wrong
                else:
                    label[c] += wrong
    return group, label


def fig_confusion(conf):
    rows = conf / conf.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.imshow(rows, cmap="Blues", vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{rows[i, j]:.2f}", ha="center", va="center",
                    color="#ffffff" if rows[i, j] > 0.55 else INK, fontsize=9)
    ax.set_xticks(range(K), NAMES, rotation=30, ha="right")
    ax.set_yticks(range(K), NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for s in ax.spines.values():
        s.set_visible(False)
    title(ax, "Which classes are confused (full-res U-Net)")
    fig.savefig(OUT / "prelim_confusion.png", dpi=200, bbox_inches="tight")


def fig_split(group, label, conf):
    total = conf.sum(1)
    g, l = 100 * group / total, 100 * label / total
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ys = np.arange(K)[::-1]
    ax.barh(ys, g, color=BLUE, height=0.6, edgecolor=SURFACE, linewidth=2, label="Segmentation error (region mostly right, pixels break away)")
    ax.barh(ys, l, left=g, color=ORANGE, height=0.6, edgecolor=SURFACE, linewidth=2,
            label="Class identification error (whole region wrong class)")
    for y, a, b in zip(ys, g, l):
        ax.text(a + b + 0.8, y, f"{a + b:.0f}% wrong", va="center", color=INK, fontsize=9)
    ax.set_yticks(ys, NAMES)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(g + l) * 1.3)
    ax.set_xlabel("% of the class's pixels")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
    title(ax, "Test 1 · Which kind of error, per class")
    fig.savefig(OUT / "prelim_error_split.png", dpi=200, bbox_inches="tight")


def fig_ladder(raw, wcsp, presence, class_id, seg):
    rows = [("U-Net", raw, GRAY), ("+ WCSP smoothing", wcsp, GRAY), ("+ presence rule", presence, BLUE),
            ("Perfect class identification", class_id, BLUE), ("Perfect segmentation", seg, ORANGE)]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ys = np.arange(len(rows))[::-1]
    for y, (name, v, c) in zip(ys, rows):
        ax.barh(y, v, color=c, height=0.6, edgecolor=SURFACE, linewidth=2)
        ax.text(v + 0.004, y, f"{v:.4f}" if v == raw else f"{v:.4f}  ({v - raw:+.4f})", va="center", color=INK, fontsize=9)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    lo = raw - 0.02
    ax.set_xlim(lo, max(r[1] for r in rows) + 0.06)
    ax.set_xlabel("mIoU (axis starts at %.2f; both perfect = 1.0)" % lo)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    title(ax, "Test 2 · What fixing one problem is worth")
    fig.savefig(OUT / "prelim_ladder.png", dpi=200, bbox_inches="tight")


def fig_example(paths):
    """The tile with the most predicted pieces per true region: truth, U-Net, region oracle."""
    best, score = None, -1
    for p in paths:
        z = np.load(p)
        target, pred = z["target"], z["probs"].astype(np.float32).argmax(0)
        valid = target != IGNORE
        if valid.mean() < 0.9 or len(np.unique(target[valid])) < 2:
            continue
        n_pred = sum(ndimage.label((pred == c) & valid)[1] for c in range(K))
        n_true = sum(ndimage.label((target == c) & valid)[1] for c in range(K))
        acc = (pred[valid] == target[valid]).mean()
        s = n_pred / max(n_true, 1) * (acc > 0.6)
        if s > score:
            best, score = p, s
    z = np.load(best)
    target, probs = z["target"], z["probs"].astype(np.float32)
    pred = probs.argmax(0)
    oracle = region_oracle(probs.astype(np.float64), target)
    cmap = ListedColormap(TISSUE + ["#e4e3df"])
    show = lambda a: np.where(target == IGNORE, K, a)  # noqa: E731
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, img, name in zip(axes, (show(target), show(pred), show(oracle)),
                             ("Ground truth", "U-Net (raw)", "Region oracle")):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=K, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(name, color=INK, fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TISSUE]
    fig.legend(handles, NAMES, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Fragmentation example: {z['id'].item()[:23]}…", x=0.01, ha="left", color=INK, fontsize=11,
                 fontweight="bold")
    fig.savefig(OUT / "prelim_example.png", dpi=200, bbox_inches="tight")


def main():
    paths = sorted(INPUTS.glob("*.npz"))
    conf = np.load(RUNS / "fullres_presence_dev/confusions.npz")["raw"].sum(0)
    pres = json.loads((RUNS / "fullres_presence_dev/report.json").read_text())
    orac = json.loads((RUNS / "fullres_oracles_dev/report.json").read_text())
    fig_confusion(conf)
    group, label = per_class_split(paths)
    fig_split(group, label, conf)
    fig_ladder(pres["raw"]["mIoU"], pres["wcsp"]["mIoU"], pres["presence"]["mIoU"],
               orac["presence_oracle"]["mIoU"], orac["region_oracle"]["mIoU"])
    fig_example(paths)
    split = {n: {"wrong_pct": round(100 * (g + l) / t, 1), "grouping_pct": round(100 * g / t, 1),
                 "labeling_pct": round(100 * l / t, 1)} for n, g, l, t in zip(NAMES, group, label, conf.sum(1))}
    rows = conf / conf.sum(1, keepdims=True)
    print(json.dumps({"split": split, "confusion_rows": np.round(rows, 3).tolist()}, indent=1))


if __name__ == "__main__":
    main()


# ---- cleaner BCSS preliminary-test figures ------------------------------------------------

def fig_split_diagram():
    """Two toy regions: a segmentation error (region mostly right) and a class identification
    error (region mostly wrong). Colours: stroma blue, tumor red, other pink (wcsp.labels)."""
    from wcsp.labels import COLORS
    col = lambda k: np.asarray(COLORS[k]) / 255  # noqa: E731
    stroma, tumor, other = 2, 0, 4
    n = 10
    truth = np.full((n, n), stroma)
    seg = truth.copy()
    seg[6:9, 6:9] = tumor; seg[0, 0] = tumor                               # 10 of 100 wrong
    cls = np.full((n, n), other)
    cls[:, :3] = stroma                                                    # 70 of 100 wrong
    fig, axes = plt.subplots(1, 5, figsize=(11, 2.9), gridspec_kw={"width_ratios": [1, 1, 0.25, 1, 1]})
    def grid(ax, labels, title_text):
        for (r, c), k in np.ndenumerate(labels):
            ax.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1, facecolor=col(k), edgecolor=SURFACE, linewidth=1.5))
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_aspect("equal")
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title_text, color=INK, fontsize=10)
    grid(axes[0], truth, "True region: stroma")
    grid(axes[1], seg, "U-Net")
    grid(axes[3], truth, "True region: stroma")
    grid(axes[4], cls, "U-Net")
    axes[2].axis("off")
    axes[1].text(0.5, -0.12, "Mostly stroma ✓ → the 10 wrong\npixels are SEGMENTATION errors",
                 transform=axes[1].transAxes, ha="center", va="top", color=BLUE, fontsize=10, fontweight="bold")
    axes[4].text(0.5, -0.12, "Mostly \"other\" ✗ → the 70 wrong\npixels are CLASS IDENTIFICATION errors",
                 transform=axes[4].transAxes, ha="center", va="top", color=ORANGE, fontsize=10, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=col(k)) for k in (stroma, tumor, other)]
    fig.legend(handles, ["Stroma", "Tumor", "Other"], loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    fig.savefig(OUT / "prelim_split_diagram.png", dpi=200, bbox_inches="tight")


def fig_split_v2(seg_pct, cls_pct, overall=(48, 52)):
    """Test 1: per class, % of the class's pixels wrong, split by error type; 'All' row on top."""
    names = ["All classes"] + NAMES
    rows = [None] + list(zip(seg_pct, cls_pct))
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ys = np.arange(len(names))[::-1]
    for y, name, r in zip(ys, names, rows):
        if r is None:   # overall: share of all wrong pixels, drawn on a 0-100 bar of its own
            a, b = overall
            ax.barh(y, a * 0.4, color=BLUE, height=0.62, edgecolor=SURFACE, linewidth=2)
            ax.barh(y, b * 0.4, left=a * 0.4, color=ORANGE, height=0.62, edgecolor=SURFACE, linewidth=2)
            ax.text(a * 0.2, y, f"{a}%", ha="center", va="center", color="#ffffff", fontsize=9, fontweight="bold")
            ax.text(a * 0.4 + b * 0.2, y, f"{b}%", ha="center", va="center", color="#ffffff", fontsize=9, fontweight="bold")
            ax.text(40.8, y, "of all wrong pixels", va="center", color=MUTED, fontsize=8.5)
            continue
        g, l = r
        ax.barh(y, g, color=BLUE, height=0.62, edgecolor=SURFACE, linewidth=2)
        ax.barh(y, l, left=g, color=ORANGE, height=0.62, edgecolor=SURFACE, linewidth=2)
        if g >= 4:
            ax.text(g / 2, y, f"{g:.0f}", ha="center", va="center", color="#ffffff", fontsize=8.5)
        if l >= 4:
            ax.text(g + l / 2, y, f"{l:.0f}", ha="center", va="center", color="#ffffff", fontsize=8.5)
        ax.text(g + l + 0.8, y, f"{g + l:.0f}% of {name.lower() if name != 'Other' else 'other'} pixels wrong",
                va="center", color=INK, fontsize=8.5)
    ax.axhline(ys[0] - 0.5, color=GRID, linewidth=1)
    ax.set_yticks(ys, names)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 56)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE)]
    ax.legend(handles, ["Segmentation error", "Class identification error"], frameon=False, ncol=2,
              loc="upper center", bbox_to_anchor=(0.45, 1.16))
    fig.savefig(OUT / "prelim_error_split.png", dpi=200, bbox_inches="tight")


def fig_ladder_v2(raw, wcsp, presence, class_id, seg):
    """Test 2: today's methods vs. each problem made perfect, on one mIoU axis up to 1.0."""
    rows = [("U-Net", raw, GRAY, "Today"), ("+ WCSP smoothing", wcsp, GRAY, "Today"),
            ("+ presence rule", presence, BLUE, "Today"),
            ("Perfect class identification", class_id, BLUE, "If perfect"),
            ("Perfect segmentation", seg, ORANGE, "If perfect"),
            ("Both perfect (= ground truth)", 1.0, "#d9d8d2", "If perfect")]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ys = np.array([6.4, 5.4, 4.4, 2.6, 1.6, 0.6])
    lo = 0.6
    for y, (name, v, c, _) in zip(ys, rows):
        ax.barh(y, v - lo, left=lo, color=c, height=0.66, edgecolor=SURFACE, linewidth=2)
        if v == 1.0:
            ax.text(0.99, y, "1.0", va="center", ha="right", color=INK, fontsize=9)
            continue
        txt = f"{v:.4f}" if name == "U-Net" else f"{v:.4f}  ({v - raw:+.4f})"
        ax.text(v + 0.006, y, txt, va="center", color=INK, fontsize=9)
    ax.axvline(raw, color=INK, linewidth=1, linestyle="--", zorder=2)
    for y, text in ((7.2, "Today"), (3.4, "If one problem were perfect")):
        ax.text(lo + 0.002, y, text, color=MUTED, fontsize=9.5, fontweight="bold", va="center", zorder=5,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(lo, 1.08)
    ax.set_ylim(0, 7.6)
    ax.set_xlabel("mIoU (axis starts at 0.60; dashed line = U-Net)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    fig.savefig(OUT / "prelim_ladder.png", dpi=200, bbox_inches="tight")
