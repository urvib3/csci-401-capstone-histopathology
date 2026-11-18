"""Render representative and worst losses of combined decay versus equal-weight V1."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from run import HERE, ROOT, cache_inputs, core, evaluate_patch, read, write

COLORS = ["#171717", "#e64b55", "#409fda", "#3eb27d", "#f1ba45", "#bb76d2"]
LABELS = ["Background", "Neoplastic", "Inflammatory", "Connective", "Dead", "Non-neoplastic"]


def main():
    cohort = read(HERE / "protocol.json")["comparison"]
    configs = {c["name"]: c for c in read(HERE / "frozen.json")["configurations"]}
    with np.load(HERE / "confusions.npz") as f:
        v1, decay = f["v1"], f["combined"]
    score = lambda x: core.metrics(x)["biological_miou"]
    deltas = np.array([score(b) - score(a) for a, b in zip(v1, decay)])
    losses = sorted(np.flatnonzero(deltas < -1e-12), key=lambda i: deltas[i])
    picks = [losses[len(losses) // 2], losses[0]]
    rows = [cohort[i] for i in picks]
    cache = ROOT / "presence_scale_20261114/cache"
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.4), layout="constrained")
    details = []
    cmap = ListedColormap(COLORS)
    for row, (i, patch) in enumerate(zip(picks, cache_inputs(cache, rows, False))):
        idx, _, gt, *_ = patch
        before, _ = evaluate_patch(configs["v1"], patch)
        after, _ = evaluate_patch(configs["combined"], patch)
        np.testing.assert_array_equal(core.confusion(gt, before), v1[i])
        np.testing.assert_array_equal(core.confusion(gt, after), decay[i])
        changed = before != after
        fixed = changed & (before != gt) & (after == gt)
        broken = changed & (before == gt) & (after != gt)
        label = "Median loss" if row == 0 else "Largest loss"
        transitions = np.bincount((before[broken] * 6 + after[broken]).astype(int), minlength=36)
        mistakes = int(transitions.argmax())
        details.append(dict(patch_id=idx, case=label, v1=score(v1[i]), decay=score(decay[i]),
                            delta=float(deltas[i]), fixed=int(fixed.sum()), broken=int(broken.sum()),
                            wrong_to_wrong=int((changed & ~fixed & ~broken).sum()),
                            main_break=[LABELS[mistakes//6], LABELS[mistakes%6], int(transitions[mistakes])]))
        delta_map = np.zeros((*gt.shape, 3), dtype=np.uint8)
        delta_map[changed] = (125, 125, 125)
        delta_map[fixed] = (62, 178, 125)
        delta_map[broken] = (230, 75, 85)
        images = [np.load(cache / f"rgb_{idx:04d}.npy"), gt, before, after, delta_map]
        titles = [f"{label}\nPatch {idx}", "Ground truth", f"V1 · {score(v1[i]):.4f}",
                  f"Combined decay · {score(decay[i]):.4f}", f"{fixed.sum()} fixed / {broken.sum()} broke"]
        for ax, image, title in zip(axes[row], images, titles):
            ax.imshow(image, cmap=cmap, vmin=0, vmax=5, interpolation="nearest")
            ax.set_title(title, fontsize=10)
            ax.axis("off")
    fig.supxlabel("Change map: green = fixed, red = broke, gray = changed but still wrong", fontsize=10)
    fig.savefig(HERE / "decay_failure_examples.png", dpi=155)
    plt.close(fig)
    result = dict(wins=int((deltas > 1e-12).sum()), losses=len(losses),
                  ties=int((abs(deltas) <= 1e-12).sum()), examples=details)
    write("decay_failures.json", result)
    print(result)


if __name__ == "__main__":
    main()
