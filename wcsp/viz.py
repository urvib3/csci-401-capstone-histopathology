# Side-by-side figures: image | ground truth | classifier | WCSP | changed pixels.

import numpy as np

from .labels import COLORS, IGNORE, LABELS, colorize


def overlay(image, labels, alpha=0.45):
    """Blend label colours onto an RGB image; IGNORE pixels keep the image."""
    image = np.asarray(image, dtype=np.float64)
    color = colorize(labels).astype(np.float64)
    mix = np.where((np.asarray(labels) != IGNORE)[..., None], (1 - alpha) * image + alpha * color, image)
    return mix.astype(np.uint8)


def comparison_figure(image, target, baseline, refined, title="", path=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    changed = np.asarray(baseline) != np.asarray(refined)
    panels = [
        ("image", np.asarray(image)),
        ("ground truth", overlay(image, target) if target is not None else image),
        ("classifier (argmax)", colorize(baseline)),
        ("WCSP", colorize(refined)),
        (f"changed ({changed.mean():.1%})", np.where(changed[..., None], colorize(refined), 255).astype(np.uint8)),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
    for ax, (name, panel) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    handles = [Patch(color=c / 255, label=n) for n, c in zip(LABELS, COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=len(LABELS), fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    if path:
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig
