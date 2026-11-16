"""Plot paired differences for the frozen PanNuke WCSP variants."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prepare_fresh import HERE, source


def main():
    final = source.load(HERE / "evaluation.json")
    raw = source.load(HERE / "raw_only_evaluation.json")
    frozen = source.load(HERE / "frozen.json")
    presence, counts = (frozen[name]["name"] for name in ("presence", "count"))
    c = final["comparisons"]
    rows = [
        ("Raw-only component WCSP vs. Raw TP", raw),
        ("Fixed-mask vote vs. Raw TP", c["vote minus raw_tp"]),
        ("Type-set WCSP vs. Raw TP", c[f"{presence} minus raw_tp"]),
        ("Type-set WCSP vs. fixed-mask vote", c[f"{presence} minus vote"]),
        ("Count WCSP vs. fixed-mask vote", c[f"{counts} minus vote"]),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    ax.axvline(0, linestyle="--", color="0.5", lw=1)
    for i, (_, result) in enumerate(rows):
        low, high = result["ci95"]
        color = "#29725a" if low > 0 else "#2b6289"
        ax.hlines(i, 100 * low, 100 * high, color=color, lw=2)
        ax.plot(100 * result["delta"], i, "o", color=color)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.invert_yaxis()
    ax.margins(y=.20)
    ax.set(xlabel="Biological mIoU change (pp), with 95% paired CI",
           title="PanNuke WCSP · 100 newly sampled patches")
    fig.savefig(HERE / "comparison.png", dpi=165)


if __name__ == "__main__":
    main()
