"""Plot measured WCSP effects separately from the oracle diagnostic."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prepare import HERE, load


def main():
    final = load(HERE / "evaluation.json")
    frozen = load(HERE / "frozen.json")
    wcsp, unary = frozen["wcsp"]["name"], frozen["unary_control"]["name"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
    metrics = final["rows"]
    names = ["Raw TP", "Instance vote", "Nucleus WCSP", "Correct-type diagnostic"]
    scores = [final["raw_bio_miou"], metrics["instance_vote"]["bio_miou"],
              metrics[wcsp]["bio_miou"], metrics["type_oracle"]["bio_miou"]]
    colors = ["#838d99", "#34719c", "#34719c", "#d79456"]
    bars = left.barh(names, scores, color=colors)
    left.invert_yaxis()
    left.set(xlabel="Pooled biological mIoU", title="Fixed-mask type diagnostic")
    left.set_xlim(0, max(scores) * 1.19)
    left.bar_label(bars, labels=[f"{x:.4f}" for x in scores], padding=4, fontsize=9)

    comparisons = [(f"{wcsp} minus instance_vote", "WCSP − instance vote"),
                   (f"{wcsp} minus {unary}", "WCSP − unary only"),
                   (f"{wcsp} minus raw_tp", "WCSP − Raw TP")]
    for i, (key, label) in enumerate(comparisons):
        item = final["comparisons"][key]
        lo, hi = [100 * n for n in item["ci95"]]
        right.hlines(i, lo, hi, color="#34719c", lw=2)
        right.plot(100 * item["delta"], i, "o", color="#34719c")
    right.set_yticks(range(len(comparisons)), [label for _, label in comparisons])
    right.invert_yaxis()
    right.set_ylim(len(comparisons)-.5, -.5)
    right.axvline(0, linestyle="--", color="0.5", lw=1)
    right.set(xlabel="mIoU difference (pp), with 95% paired CI", title="Measured refinement effect")
    fig.suptitle("PanNuke nucleus-type WCSP · 100 newly sampled full patches")
    fig.savefig(HERE / "comparison.png", dpi=165)


if __name__ == "__main__":
    main()
