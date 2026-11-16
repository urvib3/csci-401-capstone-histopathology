"""Render the frozen replication and its Dead-excluded sensitivity."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prepare_scale import HERE, source


def main():
    report = source.load(HERE / "evaluation.json")
    key = report["config"]["name"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True, layout="constrained")
    results = report["cohorts"]
    labels = ["WCSP vs. Raw TP", "WCSP vs. fixed-mask vote",
              "WCSP vs. Raw TP, without Dead", "WCSP vs. vote, without Dead"]
    for ax, cohort, title in zip(axes, ("primary", "dead_stress"),
                                  ("Random sample", "Dead-positive stress sample")):
        data = results[cohort]
        ax.axvline(0, color="0.5", ls="--", lw=1)
        for i, (baseline, omit_dead) in enumerate([("raw_tp", False), ("vote", False),
                                                  ("raw_tp", True), ("vote", True)]):
            row = data["comparisons"][key + " minus " + baseline]
            low, high = row["no_dead_ci95" if omit_dead else "ci95"]
            value = row["no_dead_delta" if omit_dead else "delta"]
            color = "#2b816a" if low > 0 else "#2c6089"
            ax.hlines(i, low * 100, high * 100, color=color, lw=2)
            ax.plot(value * 100, i, "o", color=color)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_ylim(len(labels)-.5, -.5)
        ax.set(xlabel="mIoU difference (pp), 95% paired CI",
               title=f'{title} · {data["n_patches"]} patches')
    fig.suptitle("PanNuke · Frozen type-set WCSP replication")
    fig.savefig(HERE / "comparison.png", dpi=165)


if __name__ == "__main__":
    main()
