from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

repo = Path(__file__).resolve().parents[2]

candidate_path = (
    repo
    / "experiments"
    / "pannuke_d5"
    / "pannuke_candidate_patches.csv"
)

evaluation_set_path = (
    repo
    / "experiments"
    / "pannuke_d5"
    / "pannuke_multi_patch_selection.csv"
)

output_path = (
    repo
    / "experiments"
    / "pannuke_d5"
    / "pannuke_lambda_dev_selection.csv"
)


# ============================================================
# Load candidate pool + already-used evaluation set
# ============================================================

candidates = pd.read_csv(candidate_path)

used = pd.read_csv(
    evaluation_set_path
)

used_indices = set(
    used["image_index"].astype(int)
)


# ============================================================
# Remove the 20 patches already evaluated
# ============================================================

dev_pool = candidates[
    ~candidates["image_index"].isin(
        used_indices
    )
].copy()


# ============================================================
# Keep useful multi-class cases
# ============================================================

dev_pool = dev_pool[
    (dev_pool["num_gt_classes"] >= 3)
    & (
        dev_pool[
            "hovernet_predicted_nuclei"
        ] >= 5
    )
].copy()


print(
    "Eligible unused dev candidates:",
    len(dev_pool),
)


# ============================================================
# Reproducible development subset
# ============================================================

dev = dev_pool.sample(
    n=15,
    random_state=402,
).copy()


# ============================================================
# Save
# ============================================================

dev.to_csv(
    output_path,
    index=False,
)


print("\nLambda development set:\n")

print(
    dev[
        [
            "image_index",
            "image",
            "tissue_type",
            "num_gt_classes",
            "gt_classes",
            "hovernet_predicted_nuclei",
        ]
    ].to_string(
        index=False
    )
)

print("\nSaved:")
print(output_path)