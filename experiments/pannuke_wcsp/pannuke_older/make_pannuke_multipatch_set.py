from pathlib import Path
from collections import Counter

import pandas as pd


# ============================================================
# Configuration
# ============================================================

TARGET_PATCHES = 20
RANDOM_SEED = 401

MAX_PER_TISSUE = 2
MAX_PER_CLASS_SIGNATURE = 5


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

output_path = (
    repo
    / "experiments"
    / "pannuke_d5"
    / "pannuke_multi_patch_selection.csv"
)


# ============================================================
# Load candidates
# ============================================================

df = pd.read_csv(candidate_path)

print("Candidate patches:", len(df))


# ============================================================
# Basic eligibility
# ============================================================
#
# We want:
# - HoVer-Net predictions
# - at least 3 GT biological classes
# - at least a few predicted nuclei
#
# We prefer 4-class patches, but 3-class patches can be used
# if needed for diversity.
# ============================================================

eligible = df[
    (df["has_hovernet_predictions"] == True)
    & (df["num_gt_classes"] >= 3)
    & (df["hovernet_predicted_nuclei"] >= 5)
].copy()

print("Eligible patches:", len(eligible))


# ============================================================
# Shuffle reproducibly
# ============================================================
#
# Separate 4-class and 3-class patches so that 4-class patches
# receive priority, but randomize within each group to avoid
# simply choosing the densest/easiest-looking examples.
# ============================================================

four_class = eligible[
    eligible["num_gt_classes"] >= 4
].sample(
    frac=1,
    random_state=RANDOM_SEED,
)

three_class = eligible[
    eligible["num_gt_classes"] == 3
].sample(
    frac=1,
    random_state=RANDOM_SEED + 1,
)

pool = pd.concat(
    [four_class, three_class],
    ignore_index=True,
)


# ============================================================
# Greedy balanced selection
# ============================================================

selected_rows = []

tissue_counts = Counter()
signature_counts = Counter()

used_indices = set()


for _, row in pool.iterrows():

    if len(selected_rows) >= TARGET_PATCHES:
        break

    tissue = row["tissue_type"]
    signature = row["gt_classes"]
    image_index = int(row["image_index"])

    if tissue_counts[tissue] >= MAX_PER_TISSUE:
        continue

    if signature_counts[signature] >= MAX_PER_CLASS_SIGNATURE:
        continue

    selected_rows.append(row)

    tissue_counts[tissue] += 1
    signature_counts[signature] += 1
    used_indices.add(image_index)


# ============================================================
# Fallback
# ============================================================
#
# If the strict diversity rules did not give us 20 patches,
# fill the remaining positions from unused eligible patches.
# ============================================================

if len(selected_rows) < TARGET_PATCHES:

    for _, row in pool.iterrows():

        if len(selected_rows) >= TARGET_PATCHES:
            break

        image_index = int(row["image_index"])

        if image_index in used_indices:
            continue

        selected_rows.append(row)
        used_indices.add(image_index)


selected = pd.DataFrame(selected_rows)


# ============================================================
# Save
# ============================================================

selected.to_csv(
    output_path,
    index=False,
)


# ============================================================
# Display
# ============================================================

columns = [
    "image_index",
    "image",
    "tissue_type",
    "num_gt_classes",
    "gt_classes",
    "gt_nucleus_pixels",
    "hovernet_predicted_nuclei",
]

print("\nSelected multi-patch set:\n")

print(
    selected[columns]
    .to_string(index=False)
)


print("\nTissue distribution:")

print(
    selected["tissue_type"]
    .value_counts()
    .to_string()
)


print("\nClass-signature distribution:")

print(
    selected["gt_classes"]
    .value_counts()
    .to_string()
)


print("\nSaved:")
print(output_path)