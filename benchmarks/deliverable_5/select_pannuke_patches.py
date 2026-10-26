from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

repo = Path(__file__).resolve().parents[2]

pannuke_dir = repo / "benchmarks" / "pannuke"

masks_path = (
    pannuke_dir
    / "data"
    / "masks"
    / "fold2"
    / "masks.npy"
)

types_path = (
    pannuke_dir
    / "data"
    / "images"
    / "fold2"
    / "types.npy"
)

prediction_csv = list(
    pannuke_dir.rglob("hovernet_fold2_predictions.csv")
)[0]


# ============================================================
# Labels
# ============================================================

CLASS_NAMES = [
    "Neoplastic",
    "Inflammatory",
    "Connective",
    "Dead",
    "Non-neoplastic epithelial",
]


# ============================================================
# Load data
# ============================================================

print("Loading PanNuke masks...")

masks = np.load(
    masks_path,
    mmap_mode="r",
)

types = np.load(
    types_path,
    allow_pickle=True,
)

pred = pd.read_csv(prediction_csv)


print("Masks:", masks.shape)
print("Tissue types:", types.shape)
print("Prediction rows:", len(pred))


# ============================================================
# Count HoVer-Net predictions per image
# ============================================================

prediction_counts = (
    pred.groupby("image_index")
    .size()
    .to_dict()
)


# ============================================================
# Scan every patch
# ============================================================

rows = []

for i in range(len(masks)):

    patch = masks[i]

    class_pixel_counts = []

    for channel in range(5):
        count = int(
            np.count_nonzero(
                patch[:, :, channel] > 0
            )
        )
        class_pixel_counts.append(count)

    present_classes = [
        CLASS_NAMES[c]
        for c, count in enumerate(class_pixel_counts)
        if count > 0
    ]

    num_classes = len(present_classes)

    gt_nucleus_pixels = sum(class_pixel_counts)

    hovernet_nuclei = int(
        prediction_counts.get(i, 0)
    )

    rows.append({
        "image_index": i,
        "image": f"image_{i}",
        "tissue_type": str(types[i]),
        "num_gt_classes": num_classes,
        "gt_classes": "; ".join(present_classes),
        "gt_nucleus_pixels": gt_nucleus_pixels,
        "hovernet_predicted_nuclei": hovernet_nuclei,
        "has_hovernet_predictions": hovernet_nuclei > 0,

        "gt_neoplastic_pixels": class_pixel_counts[0],
        "gt_inflammatory_pixels": class_pixel_counts[1],
        "gt_connective_pixels": class_pixel_counts[2],
        "gt_dead_pixels": class_pixel_counts[3],
        "gt_non_neoplastic_pixels": class_pixel_counts[4],
    })


df = pd.DataFrame(rows)


# ============================================================
# Save full catalog
# ============================================================

out_dir = (
    repo
    / "benchmarks"
    / "deliverable_5"
)

out_dir.mkdir(
    parents=True,
    exist_ok=True,
)

catalog_path = (
    out_dir
    / "pannuke_patch_catalog.csv"
)

df.to_csv(
    catalog_path,
    index=False,
)


# ============================================================
# Candidate patches
# ============================================================

candidates = df[
    (df["has_hovernet_predictions"])
    & (df["num_gt_classes"] >= 2)
].copy()

candidates = candidates.sort_values(
    by=[
        "num_gt_classes",
        "gt_nucleus_pixels",
    ],
    ascending=[
        False,
        False,
    ],
)


candidate_path = (
    out_dir
    / "pannuke_candidate_patches.csv"
)

candidates.to_csv(
    candidate_path,
    index=False,
)


# ============================================================
# Print useful summary
# ============================================================

print("\nTotal Fold 2 patches:")
print(len(df))

print("\nPatches with HoVer-Net predictions:")
print(df["has_hovernet_predictions"].sum())

print("\nMulti-class patches with predictions:")
print(len(candidates))

print("\nTop candidate patches:")
print(
    candidates[
        [
            "image_index",
            "image",
            "tissue_type",
            "num_gt_classes",
            "gt_classes",
            "gt_nucleus_pixels",
            "hovernet_predicted_nuclei",
        ]
    ].head(25).to_string(index=False)
)

print("\nSaved:")
print(catalog_path)
print(candidate_path)