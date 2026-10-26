from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

TARGET_IMAGE = "image_00748"
OFFSET = 46

CLASS_TO_ID = {
    "Neoplastic": 1,
    "Inflammatory": 2,
    "Connective": 3,
    "Dead": 4,
    "Non-neoplastic epithelial": 5,
}

ID_TO_CLASS = {
    0: "Background / inactive",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Non-neoplastic epithelial",
}


# ============================================================
# Find repository and existing D3/D4 artifacts
# ============================================================

repo = Path(__file__).resolve().parents[3]

pannuke_dir = repo / "benchmarks" / "pannuke"

csv_matches = list(
    pannuke_dir.rglob("hovernet_fold2_predictions.csv")
)

npz_matches = sorted(
    pannuke_dir.rglob("hovernet_instance_maps_*.npz")
)

masks_path = (
    pannuke_dir
    / "data"
    / "masks"
    / "fold2"
    / "masks.npy"
)

if not masks_path.exists():
    raise FileNotFoundError(
        f"Full Fold 2 masks not found at: {masks_path}"
    )
if not csv_matches:
    raise FileNotFoundError(
        "Could not find hovernet_fold2_predictions.csv"
    )

if not npz_matches:
    raise FileNotFoundError(
        "Could not find hovernet_instance_maps_*.npz"
    )



prediction_csv = csv_matches[0]


print("Prediction CSV:")
print(prediction_csv)

print("\nGround truth:")
print(masks_path)


# ============================================================
# Load HoVer-Net CSV
# ============================================================

pred = pd.read_csv(prediction_csv)

print("\nPrediction columns:")
print(pred.columns.tolist())


def normalize_name(name):
    return Path(str(name)).stem


# Identify image column
for candidate in ["image", "image_name", "filename"]:
    if candidate in pred.columns:
        image_col = candidate
        break
else:
    raise ValueError(
        "Could not determine image-name column."
    )


# Identify biological class column
for candidate in [
    "predicted_type",
    "predicted_class",
    "type_name",
]:
    if candidate in pred.columns:
        class_col = candidate
        break
else:
    raise ValueError(
        "Could not determine predicted-class column."
    )


if (
    "centroid_x" not in pred.columns
    or "centroid_y" not in pred.columns
):
    raise ValueError(
        "Expected centroid_x and centroid_y columns."
    )


pred["_normalized_image"] = (
    pred[image_col].map(normalize_name)
)

target_rows = pred[
    pred["_normalized_image"] == TARGET_IMAGE
].copy()

print(
    f"\nPredicted nuclei for {TARGET_IMAGE}:",
    len(target_rows),
)

if target_rows.empty:
    raise RuntimeError(
        f"No CSV predictions found for {TARGET_IMAGE}"
    )


# ============================================================
# Locate the correct 164x164 instance map
# ============================================================

instance_map = None
source_npz = None


for npz_path in npz_matches:

    data = np.load(
        npz_path,
        allow_pickle=True,
    )

    map_key = None
    name_key = None

    # Identify the arrays stored in this NPZ
    for key in data.files:

        arr = np.asarray(data[key])

        if (
            arr.ndim == 3
            and arr.shape[-2:] == (164, 164)
        ):
            map_key = key

        elif arr.ndim == 1:
            name_key = key

    if map_key is None or name_key is None:
        continue

    maps = np.asarray(data[map_key])
    names = [
        normalize_name(x)
        for x in np.asarray(data[name_key])
    ]

    if TARGET_IMAGE in names:

        idx = names.index(TARGET_IMAGE)

        instance_map = maps[idx]
        source_npz = npz_path

        print(
            "\nFound instance map in:",
            npz_path.name,
        )

        print(
            "Instance map shape:",
            instance_map.shape,
        )

        break


if instance_map is None:
    raise RuntimeError(
        f"Could not locate instance map for {TARGET_IMAGE}"
    )


# ============================================================
# Convert HoVer-Net instances -> biological pixel labels
# ============================================================

predicted_labels = np.zeros(
    instance_map.shape,
    dtype=np.uint8,
)

instance_to_class = {}


for _, row in target_rows.iterrows():

    class_name = str(row[class_col])

    # Ignore type 0 / unknown / background
    if class_name not in CLASS_TO_ID:
        continue

    x = int(round(float(row["centroid_x"])))
    y = int(round(float(row["centroid_y"])))

    if not (
        0 <= x < instance_map.shape[1]
        and 0 <= y < instance_map.shape[0]
    ):
        continue

    instance_id = int(instance_map[y, x])

    # Centroid unexpectedly falls on background
    if instance_id == 0:
        continue

    instance_to_class[instance_id] = (
        CLASS_TO_ID[class_name]
    )


for instance_id, class_id in instance_to_class.items():

    predicted_labels[
        instance_map == instance_id
    ] = class_id


active_mask = predicted_labels > 0


print(
    "\nMapped nucleus instances:",
    len(instance_to_class),
)

print(
    "Active predicted pixels:",
    int(active_mask.sum()),
)

print(
    "Inactive/background pixels:",
    int((~active_mask).sum()),
)


# ============================================================
# Load aligned PanNuke ground truth
# ============================================================

# image_00748 -> dataset index 748
image_index = int(
    TARGET_IMAGE.split("_")[1]
)

masks = np.load(
    masks_path,
    mmap_mode="r",
)

print(
    "\nGround-truth masks shape:",
    masks.shape,
)

gt_full = masks[image_index]

if gt_full.shape[:2] != (256, 256):
    raise ValueError(
        f"Expected 256x256 ground truth, got "
        f"{gt_full.shape}"
    )


# PanNuke channels:
# 0 Neoplastic
# 1 Inflammatory
# 2 Connective
# 3 Dead
# 4 Non-neoplastic epithelial
# 5 Background
gt_labels_full = np.zeros(
    (256, 256),
    dtype=np.uint8,
)

for channel in range(5):

    gt_labels_full[
        gt_full[:, :, channel] > 0
    ] = channel + 1


# HoVer-Net valid output corresponds to
# [46:210, 46:210] in the original patch.
gt_labels = gt_labels_full[
    OFFSET:OFFSET + 164,
    OFFSET:OFFSET + 164,
]

if gt_labels.shape != (164, 164):
    raise RuntimeError(
        f"Unexpected cropped GT shape: "
        f"{gt_labels.shape}"
    )


# ============================================================
# Basic alignment diagnostics
# ============================================================

gt_active = gt_labels > 0

overlap = active_mask & gt_active

predicted_only = (
    active_mask & ~gt_active
)

ground_truth_only = (
    gt_active & ~active_mask
)


if overlap.sum() > 0:
    overlap_class_accuracy = float(
        np.mean(
            predicted_labels[overlap]
            == gt_labels[overlap]
        )
    )
else:
    overlap_class_accuracy = None


print("\nAlignment diagnostics")

print(
    "Ground-truth nucleus pixels:",
    int(gt_active.sum()),
)

print(
    "Prediction / GT overlap pixels:",
    int(overlap.sum()),
)

print(
    "Predicted nucleus pixels where GT is background:",
    int(predicted_only.sum()),
)

print(
    "GT nucleus pixels missed by predicted active mask:",
    int(ground_truth_only.sum()),
)

print(
    "Class agreement on overlapping nucleus pixels:",
    overlap_class_accuracy,
)


# ============================================================
# Save everything for Tasks 2 and 3
# ============================================================

out_dir = (
    repo
    / "experiments"
    / "pannuke_d5"
    / "pannuke_aakanksha"
)

out_dir.mkdir(
    parents=True,
    exist_ok=True,
)


np.save(
    out_dir / "hovernet_labels_164.npy",
    predicted_labels,
)

np.save(
    out_dir / "active_mask_164.npy",
    active_mask,
)

np.save(
    out_dir / "pannuke_gt_labels_164.npy",
    gt_labels,
)

np.save(
    out_dir / "hovernet_instance_map_164.npy",
    instance_map,
)


np.savez_compressed(
    out_dir / "pannuke_d5_patch_package.npz",
    predicted_labels=predicted_labels,
    active_mask=active_mask,
    ground_truth_labels=gt_labels,
    instance_map=instance_map,
)


metadata = {
    "dataset": "PanNuke Fold 2",
    "image": TARGET_IMAGE,
    "dataset_index": image_index,

    "model": "hovernet_fast-pannuke",

    "valid_prediction_shape": [
        164,
        164,
    ],

    "original_patch_shape": [
        256,
        256,
    ],

    "offset": OFFSET,

    "original_region": {
        "x_start": 46,
        "x_end": 209,
        "y_start": 46,
        "y_end": 209,
    },

    "label_dictionary": {
        str(k): v
        for k, v in ID_TO_CLASS.items()
    },

    "active_pixel_policy": (
        "Only pixels belonging to a HoVer-Net "
        "instance with one of the five biological "
        "classes are active WCSP variables."
    ),

    "background_policy": (
        "HoVer-Net background/untyped pixels remain "
        "inactive and are not interpreted as healthy."
    ),

    "mapped_instances": len(instance_to_class),

    "predicted_active_pixels": int(
        active_mask.sum()
    ),

    "ground_truth_active_pixels": int(
        gt_active.sum()
    ),

    "overlap_pixels": int(
        overlap.sum()
    ),

    "predicted_only_pixels": int(
        predicted_only.sum()
    ),

    "ground_truth_only_pixels": int(
        ground_truth_only.sum()
    ),

    "overlap_class_accuracy": (
        overlap_class_accuracy
    ),

    "prediction_csv": str(prediction_csv),
    "instance_map_source": str(source_npz),
    "ground_truth_source": str(masks_path),
}


with open(
    out_dir / "metadata.json",
    "w",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )


print("\nSaved D5 files:")

for filename in [
    "hovernet_labels_164.npy",
    "active_mask_164.npy",
    "pannuke_gt_labels_164.npy",
    "hovernet_instance_map_164.npy",
    "pannuke_d5_patch_package.npz",
    "metadata.json",
]:
    print(" ", out_dir / filename)


print("\nAakanksha D5 input preparation complete.")