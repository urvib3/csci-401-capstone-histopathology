from pathlib import Path
import json

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# 1. Find the existing HoVer-Net results
# ---------------------------------------------------------

repo = Path(__file__).resolve().parents[3]

prediction_csvs = list(
    (repo / "benchmarks" / "pannuke").rglob(
        "hovernet_fold2_predictions.csv"
    )
)

instance_files = sorted(
    (repo / "benchmarks" / "pannuke").rglob(
        "hovernet_instance_maps_*.npz"
    )
)

if not prediction_csvs:
    raise FileNotFoundError(
        "Could not find hovernet_fold2_predictions.csv"
    )

if not instance_files:
    raise FileNotFoundError(
        "Could not find hovernet_instance_maps_*.npz"
    )

prediction_csv = prediction_csvs[0]

print("Prediction CSV:")
print(prediction_csv)

print("\nInstance-map files:")
for f in instance_files:
    print(" ", f)


# ---------------------------------------------------------
# 2. Load the prediction CSV
# ---------------------------------------------------------

pred = pd.read_csv(prediction_csv)

print("\nCSV columns:")
print(pred.columns.tolist())

print("\nFirst five rows:")
print(pred.head())


# ---------------------------------------------------------
# 3. Model-class mapping
# ---------------------------------------------------------

class_to_id = {
    "Neoplastic": 1,
    "Inflammatory": 2,
    "Connective": 3,
    "Dead": 4,
    "Non-neoplastic epithelial": 5,
}

id_to_class = {
    0: "Background / inactive",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Non-neoplastic epithelial",
}


# ---------------------------------------------------------
# Helper: normalize image names
# ---------------------------------------------------------

def normalize_name(name):
    """
    Convert things such as:
        image_000.png
        image_000
    into:
        image_000
    """
    return Path(str(name)).stem


# Find likely CSV image column
possible_image_cols = [
    "image",
    "image_name",
    "filename",
]

image_col = None

for c in possible_image_cols:
    if c in pred.columns:
        image_col = c
        break

if image_col is None:
    raise ValueError(
        f"Could not identify image-name column. "
        f"Columns are: {pred.columns.tolist()}"
    )

pred["_normalized_image"] = pred[image_col].map(normalize_name)


# ---------------------------------------------------------
# Helper: identify class column
# ---------------------------------------------------------

possible_class_cols = [
    "predicted_type",
    "predicted_class",
    "type_name",
]

class_col = None

for c in possible_class_cols:
    if c in pred.columns:
        class_col = c
        break

if class_col is None:
    raise ValueError(
        f"Could not identify predicted class column. "
        f"Columns are: {pred.columns.tolist()}"
    )


# ---------------------------------------------------------
# Helper: identify centroid columns
# ---------------------------------------------------------

if "centroid_x" not in pred.columns or "centroid_y" not in pred.columns:
    raise ValueError(
        "Expected centroid_x and centroid_y columns."
    )


# ---------------------------------------------------------
# 4. Search all instance maps for a useful REAL 3x3 crop
# ---------------------------------------------------------

best_candidate = None


for npz_path in instance_files:

    data = np.load(npz_path, allow_pickle=True)

    print("\nInspecting:", npz_path.name)
    print("NPZ keys:", data.files)

    # Try to identify map and image-name arrays automatically.
    map_key = None
    image_key = None

    for key in data.files:
        arr = np.asarray(data[key])

        # Expected batch of 2D instance maps
        if arr.ndim == 3:
            map_key = key

        # Expected list of image names
        elif arr.ndim == 1:
            image_key = key

    if map_key is None:
        print("Could not identify instance-map key. Skipping.")
        continue

    if image_key is None:
        print("Could not identify image-name key. Skipping.")
        continue

    instance_maps = np.asarray(data[map_key])
    image_names = np.asarray(data[image_key])

    print(
        "Using map key:",
        map_key,
        instance_maps.shape,
    )
    print(
        "Using image-name key:",
        image_key,
        image_names.shape,
    )

    for local_index, raw_name in enumerate(image_names):

        image_name = normalize_name(raw_name)
        inst_map = instance_maps[local_index]

        # We expect HoVer-Net fast's valid output area
        if inst_map.ndim != 2:
            continue

        rows = pred[
            pred["_normalized_image"] == image_name
        ].copy()

        if rows.empty:
            continue

        # -------------------------------------------------
        # 5. Convert INSTANCE IDs -> BIOLOGICAL class IDs
        # -------------------------------------------------

        pixel_class_map = np.zeros(
            inst_map.shape,
            dtype=np.int8,
        )

        instance_to_class = {}

        for _, row in rows.iterrows():

            class_name = str(row[class_col])

            # Ignore type-0/background/untyped predictions
            if class_name not in class_to_id:
                continue

            x = int(round(float(row["centroid_x"])))
            y = int(round(float(row["centroid_y"])))

            # Safety check
            if (
                y < 0
                or y >= inst_map.shape[0]
                or x < 0
                or x >= inst_map.shape[1]
            ):
                continue

            instance_id = int(inst_map[y, x])

            if instance_id == 0:
                # Centroid unexpectedly lands on background.
                continue

            class_id = class_to_id[class_name]

            instance_to_class[instance_id] = class_id

        # Paint every pixel belonging to that nucleus
        for instance_id, class_id in instance_to_class.items():
            pixel_class_map[
                inst_map == instance_id
            ] = class_id

        # -------------------------------------------------
        # 6. Search for informative 3x3 windows
        # -------------------------------------------------

        H, W = pixel_class_map.shape

        for y0 in range(H - 2):
            for x0 in range(W - 2):

                crop = pixel_class_map[
                    y0:y0 + 3,
                    x0:x0 + 3
                ]

                active = crop[crop > 0]

                if len(active) == 0:
                    continue

                classes = np.unique(active)

                candidate = {
                    "image_name": image_name,
                    "npz_path": str(npz_path),
                    "batch_index": int(local_index),
                    "x0": int(x0),
                    "y0": int(y0),
                    "crop": crop.copy(),
                    "instance_crop": inst_map[
                        y0:y0 + 3,
                        x0:x0 + 3
                    ].copy(),
                    "active_pixels": int(len(active)),
                    "num_classes": int(len(classes)),
                    "classes": [
                        int(v) for v in classes
                    ],
                }

                # Ranking:
                # 1. Prefer >=2 biological classes
                # 2. Prefer more active pixels
                score = (
                    int(len(classes) >= 2),
                    len(classes),
                    len(active),
                )

                if (
                    best_candidate is None
                    or score > best_candidate["score"]
                ):
                    candidate["score"] = score
                    best_candidate = candidate


if best_candidate is None:
    raise RuntimeError(
        "Could not find any active 3x3 crop."
    )


# ---------------------------------------------------------
# 7. Save the selected crop
# ---------------------------------------------------------

out_dir = (
    repo
    / "benchmarks"
    / "deliverable_4"
    / "pannuke_task1"
)

out_dir.mkdir(
    parents=True,
    exist_ok=True,
)

crop = best_candidate["crop"]
instance_crop = best_candidate["instance_crop"]

np.save(
    out_dir / "pannuke_task1_label_crop.npy",
    crop,
)

np.save(
    out_dir / "pannuke_task1_instance_crop.npy",
    instance_crop,
)

pd.DataFrame(crop).to_csv(
    out_dir / "pannuke_task1_label_crop.csv",
    index=False,
    header=False,
)


# ---------------------------------------------------------
# 8. Coordinate conversion
# ---------------------------------------------------------

# HoVer-Net valid output is 164x164 within original 256x256.
# (256 - 164) / 2 = 46 pixels.
OFFSET = 46

x0 = best_candidate["x0"]
y0 = best_candidate["y0"]

original_x0 = x0 + OFFSET
original_y0 = y0 + OFFSET


metadata = {
    "source_dataset": "PanNuke Fold 2",
    "source_model": "hovernet_fast-pannuke",
    "source_image": best_candidate["image_name"],

    "hovernet_output_coordinates": {
        "x_start": x0,
        "x_end": x0 + 2,
        "y_start": y0,
        "y_end": y0 + 2,
    },

    "original_256x256_coordinates": {
        "x_start": original_x0,
        "x_end": original_x0 + 2,
        "y_start": original_y0,
        "y_end": original_y0 + 2,
    },

    "label_dictionary": {
        str(k): v
        for k, v in id_to_class.items()
    },

    "active_pixel_definition": (
        "Pixel belongs to a detected nucleus "
        "with one of the five biological classes."
    ),

    "background_handling": (
        "Instance-map value 0 and untyped/background "
        "predictions are inactive and excluded from "
        "the WCSP variable set."
    ),

    "active_pixels": best_candidate["active_pixels"],
    "num_unique_active_classes": best_candidate["num_classes"],
    "active_class_ids": best_candidate["classes"],

    "label_grid": crop.tolist(),
    "instance_id_grid": instance_crop.tolist(),

    "valid_output_shape": [
        int(instance_crop.shape[0])
        if False
        else 164,
        164,
    ],

    "coordinate_offset_to_original_patch": 46,
}

with open(
    out_dir / "pannuke_task1_metadata.json",
    "w",
) as f:
    json.dump(metadata, f, indent=2)


# ---------------------------------------------------------
# 9. Print everything needed for the report
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SELECTED REAL PANNUKE CROP")
print("=" * 60)

print(
    "\nSource image:",
    best_candidate["image_name"],
)

print("\nHoVer-Net output coordinates:")
print(
    f"x = {x0}..{x0 + 2}, "
    f"y = {y0}..{y0 + 2}"
)

print("\nOriginal 256x256 patch coordinates:")
print(
    f"x = {original_x0}..{original_x0 + 2}, "
    f"y = {original_y0}..{original_y0 + 2}"
)

print("\nInstance-ID grid:")
print(instance_crop)

print("\nBiological label grid:")
print(crop)

print("\nLabel dictionary:")
for k, v in id_to_class.items():
    print(f"{k}: {v}")

print(
    "\nActive pixels:",
    best_candidate["active_pixels"],
    "/ 9",
)

print(
    "Unique active biological classes:",
    best_candidate["num_classes"],
)

print("\nSaved outputs:")
for p in [
    out_dir / "pannuke_task1_label_crop.npy",
    out_dir / "pannuke_task1_instance_crop.npy",
    out_dir / "pannuke_task1_label_crop.csv",
    out_dir / "pannuke_task1_metadata.json",
]:
    print(" ", p)

print("\nTask 1 complete.")