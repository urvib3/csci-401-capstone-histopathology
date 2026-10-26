from pathlib import Path
import subprocess
import sys

import pandas as pd


# ============================================================
# Paths
# ============================================================

repo = Path(__file__).resolve().parents[2]

selection_csv = (
    repo
    / "experiments"
    / "pannuke_d5"
    / "pannuke_lambda_dev_selection.csv"
)

prepare_script = (
    repo
    / "experiments"
    / "pannuke_d5"
    / "pannuke_preparation"
    / "prepare_full_patch.py"
)

patch_root = (
    repo
    / "experiments"
    / "pannuke_d5"
    / "pannuke_multi_patch"
)


# ============================================================
# Load development selection
# ============================================================

selection = pd.read_csv(selection_csv)

successful = []
failed = []


# ============================================================
# Prepare every development patch
# ============================================================

for _, row in selection.iterrows():

    image_index = int(row["image_index"])

    # Actual PanNuke/HoVer-Net naming convention used in
    # prepare_full_patch.py.
    image_name = f"image_00{image_index}"

    output_dir = (
        patch_root
        / image_name
    )

    print("\n" + "=" * 70)
    print(
        f"Preparing {image_name} "
        f"({row['tissue_type']})"
    )

    cmd = [
        sys.executable,
        str(prepare_script),
        "--index",
        str(image_index),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

    # --------------------------------------------------------
    # Do not trust return code alone.
    #
    # prepare_full_patch.py may skip a patch cleanly.
    # metadata.json confirms that preparation actually occurred.
    # --------------------------------------------------------

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    required_files = [
        output_dir
        / "hovernet_labels_164.npy",

        output_dir
        / "active_mask_164.npy",

        output_dir
        / "pannuke_gt_labels_164.npy",

        output_dir
        / "hovernet_instance_map_164.npy",

        output_dir
        / "pannuke_d5_patch_package.npz",

        metadata_path,
    ]

    complete = all(
        path.exists()
        for path
        in required_files
    )

    if (
        result.returncode == 0
        and complete
    ):

        successful.append(
            image_name
        )

        print(
            f"SUCCESS: {image_name}"
        )

    else:

        failed.append(
            image_name
        )

        print(
            f"FAILED/SKIPPED: {image_name}"
        )


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("LAMBDA DEV PREPARATION SUMMARY")

print(
    "\nSuccessful:",
    len(successful),
)

print(
    "Failed/skipped:",
    len(failed),
)


if successful:

    print(
        "\nSuccessful patches:"
    )

    for name in successful:
        print(
            " ",
            name,
        )


if failed:

    print(
        "\nFailed/skipped patches:"
    )

    for name in failed:
        print(
            " ",
            name,
        )