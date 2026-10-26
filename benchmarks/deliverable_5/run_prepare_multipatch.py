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
    / "benchmarks"
    / "deliverable_5"
    / "pannuke_multi_patch_selection.csv"
)

prepare_script = (
    repo
    / "benchmarks"
    / "deliverable_5"
    / "pannuke_preparation"
    / "prepare_full_patch.py"
)

out_dir = (
    repo
    / "benchmarks"
    / "deliverable_5"
    / "pannuke_multi_patch"
)

summary_csv = (
    repo
    / "benchmarks"
    / "deliverable_5"
    / "pannuke_multi_patch_prepare_summary.csv"
)


# ============================================================
# Load selected patches
# ============================================================

df = pd.read_csv(selection_csv)

print("Selected patches:", len(df))
print("Using prepare script:", prepare_script)
print()


# ============================================================
# Run prepare_full_patch.py on each selected patch
# ============================================================

rows = []

for _, row in df.iterrows():

    image_index = int(row["image_index"])
    image_name = row["image"]

    print("=" * 70)
    print(f"Preparing {image_name} (index={image_index})")

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

    canonical_image_name = f"image_00{image_index}"

    patch_dir = out_dir / canonical_image_name
    metadata_path = patch_dir / "metadata.json"

    success = (
        result.returncode == 0
        and metadata_path.exists()
    )

    

    rows.append({
        "image_index": image_index,
        "image": image_name,
        "success": success,
        "return_code": result.returncode,
        "metadata_exists": metadata_path.exists(),
    })

    if success:
        print(f"SUCCESS: {image_name}")
    else:
        print(f"FAILED: {image_name}")

    print("\n--- stdout ---")
    print(result.stdout.strip())

    if result.stderr.strip():
        print("\n--- stderr ---")
        print(result.stderr.strip())

    print()


# ============================================================
# Save run summary
# ============================================================

summary_df = pd.DataFrame(rows)

summary_df.to_csv(
    summary_csv,
    index=False,
)

print("=" * 70)
print("Batch preparation complete.")
print("Successful patches:", int(summary_df["success"].sum()))
print("Failed patches:", int((~summary_df["success"]).sum()))
print("Saved summary:")
print(summary_csv)