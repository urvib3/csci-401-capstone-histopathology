from pathlib import Path
import json
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

patch_root = (
    repo
    / "benchmarks"
    / "deliverable_5"
    / "pannuke_multi_patch"
)

solver_script = (
    repo
    / "benchmarks"
    / "deliverable_5"
    / "pannuke_abhishek"
    / "run_wcsp.py"
)

summary_path = (
    repo
    / "benchmarks"
    / "deliverable_5"
    / "pannuke_multi_patch_wcsp_summary.csv"
)


# ============================================================
# Load selected patches
# ============================================================

selection = pd.read_csv(selection_csv)

rows = []


# ============================================================
# Run solver
# ============================================================

for _, row in selection.iterrows():

    image_index = int(row["image_index"])

    # Match the real HoVer-Net naming convention
    image_name = f"image_00{image_index}"

    source = patch_root / image_name
    output = source / "wcsp"

    print("=" * 70)
    print(f"Running WCSP: {image_name}")

    if not source.exists():
        print("MISSING SOURCE:", source)

        rows.append({
            "image_index": image_index,
            "image": image_name,
            "success": False,
            "error": "missing prepared patch",
        })

        continue

    cmd = [
        sys.executable,
        str(solver_script),
        "--source",
        str(source),
        "--output",
        str(output),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

    results_json = output / "results.json"

    if (
        result.returncode != 0
        or not results_json.exists()
    ):
        rows.append({
            "image_index": image_index,
            "image": image_name,
            "success": False,
            "error": result.stderr.strip(),
        })

        continue

    with open(results_json) as f:
        report = json.load(f)

    runs = {
        run["setting"]: run
        for run in report["runs"]
    }

    strong = runs["strong_fidelity"]
    weak = runs["weaker_fidelity"]

    rows.append({
        "image_index": image_index,
        "image": image_name,
        "tissue_type": row["tissue_type"],
        "num_gt_classes": row["num_gt_classes"],

        "success": True,

        "variable_count":
            report["variable_count"],

        "edge_count":
            report["edge_count"],

        "initial_pairwise_energy":
            weak["initial_pairwise"],

        "strong_changed_pixels":
            strong["changed_pixels"],

        "weak_changed_pixels":
            weak["changed_pixels"],

        "weak_initial_objective":
            weak["initial_objective"],

        "weak_returned_objective":
            weak["returned_objective"],

        "weak_returned_unary":
            weak["returned_unary"],

        "weak_returned_pairwise":
            weak["returned_pairwise"],

        "weak_runtime_seconds":
            weak["runtime_seconds"],
    })


# ============================================================
# Save summary
# ============================================================

summary = pd.DataFrame(rows)

summary.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# Print analysis
# ============================================================

successful = summary[
    summary["success"] == True
].copy()

print("\n" + "=" * 70)
print("MULTI-PATCH WCSP SUMMARY")

print("\nSuccessful:")
print(len(successful))

if len(successful) > 0:

    zero_pairwise = (
        successful["initial_pairwise_energy"]
        == 0
    ).sum()

    changed = (
        successful["weak_changed_pixels"]
        > 0
    ).sum()

    print(
        "\nPatches with zero initial pairwise energy:",
        zero_pairwise,
        "/",
        len(successful),
    )

    print(
        "Patches where weak WCSP changed pixels:",
        changed,
        "/",
        len(successful),
    )

    print("\nPer-patch results:")

    print(
        successful[
            [
                "image",
                "tissue_type",
                "num_gt_classes",
                "variable_count",
                "edge_count",
                "initial_pairwise_energy",
                "weak_changed_pixels",
                "weak_initial_objective",
                "weak_returned_objective",
            ]
        ].to_string(index=False)
    )


print("\nSaved:")
print(summary_path)