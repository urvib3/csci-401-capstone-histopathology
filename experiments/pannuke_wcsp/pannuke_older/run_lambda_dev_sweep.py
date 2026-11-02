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
    / "experiments"
    / "pannuke_d5"
    / "pannuke_lambda_dev_selection.csv"
)

patch_root = (
    repo
    / "experiments"
    / "pannuke_d5"
    / "pannuke_multi_patch"
)

wcsp_script = (
    repo
    / "experiments"
    / "pannuke_d5"
    / "pannuke_refinement"
    / "run_wcsp.py"
)

summary_path = (
    repo
    / "experiments"
    / "pannuke_d5"
    / "pannuke_lambda_dev_wcsp_summary.csv"
)


# ============================================================
# Lambda settings
# ============================================================

LAMBDA_SETTINGS = [
    (1.0, "1"),
    (0.5, "05"),
    (0.25, "025"),
    (0.0, "0"),
]


# ============================================================
# Load development patches
# ============================================================

selection = pd.read_csv(selection_csv)

rows = []

successful = 0
failed = 0


# ============================================================
# Run sweep
# ============================================================

for _, row in selection.iterrows():

    image_index = int(row["image_index"])
    image_name = f"image_00{image_index}"

    source = (
        patch_root
        / image_name
    )

    print("\n" + "=" * 70)
    print(
        f"PATCH: {image_name} "
        f"({row['tissue_type']})"
    )

    for lam, suffix in LAMBDA_SETTINGS:

        output_dir = (
            source
            / f"wcsp_lambda_{suffix}"
        )

        print("\n" + "-" * 60)
        print(
            f"Running lambda={lam}"
        )

        cmd = [
            sys.executable,
            str(wcsp_script),
            "--source",
            str(source),
            "--output",
            str(output_dir),
            "--weak-lambda",
            str(lam),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        print(result.stdout)

        if result.stderr:
            print(result.stderr)

        results_path = (
            output_dir
            / "results.json"
        )

        weak_labels_path = (
            output_dir
            / "wcsp_weak_labels.npy"
        )

        # ----------------------------------------------------
        # Verify run completed
        # ----------------------------------------------------

        if (
            result.returncode != 0
            or not results_path.exists()
            or not weak_labels_path.exists()
        ):

            failed += 1

            rows.append({
                "image_index": image_index,
                "image": image_name,
                "tissue_type": row["tissue_type"],
                "lambda": lam,
                "success": False,
            })

            print(
                f"FAILED: {image_name}, "
                f"lambda={lam}"
            )

            continue

        # ----------------------------------------------------
        # Read WCSP report
        # ----------------------------------------------------

        with open(results_path) as f:
            report = json.load(f)

        weak_run = None

        for run in report["runs"]:

            if (
                run["setting"]
                == "weaker_fidelity"
            ):
                weak_run = run
                break

        if weak_run is None:

            failed += 1

            rows.append({
                "image_index": image_index,
                "image": image_name,
                "tissue_type": row["tissue_type"],
                "lambda": lam,
                "success": False,
            })

            print(
                "FAILED: weaker_fidelity "
                "result missing"
            )

            continue

        successful += 1

        rows.append({
            "image_index":
                image_index,

            "image":
                image_name,

            "tissue_type":
                row["tissue_type"],

            "lambda":
                lam,

            "success":
                True,

            "variable_count":
                report["variable_count"],

            "edge_count":
                report["edge_count"],

            "initial_objective":
                weak_run[
                    "initial_objective"
                ],

            "initial_unary":
                weak_run[
                    "initial_unary"
                ],

            "initial_pairwise":
                weak_run[
                    "initial_pairwise"
                ],

            "returned_objective":
                weak_run[
                    "returned_objective"
                ],

            "returned_unary":
                weak_run[
                    "returned_unary"
                ],

            "returned_pairwise":
                weak_run[
                    "returned_pairwise"
                ],

            "changed_pixels":
                weak_run[
                    "changed_pixels"
                ],

            "optimization_rounds":
                weak_run[
                    "optimization_rounds"
                ],

            "runtime_seconds":
                weak_run[
                    "runtime_seconds"
                ],
        })

        print(
            f"SUCCESS: {image_name}, "
            f"lambda={lam}, "
            f"changed="
            f"{weak_run['changed_pixels']}"
        )


# ============================================================
# Build summary
# ============================================================

summary = pd.DataFrame(rows)

summary.to_csv(
    summary_path,
    index=False,
)


print("\n" + "=" * 70)
print("LAMBDA SWEEP COMPLETE")

print(
    "\nSuccessful runs:",
    successful,
)

print(
    "Failed runs:",
    failed,
)


# ============================================================
# Lambda-level quick summary
# ============================================================

successful_rows = summary[
    summary["success"] == True
].copy()


if len(successful_rows) > 0:

    print("\nChanges by lambda:\n")

    lambda_summary = (
        successful_rows
        .groupby("lambda")
        .agg(
            patches=(
                "image",
                "count",
            ),
            total_changed_pixels=(
                "changed_pixels",
                "sum",
            ),
            mean_changed_pixels=(
                "changed_pixels",
                "mean",
            ),
            patches_with_changes=(
                "changed_pixels",
                lambda x: int(
                    (x > 0).sum()
                ),
            ),
            mean_initial_pairwise=(
                "initial_pairwise",
                "mean",
            ),
            mean_returned_pairwise=(
                "returned_pairwise",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        lambda_summary.to_string(
            index=False
        )
    )


print("\nSaved:")
print(summary_path)