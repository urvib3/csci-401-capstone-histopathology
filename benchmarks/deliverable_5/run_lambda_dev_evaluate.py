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
    / "pannuke_lambda_dev_selection.csv"
)

patch_root = (
    repo
    / "benchmarks"
    / "deliverable_5"
    / "pannuke_multi_patch"
)

evaluate_script = (
    repo
    / "benchmarks"
    / "deliverable_5"
    / "pannuke_abhishek"
    / "evaluate.py"
)

summary_path = (
    repo
    / "benchmarks"
    / "deliverable_5"
    / "pannuke_lambda_dev_evaluation_summary.csv"
)


# ============================================================
# Lambda directories
# ============================================================

LAMBDA_SETTINGS = [
    (1.0, "1"),
    (0.5, "05"),
    (0.25, "025"),
    (0.0, "0"),
]


# ============================================================
# Load development set
# ============================================================

selection = pd.read_csv(selection_csv)

rows = []

successful_runs = 0
failed_runs = 0


# ============================================================
# Evaluate every lambda output
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

        wcsp_dir = (
            source
            / f"wcsp_lambda_{suffix}"
        )

        print("\n" + "-" * 60)
        print(
            f"Evaluating lambda={lam}"
        )

        cmd = [
            sys.executable,
            str(evaluate_script),
            "--source",
            str(source),
            "--wcsp-dir",
            str(wcsp_dir),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        print(result.stdout)

        if result.stderr:
            print(result.stderr)

        evaluation_path = (
            wcsp_dir
            / "evaluation.json"
        )

        # ----------------------------------------------------
        # Failure check
        # ----------------------------------------------------

        if (
            result.returncode != 0
            or not evaluation_path.exists()
        ):

            failed_runs += 1

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
                    False,
            })

            print(
                f"FAILED: {image_name}, "
                f"lambda={lam}"
            )

            continue

        # ----------------------------------------------------
        # Load evaluation JSON
        # ----------------------------------------------------

        with open(evaluation_path) as f:
            e = json.load(f)

        weak = e["weak_fidelity"]
        changes = weak["changes"]

        before_after = e["before_after"]
        errors = e["error_breakdown"]

        successful_runs += 1

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

            # ================================================
            # Changed pixels
            # ================================================

            "changed_pixels":
                weak["changed_pixels"],

            "wrong_to_correct":
                changes[
                    "wrong_to_correct"
                ]["count"],

            "correct_to_wrong":
                changes[
                    "correct_to_wrong"
                ]["count"],

            "wrong_to_different_wrong":
                changes[
                    "wrong_to_different_wrong"
                ]["count"],

            "gt_background_changes":
                changes[
                    "gt_background"
                ]["count"],

            # ================================================
            # Overlap class accuracy
            # ================================================

            "hovernet_overlap_accuracy":
                before_after[
                    "class_agreement_on_overlap"
                ]["hovernet"],

            "wcsp_overlap_accuracy":
                before_after[
                    "class_agreement_on_overlap"
                ]["wcsp_weak"],

            "delta_overlap_accuracy":
                before_after[
                    "class_agreement_on_overlap"
                ]["difference"],

            # ================================================
            # Active-region accuracy
            # ================================================

            "hovernet_active_accuracy":
                before_after[
                    "accuracy_on_active_region"
                ]["hovernet"],

            "wcsp_active_accuracy":
                before_after[
                    "accuracy_on_active_region"
                ]["wcsp_weak"],

            "delta_active_accuracy":
                before_after[
                    "accuracy_on_active_region"
                ]["difference"],

            # ================================================
            # GT nucleus-region accuracy
            # ================================================

            "hovernet_gt_nucleus_accuracy":
                before_after[
                    "accuracy_on_gt_nucleus_region"
                ]["hovernet"],

            "wcsp_gt_nucleus_accuracy":
                before_after[
                    "accuracy_on_gt_nucleus_region"
                ]["wcsp_weak"],

            "delta_gt_nucleus_accuracy":
                before_after[
                    "accuracy_on_gt_nucleus_region"
                ]["difference"],

            # ================================================
            # Full-patch accuracy
            # ================================================

            "hovernet_full_accuracy":
                before_after[
                    "pixel_accuracy_full_patch"
                ]["hovernet"],

            "wcsp_full_accuracy":
                before_after[
                    "pixel_accuracy_full_patch"
                ]["wcsp_weak"],

            "delta_full_accuracy":
                before_after[
                    "pixel_accuracy_full_patch"
                ]["difference"],

            # ================================================
            # mIoU
            # ================================================

            "hovernet_miou":
                before_after[
                    "miou_over_gt_classes"
                ]["hovernet"],

            "wcsp_miou":
                before_after[
                    "miou_over_gt_classes"
                ]["wcsp_weak"],

            "delta_miou":
                before_after[
                    "miou_over_gt_classes"
                ]["difference"],

            # ================================================
            # Error counts
            # ================================================

            "classification_errors_hovernet":
                errors[
                    "classification_errors_hovernet"
                ],

            "classification_errors_wcsp":
                errors[
                    "classification_errors_wcsp"
                ],
        })


# ============================================================
# Save full per-run table
# ============================================================

summary = pd.DataFrame(rows)

summary.to_csv(
    summary_path,
    index=False,
)


print("\n" + "=" * 70)
print("LAMBDA DEV EVALUATION COMPLETE")

print(
    "\nSuccessful runs:",
    successful_runs,
)

print(
    "Failed runs:",
    failed_runs,
)


# ============================================================
# Aggregate by lambda
# ============================================================

successful = summary[
    summary["success"] == True
].copy()


if len(successful) > 0:

    lambda_summary = (
        successful
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

            patches_with_changes=(
                "changed_pixels",
                lambda x: int(
                    (x > 0).sum()
                ),
            ),

            wrong_to_correct=(
                "wrong_to_correct",
                "sum",
            ),

            correct_to_wrong=(
                "correct_to_wrong",
                "sum",
            ),

            wrong_to_different_wrong=(
                "wrong_to_different_wrong",
                "sum",
            ),

            gt_background_changes=(
                "gt_background_changes",
                "sum",
            ),

            mean_delta_overlap_accuracy=(
                "delta_overlap_accuracy",
                "mean",
            ),

            mean_delta_active_accuracy=(
                "delta_active_accuracy",
                "mean",
            ),

            mean_delta_gt_nucleus_accuracy=(
                "delta_gt_nucleus_accuracy",
                "mean",
            ),

            mean_delta_full_accuracy=(
                "delta_full_accuracy",
                "mean",
            ),

            mean_delta_miou=(
                "delta_miou",
                "mean",
            ),
        )
        .reset_index()
    )


    # ========================================================
    # Patch-level outcome counts
    # ========================================================

    improved = (
        successful[
            successful["delta_miou"] > 0
        ]
        .groupby("lambda")
        .size()
        .rename("patches_improved")
    )

    worsened = (
        successful[
            successful["delta_miou"] < 0
        ]
        .groupby("lambda")
        .size()
        .rename("patches_worsened")
    )

    unchanged = (
        successful[
            successful["delta_miou"] == 0
        ]
        .groupby("lambda")
        .size()
        .rename("patches_unchanged")
    )


    lambda_summary = (
        lambda_summary
        .merge(
            improved,
            on="lambda",
            how="left",
        )
        .merge(
            worsened,
            on="lambda",
            how="left",
        )
        .merge(
            unchanged,
            on="lambda",
            how="left",
        )
    )


    for col in [
        "patches_improved",
        "patches_worsened",
        "patches_unchanged",
    ]:

        lambda_summary[col] = (
            lambda_summary[col]
            .fillna(0)
            .astype(int)
        )


    # ========================================================
    # Net correction count
    # ========================================================

    lambda_summary[
        "net_corrections"
    ] = (
        lambda_summary[
            "wrong_to_correct"
        ]
        - lambda_summary[
            "correct_to_wrong"
        ]
    )


    print(
        "\nFINAL LAMBDA COMPARISON:\n"
    )

    print(
        lambda_summary.to_string(
            index=False
        )
    )


    # ========================================================
    # Compact readable summary
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "INTERPRETATION NUMBERS"
    )

    for _, r in lambda_summary.iterrows():

        print(
            f"\nlambda={r['lambda']}"
        )

        print(
            "  changed pixels:",
            int(
                r[
                    "total_changed_pixels"
                ]
            ),
        )

        print(
            "  wrong -> correct:",
            int(
                r[
                    "wrong_to_correct"
                ]
            ),
        )

        print(
            "  correct -> wrong:",
            int(
                r[
                    "correct_to_wrong"
                ]
            ),
        )

        print(
            "  net corrections:",
            int(
                r[
                    "net_corrections"
                ]
            ),
        )

        print(
            "  patches improved:",
            int(
                r[
                    "patches_improved"
                ]
            ),
        )

        print(
            "  patches worsened:",
            int(
                r[
                    "patches_worsened"
                ]
            ),
        )

        print(
            "  mean delta mIoU:",
            r[
                "mean_delta_miou"
            ],
        )


print("\nSaved:")
print(summary_path)