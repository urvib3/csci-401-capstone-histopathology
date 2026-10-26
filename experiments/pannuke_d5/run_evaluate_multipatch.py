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
    / "pannuke_multi_patch_selection.csv"
)

patch_root = (
    repo
    / "experiments"
    / "pannuke_d5"
    / "pannuke_multi_patch"
)

evaluate_script = (
    repo
    / "experiments"
    / "pannuke_d5"
    / "pannuke_refinement"
    / "evaluate.py"
)

summary_path = (
    repo
    / "experiments"
    / "pannuke_d5"
    / "pannuke_multi_patch_evaluation_summary.csv"
)


# ============================================================
# Load selected patches
# ============================================================

selection = pd.read_csv(selection_csv)

rows = []


# ============================================================
# Evaluate each selected patch
# ============================================================

for _, row in selection.iterrows():

    image_index = int(row["image_index"])
    image_name = f"image_00{image_index}"

    source = patch_root / image_name
    wcsp_dir = source / "wcsp"

    print("=" * 70)
    print(f"Evaluating {image_name}")

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

    # --------------------------------------------------------
    # Handle failures
    # --------------------------------------------------------

    if (
        result.returncode != 0
        or not evaluation_path.exists()
    ):

        rows.append({
            "image_index": image_index,
            "image": image_name,
            "tissue_type": row["tissue_type"],
            "success": False,
        })

        continue

    # --------------------------------------------------------
    # Load evaluation result
    # --------------------------------------------------------

    with open(evaluation_path) as f:
        e = json.load(f)

    weak = e["weak_fidelity"]
    changes = weak["changes"]

    before_after = e["before_after"]
    errors = e["error_breakdown"]

    # --------------------------------------------------------
    # Save one summary row
    # --------------------------------------------------------

    rows.append({
        "image_index": image_index,
        "image": image_name,
        "tissue_type": row["tissue_type"],
        "success": True,

        # ====================================================
        # WCSP changes
        # ====================================================

        "weak_changed_pixels":
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

        # ====================================================
        # Class agreement on overlap
        # ====================================================

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

        # ====================================================
        # Accuracy on active HoVer-Net region
        # ====================================================

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

        # ====================================================
        # Accuracy on GT nucleus region
        # ====================================================

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

        # ====================================================
        # Full-patch pixel accuracy
        # ====================================================

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

        # ====================================================
        # mIoU
        # ====================================================

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

        # ====================================================
        # Error decomposition
        # ====================================================

        "missed_gt_nucleus_pixels":
            errors[
                "detection_missed_gt_nucleus_pixels"
            ],

        "false_positive_pixels":
            errors[
                "detection_false_positive_pixels"
            ],

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
# Build summary DataFrame
# ============================================================

summary = pd.DataFrame(rows)


# ============================================================
# Save summary CSV
# ============================================================

summary.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# Analyze successful evaluations
# ============================================================

successful = summary[
    summary["success"] == True
].copy()


print("\n" + "=" * 70)
print("MULTI-PATCH EVALUATION SUMMARY")

print(
    "\nSuccessful:",
    len(successful),
)


if len(successful) > 0:

    # --------------------------------------------------------
    # Total changes
    # --------------------------------------------------------

    total_changed = int(
        successful[
            "weak_changed_pixels"
        ].sum()
    )

    total_corrected = int(
        successful[
            "wrong_to_correct"
        ].sum()
    )

    total_new_errors = int(
        successful[
            "correct_to_wrong"
        ].sum()
    )

    total_wrong_to_wrong = int(
        successful[
            "wrong_to_different_wrong"
        ].sum()
    )

    total_background_changes = int(
        successful[
            "gt_background_changes"
        ].sum()
    )


    print(
        "\nTotal WCSP-changed pixels:",
        total_changed,
    )

    print(
        "Total wrong -> correct:",
        total_corrected,
    )

    print(
        "Total correct -> wrong:",
        total_new_errors,
    )

    print(
        "Total wrong -> different wrong:",
        total_wrong_to_wrong,
    )

    print(
        "Total changes on GT background:",
        total_background_changes,
    )


    # --------------------------------------------------------
    # Mean metric changes
    # --------------------------------------------------------

    print(
        "\nMean delta overlap accuracy:",
        successful[
            "delta_overlap_accuracy"
        ].mean(),
    )

    print(
        "Mean delta active accuracy:",
        successful[
            "delta_active_accuracy"
        ].mean(),
    )

    print(
        "Mean delta GT-nucleus accuracy:",
        successful[
            "delta_gt_nucleus_accuracy"
        ].mean(),
    )

    print(
        "Mean delta full-patch accuracy:",
        successful[
            "delta_full_accuracy"
        ].mean(),
    )

    print(
        "Mean delta mIoU:",
        successful[
            "delta_miou"
        ].mean(),
    )


    # --------------------------------------------------------
    # Patch-level outcome counts
    # --------------------------------------------------------

    improved = int(
        (
            successful[
                "delta_miou"
            ] > 0
        ).sum()
    )

    worsened = int(
        (
            successful[
                "delta_miou"
            ] < 0
        ).sum()
    )

    unchanged = int(
        (
            successful[
                "delta_miou"
            ] == 0
        ).sum()
    )


    print("\nPatches by mIoU:")

    print(
        "Improved:",
        improved,
    )

    print(
        "Worsened:",
        worsened,
    )

    print(
        "Unchanged:",
        unchanged,
    )


    # --------------------------------------------------------
    # Error totals
    # --------------------------------------------------------

    print(
        "\nTotal missed GT nucleus pixels:",
        int(
            successful[
                "missed_gt_nucleus_pixels"
            ].sum()
        ),
    )

    print(
        "Total false-positive pixels:",
        int(
            successful[
                "false_positive_pixels"
            ].sum()
        ),
    )

    print(
        "Total HoVer-Net classification errors:",
        int(
            successful[
                "classification_errors_hovernet"
            ].sum()
        ),
    )

    print(
        "Total WCSP classification errors:",
        int(
            successful[
                "classification_errors_wcsp"
            ].sum()
        ),
    )


    # --------------------------------------------------------
    # Per-patch summary
    # --------------------------------------------------------

    print("\nPer-patch changes:")

    print(
        successful[
            [
                "image",
                "tissue_type",
                "weak_changed_pixels",
                "wrong_to_correct",
                "correct_to_wrong",
                "wrong_to_different_wrong",
                "delta_overlap_accuracy",
                "delta_miou",
            ]
        ].to_string(
            index=False
        )
    )


print("\nSaved:")
print(summary_path)