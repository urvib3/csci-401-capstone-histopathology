"""Ground-truth evaluation of the WCSP outputs from run_wcsp.py.

Run after run_wcsp.py; this is the only script that reads the ground truth.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "pannuke_aakanksha"
CLASSES = {1: "Neoplastic", 2: "Inflammatory", 3: "Connective", 4: "Dead",
           5: "Non-neoplastic epithelial"}


def scores(pred, gt):
    """Agreement and IoU in the three scopes the handoff asks to keep apart."""
    pred_nuc, gt_nuc = pred > 0, gt > 0
    overlap = pred_nuc & gt_nuc
    iou = {}
    for k, name in CLASSES.items():
        union = int(((pred == k) | (gt == k)).sum())
        # Classes absent from both have no IoU; present only in pred give 0.
        iou[name] = None if union == 0 else int(((pred == k) & (gt == k)).sum()) / union
    present = [v for k, v in zip(CLASSES, iou.values()) if (gt == k).any()]
    return {
        "class_agreement_on_overlap": float((pred == gt)[overlap].mean()),
        "accuracy_on_active_region": float((pred == gt)[pred_nuc].mean()),
        "accuracy_on_gt_nucleus_region": float((pred == gt)[gt_nuc].mean()),
        "pixel_accuracy_full_patch": float((pred == gt).mean()),
        "class_iou": iou,
        "miou_over_gt_classes": float(np.mean(present)),
        "confusion_matrix_rows_gt_cols_pred": confusion(gt, pred).tolist(),
    }


def confusion(gt, pred):
    out = np.zeros((6, 6), dtype=int)
    np.add.at(out, (gt.ravel(), pred.ravel()), 1)
    return out


def classify_changes(initial, refined, gt, active):
    moved = active & (initial != refined)
    nucleus = gt > 0
    was, now = initial == gt, refined == gt
    buckets = {
        "wrong_to_correct": moved & nucleus & ~was & now,
        "correct_to_wrong": moved & nucleus & was & ~now,
        "wrong_to_different_wrong": moved & nucleus & ~was & ~now,
        "gt_background": moved & ~nucleus,
    }
    detail = {}
    for name, mask in buckets.items():
        pairs = {}
        for a, b in zip(initial[mask], refined[mask]):
            key = f"{CLASSES[a]}->{CLASSES[b]}"
            pairs[key] = pairs.get(key, 0) + 1
        detail[name] = {"count": int(mask.sum()), "transitions": pairs}
    return int(moved.sum()), detail


def main():
    initial = np.load(SOURCE / "hovernet_labels_164.npy")
    active = np.load(SOURCE / "active_mask_164.npy").astype(bool)
    gt = np.load(SOURCE / "pannuke_gt_labels_164.npy")
    strong = np.load(HERE / "wcsp_strong_labels.npy")
    weak = np.load(HERE / "wcsp_weak_labels.npy")
    wcsp = json.loads((HERE / "results.json").read_text())
    runs = {r["setting"]: r for r in wcsp["runs"]}
    if wcsp.get("ground_truth_used") is not False:
        raise ValueError("results.json does not certify a ground-truth-free run")
    for name, arr in (("strong", strong), ("weak", weak)):
        if arr.shape != initial.shape or (arr[~active] != 0).any():
            raise ValueError(f"{name} output must be 164x164 with inactive pixels 0")

    s = runs["strong_fidelity"]
    strong_check = {
        "identical_to_hovernet": bool(np.array_equal(strong, initial)),
        "changed_pixels": int((strong != initial)[active].sum()),
        "lambda": s["lambda"],
        "initial_objective": s["initial_objective"],
        "returned_objective": s["returned_objective"],
        "runtime_seconds": s["runtime_seconds"],
        "status": s["status"],
    }

    changed, buckets = classify_changes(initial, weak, gt, active)
    before, after = scores(initial, gt), scores(weak, gt)
    keys = ["class_agreement_on_overlap", "accuracy_on_active_region",
            "accuracy_on_gt_nucleus_region", "pixel_accuracy_full_patch",
            "miou_over_gt_classes"]
    comparison = {k: {"hovernet": before[k], "wcsp_weak": after[k],
                      "difference": after[k] - before[k]} for k in keys}

    pred_nuc, gt_nuc = initial > 0, gt > 0
    overlap = pred_nuc & gt_nuc
    errors = {
        "detection_missed_gt_nucleus_pixels": int((gt_nuc & ~pred_nuc).sum()),
        "detection_false_positive_pixels": int((pred_nuc & ~gt_nuc).sum()),
        "classification_errors_hovernet": int((overlap & (initial != gt)).sum()),
        "classification_errors_wcsp": int((overlap & (weak != gt)).sum()),
        "wcsp_corrections": buckets["wrong_to_correct"]["count"],
        "wcsp_new_errors": buckets["correct_to_wrong"]["count"],
        "overlap_pixels": int(overlap.sum()),
        "note": ("detection errors are outside the WCSP's reach: missed pixels are "
                 "never variables and the WCSP cannot switch a pixel to background"),
    }

    report = {
        "source_image": "image_00748",
        "gt_classes_present": {CLASSES[int(k)]: int((gt == k).sum())
                               for k in np.unique(gt) if k > 0},
        "strong_fidelity_check": strong_check,
        "weak_fidelity": {"lambda": runs["weaker_fidelity"]["lambda"],
                          "changed_pixels": changed, "changes": buckets},
        "before_after": comparison,
        "hovernet_full": before,
        "wcsp_weak_full": after,
        "error_breakdown": errors,
        "scopes": {
            "overlap": "pixels that are nucleus in both HoVer-Net and GT",
            "active_region": "HoVer-Net active pixels; GT background counts as wrong",
            "gt_nucleus_region": "GT nucleus pixels; missed pixels count as wrong",
            "full_patch": "all 164x164 pixels, background included",
        },
    }
    (HERE / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"strong: identical={strong_check['identical_to_hovernet']}, "
          f"changed={strong_check['changed_pixels']}")
    print(f"weak: changed={changed}")
    for name, b in buckets.items():
        print(f"  {name}: {b['count']} {b['transitions']}")
    for k, v in comparison.items():
        print(f"{k}: {v['hovernet']:.4f} -> {v['wcsp_weak']:.4f} ({v['difference']:+.4f})")
    print({k: v for k, v in errors.items() if k != "note"})


if __name__ == "__main__":
    main()
