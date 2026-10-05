"""Evaluate the pretrained TIAToolbox BCSS U-Net on the BCSS_512 validation split.

Protocol: the network predicts only the central 256x256 of each 512x512 tile
(TIAToolbox crops its upsampled output to half the input size), so the
ground truth is cropped to the same area before scoring. Pixels whose ground
truth is don't-care (outside ROI, exclude) are ignored by bcss.metrics.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from bcss import BCSSDataset, IGNORE_INDEX, coarse_lut  # noqa: E402
from bcss.metrics import (  # noqa: E402
    confusion_matrix, dice_per_class, iou_per_class, mean_iou, pixel_accuracy,
)

MODEL_NAME = "fcn_resnet50_unet-bcss"
OUTPUT_DIR = BASE / "results/evaluation"
TILE_SIZE = 512
CROP_SIZE = 256
OFFSET = (TILE_SIZE - CROP_SIZE) // 2


def load_model(device):
    from tiatoolbox.models.architecture import get_pretrained_model

    model, _ = get_pretrained_model(MODEL_NAME)
    return model.eval().to(device)


@torch.inference_mode()
def predict_crop(model, image, device):
    """Central 256x256 class labels for one 512x512 uint8 RGB tile."""
    x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].float()
    logits = model(model.preproc(x).to(device))  # (1, 5, 256, 256) at half resolution
    probs = F.softmax(logits, dim=1)
    probs = F.interpolate(probs, scale_factor=2, mode="bilinear", align_corners=False)
    probs = probs[..., OFFSET:OFFSET + CROP_SIZE, OFFSET:OFFSET + CROP_SIZE]
    return probs.argmax(dim=1)[0].cpu().numpy()


def per_class_table(cm, names):
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    truth = cm.sum(axis=1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(predicted > 0, tp / predicted, np.nan)
        recall = np.where(truth > 0, tp / truth, np.nan)
    return pd.DataFrame({
        "class": names,
        "truth_pixels": truth.astype(int),
        "predicted_pixels": predicted.astype(int),
        "precision": precision,
        "recall": recall,
        "iou": iou_per_class(cm),
        "dice": dice_per_class(cm),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="BCSS root (default: $BCSS_ROOT or data/bcss)")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N tiles (0 = all)")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    lut, names = coarse_lut()
    data = BCSSDataset(args.root, variant="512", split="val", lut=lut)
    count = len(data) if not args.limit else min(args.limit, len(data))
    model = load_model(args.device)

    confusion = np.zeros((len(names), len(names)), dtype=np.int64)
    start = time.perf_counter()
    for index in range(count):
        sample = data[index]
        prediction = predict_crop(model, sample.image, args.device)
        target = sample.mask[OFFSET:OFFSET + CROP_SIZE, OFFSET:OFFSET + CROP_SIZE]
        confusion += confusion_matrix(prediction, target, len(names))
    runtime = time.perf_counter() - start

    overall = pd.DataFrame([
        ("Tiles evaluated", count),
        ("Pixel accuracy", pixel_accuracy(confusion)),
        ("Mean IoU", mean_iou(confusion)),
        ("Mean Dice", float(np.nanmean(dice_per_class(confusion)))),
        ("Seconds per tile", runtime / max(count, 1)),
        ("Device", args.device),
    ], columns=["metric", "value"])
    classes = per_class_table(confusion, names)
    matrix = pd.DataFrame(confusion, index=pd.Index(names, name="ground_truth"), columns=names)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overall.to_csv(OUTPUT_DIR / "bcss512_val_overall_metrics.csv", index=False)
    classes.to_csv(OUTPUT_DIR / "bcss512_val_class_metrics.csv", index=False)
    matrix.to_csv(OUTPUT_DIR / "bcss512_val_confusion_matrix.csv")
    pd.set_option("display.width", 200)
    print(overall.to_string(index=False))
    print(classes.round(3).to_string(index=False))
    print(matrix.to_string())


if __name__ == "__main__":
    main()
