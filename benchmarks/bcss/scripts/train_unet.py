# Supervised baseline: U-Net with an ImageNet ResNet-34 encoder on BCSS_512,
# 5 coarse classes.
#
#   .venv/bin/python benchmarks/bcss/scripts/train_unet.py --epochs 20
#
# Setup (see benchmarks/bcss/README.md for the rationale):
#   - Tiles downsampled 512 -> 256 (2x less resolution, 4x faster, more
#     tissue context per pixel).
#   - Fit on train minus the dev ROIs; dev picks the best epoch. The official
#     val split (held-out hospitals) is only scored once, at the end.
#   - Cross-entropy with sqrt-inverse-frequency class weights, ignoring
#     don't-care pixels.
#   - Augmentation: flips, 90-degree rotations, per-tile colour jitter.
#
# Outputs:
#   data/outputs/unet_resnet34.pt              best checkpoint (by dev mIoU)
#   data/outputs/unet_resnet34_last.pt         resume state (delete to retrain from scratch)
#   data/outputs/unet_val_probs32.npz          val softmax pooled to 32x32
#                                              (16 px cells at 512), for WCSP unaries
#   benchmarks/bcss/results/unet_train_log.csv per-epoch loss / dev mIoU
#   benchmarks/bcss/results/unet_baseline.json final val metrics

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, OUTPUT_DIR, RESULTS_DIR,
                    BCSSDataset, LUT, REPO, class_report, confusion_matrix,
                    dev_split, load_split)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_input(images_uint8, device):
    x = torch.from_numpy(images_uint8).to(device).permute(0, 3, 1, 2).float() / 255
    return (x - MEAN.to(device)) / STD.to(device)


def augment(x, y):
    """Random flips / rot90 per batch, colour jitter per tile (x in [0,1])."""
    k = int(torch.randint(4, ()))
    x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (1, 2))
    if torch.rand(()) < 0.5:
        x, y = x.flip(3), y.flip(2)
    n = x.shape[0]
    scale = 1 + 0.1 * (2 * torch.rand(n, 3, 1, 1, device=x.device) - 1)
    shift = 0.05 * (2 * torch.rand(n, 3, 1, 1, device=x.device) - 1)
    return (x * scale + shift).clamp(0, 1), y


@torch.no_grad()
def predict(model, images, device, batch=32):
    """Softmax probabilities (N, C, H, W) as float16 numpy, at input resolution."""
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        logits = model(to_input(images[i:i + batch], device))
        # softmax/half on CPU: on MPS they crash when the last batch is smaller.
        out.append(logits.cpu().softmax(1).half().numpy())
    return np.concatenate(out)


def evaluate(probs, masks):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    for p, m in zip(probs, masks):
        cm += confusion_matrix(p.argmax(0), m, NUM_CLASSES)
    return cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--limit-steps", type=int, default=0, help="debug: steps per epoch")
    args = parser.parse_args()

    import segmentation_models_pytorch as smp

    torch.manual_seed(0)
    np.random.seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    images, masks, names = load_split("train", args.size)
    dev = dev_split(names)
    tr_x, tr_y = images[~dev], masks[~dev]
    dv_x, dv_y = images[dev], masks[dev]
    print(f"fit {len(tr_x)} tiles, dev {len(dv_x)} tiles, device {device}", flush=True)

    counts = np.bincount(tr_y.ravel(), minlength=256)[:NUM_CLASSES].astype(np.float64)
    weights = 1 / np.sqrt(counts / counts.sum())
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
    print("class weights", dict(zip(CLASS_NAMES, weights.cpu().numpy().round(2))), flush=True)

    model = smp.Unet("resnet34", encoder_weights="imagenet", classes=NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = len(tr_x) // args.batch
    if args.limit_steps:
        steps = min(steps, args.limit_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps,
                                                pct_start=0.1)
    ckpt = os.path.join(OUTPUT_DIR, "unet_resnet34.pt")
    # Full training state after every epoch, so an interrupted run resumes.
    last = os.path.join(OUTPUT_DIR, "unet_resnet34_last.pt")
    log, best, first = [], -1.0, 0
    if os.path.exists(last):
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        log, best, first = state["log"], state["best"], state["epoch"]
        np.random.set_state(state["np_rng"])
        torch.set_rng_state(state["torch_rng"])
        print(f"resuming after epoch {first}", flush=True)
    for epoch in range(first, args.epochs):
        model.train()
        order = np.random.permutation(len(tr_x))
        total, start = 0.0, time.time()
        for s in range(steps):
            idx = np.sort(order[s * args.batch:(s + 1) * args.batch])
            x = torch.from_numpy(tr_x[idx]).to(device).permute(0, 3, 1, 2).float() / 255
            y = torch.from_numpy(tr_y[idx].astype(np.int64)).to(device)
            x, y = augment(x, y)
            x = (x - MEAN.to(device)) / STD.to(device)
            loss = F.cross_entropy(model(x), y, weight=weights, ignore_index=IGNORE_INDEX)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        dev_report = class_report(evaluate(predict(model, dv_x, device), dv_y))
        row = {"epoch": epoch + 1, "loss": total / steps, "dev_mIoU": dev_report["mIoU"],
               "dev_pixel_acc": dev_report["pixel_acc"], "seconds": time.time() - start}
        log.append(row)
        print(json.dumps({k: round(v, 4) for k, v in row.items()}), flush=True)
        if dev_report["mIoU"] > best:
            best = dev_report["mIoU"]
            torch.save(model.state_dict(), ckpt)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "log": log, "best": best, "epoch": epoch + 1,
                    "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()}, last)

    import pandas as pd
    pd.DataFrame(log).to_csv(os.path.join(RESULTS_DIR, "unet_train_log.csv"), index=False)

    # Final scoring on the held-out-hospital val split, best dev checkpoint.
    model.load_state_dict(torch.load(ckpt, map_location=device))
    va_x, va_y, va_names = load_split("val", args.size)
    start = time.time()
    probs = predict(model, va_x, device)
    infer_seconds = time.time() - start
    result = {"val@256": class_report(evaluate(probs, va_y))}

    # Same predictions upsampled and scored against the native 512 masks.
    root = os.environ.get("BCSS_ROOT", os.path.join(REPO, "data", "bcss"))
    full = BCSSDataset(root=root, variant="512", split="val", lut=LUT)
    cm512 = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    for i in range(len(full)):
        assert full.tiles[i].name == va_names[i]
        up = F.interpolate(torch.from_numpy(probs[i:i + 1].astype(np.float32)), scale_factor=2,
                           mode="bilinear", align_corners=False)[0].argmax(0).numpy()
        cm512 += confusion_matrix(up, full[i].mask, NUM_CLASSES)
    result["val@512"] = class_report(cm512)
    result["confusion_matrix_val@512"] = cm512.tolist()
    result["setup"] = {**vars(args), "fit_tiles": int(len(tr_x)), "dev_tiles": int(len(dv_x)),
                       "best_dev_mIoU": best, "val_inference_seconds": infer_seconds,
                       "device": device, "classes": CLASS_NAMES}
    with open(os.path.join(RESULTS_DIR, "unet_baseline.json"), "w") as f:
        json.dump(result, f, indent=2)

    # Pool to the 32x32 CellGrid used by the WCSP primitives (16 px cells at 512).
    pooled = F.avg_pool2d(torch.from_numpy(probs.astype(np.float32)), args.size // 32).numpy()
    np.savez_compressed(os.path.join(OUTPUT_DIR, "unet_val_probs32.npz"),
                        probs=pooled.astype(np.float16), names=va_names)
    print(json.dumps({k: round(v, 4) for k, v in result["val@512"].items()}, indent=1))


if __name__ == "__main__":
    main()
