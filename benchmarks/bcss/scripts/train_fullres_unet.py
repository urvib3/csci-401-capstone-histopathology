# Train the full-resolution level of the TIAToolbox BCSS U-Net
# (wcsp/classifiers/fullres_unet.py): one new encoder level on the raw image
# and one new decoder level, so the network predicts every pixel instead of
# a half-resolution map upsampled x2.
#
#   .venv/bin/python benchmarks/bcss/scripts/train_fullres_unet.py --epochs 5
#
# Setup:
#   - Same splits as the wcsp package: fit = train tiles outside the dev ROIs,
#     dev (every 8th train ROI) picks the best epoch. Val is never touched.
#   - The pretrained U-Net is frozen by default (--finetune also trains it, at
#     --base-lr); the new head starts as an exact copy of the pretrained
#     output, so epoch 0 is the baseline.
#   - Cross-entropy on the central 256 x 256 (the part the WCSP uses; the tile
#     border lacks context), ignoring don't-care pixels. No class weights, so
#     changes come from resolution rather than rebalancing.
#   - Augmentation: flips, 90-degree rotations, per-tile colour jitter.
#
# Outputs (in --out, default data/outputs/fullres_unet/):
#   best.pt     best dev checkpoint, for configs/bcss_unet_fullres.yaml
#   last.pt     resume state (delete to retrain from scratch)
#   log.csv     per-epoch loss and dev scores

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from bcss.classes import IGNORE_INDEX
from bcss.metrics import confusion_matrix, iou_per_class
from wcsp.classifiers.fullres_unet import FullResUNet, MODEL_NAME
from wcsp.data.bcss import BCSS, WINDOW

NUM_CLASSES = 5


class Tiles(Dataset):
    """(image uint8 HWC, BCSS-order mask) for one wcsp BCSS split."""

    def __init__(self, split, root):
        self.split = BCSS(split, root=root)

    def __len__(self):
        return len(self.split)

    def __getitem__(self, i):
        tile = self.split.data[self.split.positions[i]]
        return torch.from_numpy(tile.image), torch.from_numpy(tile.mask.astype(np.int64))


def to_input(images, device):
    return images.to(device).permute(0, 3, 1, 2).float()   # 0-255, as the pretrained model expects


def augment(x, y):
    """Random flips / rot90 per batch, colour jitter per tile (x in 0-255)."""
    k = int(torch.randint(4, ()))
    x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (1, 2))
    if torch.rand(()) < 0.5:
        x, y = x.flip(3), y.flip(2)
    n = x.shape[0]
    scale = 1 + 0.1 * (2 * torch.rand(n, 3, 1, 1, device=x.device) - 1)
    shift = 12.75 * (2 * torch.rand(n, 3, 1, 1, device=x.device) - 1)
    return (x * scale + shift).clamp(0, 255), y


@torch.no_grad()
def evaluate(model, loader, device):
    """Dev confusion matrix on the central 256 x 256."""
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    for images, masks in loader:
        pred = model(to_input(images, device))[(...,) + WINDOW].argmax(1).cpu().numpy()
        for p, m in zip(pred, masks[(...,) + WINDOW].numpy()):
            cm += confusion_matrix(p, m, NUM_CLASSES)
    return cm


def report(cm):
    iou = iou_per_class(cm)
    return {"dev_mIoU": float(np.nanmean(iou)), "dev_pixel_acc": float(np.trace(cm) / cm.sum()),
            **{f"dev_IoU_{c}": float(v) for c, v in enumerate(iou)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--finetune", action="store_true", help="also train the pretrained U-Net")
    parser.add_argument("--base-lr", type=float, default=1e-5)
    parser.add_argument("--root", default=str(REPO / "data/bcss"))
    parser.add_argument("--weights", default=None, help="local TIAToolbox weights")
    parser.add_argument("--out", default=str(REPO / "data/outputs/fullres_unet"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--limit-steps", type=int, default=0, help="debug: steps per epoch")
    parser.add_argument("--limit-dev", type=int, default=0, help="debug: dev tiles scored")
    args = parser.parse_args()

    from tiatoolbox.models.architecture import get_pretrained_model

    torch.manual_seed(0)
    np.random.seed(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device

    fit, dev = Tiles("fit", args.root), Tiles("dev", args.root)
    if args.limit_dev:
        dev = torch.utils.data.Subset(dev, np.linspace(0, len(dev) - 1, args.limit_dev).round().astype(int))
    fit_loader = DataLoader(fit, batch_size=args.batch, shuffle=True, drop_last=True,
                            num_workers=args.workers, persistent_workers=args.workers > 0)
    dev_loader = DataLoader(dev, batch_size=args.batch, num_workers=args.workers)
    print(f"fit {len(fit)} tiles, dev {len(dev)} tiles, device {device}", flush=True)

    base, _ = get_pretrained_model(MODEL_NAME, pretrained_weights=args.weights)
    model = FullResUNet(base, width=args.width, freeze_base=not args.finetune).to(device)
    head = [p for n, p in model.named_parameters() if not n.startswith("base.")]
    groups = [{"params": head, "lr": args.lr}]
    if args.finetune:
        groups.append({"params": model.base.parameters(), "lr": args.base_lr})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    steps = len(fit_loader) if not args.limit_steps else min(len(fit_loader), args.limit_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups],
                                                total_steps=args.epochs * steps, pct_start=0.1)

    def checkpoint(path, extra=None):
        state = model.state_dict() if args.finetune else model.head_state_dict()
        torch.save({"model": state, "width": args.width, "args": vars(args), **(extra or {})}, path)

    log, best, first = [], -1.0, 0
    last = out / "last.pt"
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        log, best, first = state["log"], state["best"], state["epoch"]
        torch.set_rng_state(state["torch_rng"])
        print(f"resuming after epoch {first}", flush=True)
    else:
        # Epoch 0: the untrained head reproduces the pretrained U-Net.
        row = {"epoch": 0, "loss": float("nan"), **report(evaluate(model, dev_loader, device)), "seconds": 0.0}
        log.append(row)
        print(json.dumps({k: round(v, 4) for k, v in row.items()}), flush=True)

    window = (...,) + WINDOW
    for epoch in range(first, args.epochs):
        model.train()
        total, start = 0.0, time.time()
        for step, (images, masks) in enumerate(fit_loader):
            if step == steps:
                break
            x, y = augment(to_input(images, device), masks.to(device))
            loss = F.cross_entropy(model(x)[window], y[window], ignore_index=IGNORE_INDEX)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
            if step % 50 == 0:
                print(f"  epoch {epoch + 1} step {step}/{steps} loss {loss.item():.4f}", flush=True)
        row = {"epoch": epoch + 1, "loss": total / steps, **report(evaluate(model, dev_loader, device)),
               "seconds": time.time() - start}
        log.append(row)
        print(json.dumps({k: round(v, 4) for k, v in row.items()}), flush=True)
        if row["dev_mIoU"] > best:
            best = row["dev_mIoU"]
            checkpoint(out / "best.pt", {"epoch": epoch + 1, "dev": row})
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "log": log, "best": best, "epoch": epoch + 1, "torch_rng": torch.get_rng_state()}
        torch.save(state, last)

    import pandas as pd

    pd.DataFrame(log).to_csv(out / "log.csv", index=False)
    print(f"best dev mIoU {best:.4f} (epoch 0 baseline {log[0]['dev_mIoU']:.4f}); checkpoint {out / 'best.pt'}")


if __name__ == "__main__":
    main()
