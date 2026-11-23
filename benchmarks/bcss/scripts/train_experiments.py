"""Controlled TIAToolbox BCSS training arms; select epochs on dev only."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from benchmarks.bcss.scripts.train_fullres_unet import Tiles, augment, report, to_input
from benchmarks.bcss.loader.classes import IGNORE_INDEX
from benchmarks.bcss.loader.metrics import confusion_matrix
from wcsp.classifiers.fullres_unet import FullResUNet, MODEL_NAME
from wcsp.classifiers.training_loss import expected_pairwise_cost
from benchmarks.bcss.scripts.structure_losses import (PresencePrior, cluster_consistency, gt_regions,
                                                       presence_bce, slic_segments, soft_presence)
from wcsp.data.bcss import WINDOW


def logits_for(model, x, architecture):
    return model(x) if architecture == "fullres" else model(model.preproc(x))


def probs_for(logits, size):
    return F.interpolate(logits.softmax(1), size=size, mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate(model, loader, args, window):
    model.eval()
    cm = np.zeros((5, 5), dtype=np.int64)
    center = np.zeros_like(cm)
    for images, target in loader:
        logits = logits_for(model, to_input(images, args.device), args.architecture)
        probs = probs_for(logits, target.shape[-2:]).cpu()
        pred = probs.argmax(1).numpy()
        for p, t in zip(pred, target.numpy()):
            cm += confusion_matrix(p[window], t[window], 5)
            center += confusion_matrix(p[WINDOW], t[WINDOW], 5)
    return {**report(cm), "dev_center_mIoU": report(center)["dev_mIoU"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=["baseline", "fullres"], required=True)
    parser.add_argument("--field", choices=["center", "full"], default="center")
    parser.add_argument("--pairwise-weight", type=float, default=0.0)
    parser.add_argument("--table", default="handwritten")
    # structure losses (benchmarks/bcss/scripts/structure_losses.py); 0 = off, as before
    parser.add_argument("--cluster-weight", type=float, default=0.0)
    parser.add_argument("--cluster-source", choices=["gt", "slic"], default="gt",
                        help="regions for the cluster loss: ground-truth components or SLIC superpixels")
    parser.add_argument("--presence-weight", type=float, default=0.0)
    parser.add_argument("--prior-weight", type=float, default=0.0)
    parser.add_argument("--presence-prior", default="experiments/bcss_wcsp/configs/tables/bcss_presence_prior.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-steps", type=int, default=0)
    parser.add_argument("--limit-dev", type=int, default=0)
    args = parser.parse_args()
    if min(args.pairwise_weight, args.cluster_weight, args.presence_weight, args.prior_weight) < 0 \
            or args.epochs < 1 or args.batch < 1:
        parser.error("loss weights must be nonnegative; epochs and batch must be positive")
    prior = PresencePrior(REPO / args.presence_prior) if args.prior_weight else None
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    window = WINDOW if args.field == "center" else (slice(None), slice(None))
    fit, dev = Tiles("fit", args.root), Tiles("dev", args.root)
    if args.limit_dev:
        dev = Subset(dev, np.linspace(0, len(dev)-1, min(args.limit_dev, len(dev))).round().astype(int))
    train_loader = DataLoader(fit, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              persistent_workers=args.workers > 0, pin_memory=args.device.startswith("cuda"))
    dev_loader = DataLoader(dev, batch_size=args.batch, num_workers=args.workers)
    from wcsp.classifiers.tia_model import get_pretrained_model
    base, _ = get_pretrained_model(MODEL_NAME, pretrained_weights=args.weights)
    model = (FullResUNet(base, width=args.width, freeze_base=False)
             if args.architecture == "fullres" else base).to(args.device)
    if args.architecture == "fullres":
        groups = [{"params": model.base.parameters(), "lr": args.lr},
                  {"params": [p for n, p in model.named_parameters() if not n.startswith("base.")], "lr": args.head_lr}]
    else:
        groups = [{"params": model.parameters(), "lr": args.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    # Epoch schedule is explicit and verified when resuming.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    first, best, log = 0, -1.0, []
    last = out / "last.pt"
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        for key, value in vars(args).items():
            if key not in {"workers", "threads", "device", "out"} and state["args"].get(key) != value:
                raise ValueError(f"resume setting changed: {key}; use a new output directory")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        first, best, log = state["epoch"], state["best"], state["log"]
        torch.set_rng_state(state["rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
    def save(path, epoch, **extra):
        tmp = path.with_suffix(".partial.pt")
        torch.save({"model": model.state_dict(), "width": args.width, "args": vars(args),
                    "epoch": epoch, **extra}, tmp)
        tmp.replace(path)
    if not log:
        row = {"epoch": 0, "ce": float("nan"), "pairwise": float("nan"), "cluster": float("nan"),
               "presence": float("nan"), "prior": float("nan"), "loss": float("nan"),
               "seconds": 0.0, **evaluate(model, dev_loader, args, window)}
        best = row["dev_mIoU"]
        log.append(row)
        save(out / "best.pt", 0, dev=row)
        print(json.dumps(row), flush=True)
    for epoch in range(first, args.epochs):
        model.train()
        start, totals, steps = time.perf_counter(), np.zeros(6), 0
        for step, (images, masks) in enumerate(train_loader):
            if args.limit_steps and step >= args.limit_steps:
                break
            x, target = augment(to_input(images, args.device), masks.to(args.device))
            logits = logits_for(model, x, args.architecture)
            # Interpolate probabilities exactly as in classifier inference.
            probs = probs_for(logits, target.shape[-2:])[(...,) + window]
            target = target[(...,) + window]
            valid = target != IGNORE_INDEX
            if not valid.any():
                continue
            ce = F.nll_loss(probs.clamp_min(1e-7).log(), target, ignore_index=IGNORE_INDEX)
            pairwise = (expected_pairwise_cost(probs, target, args.table)
                        if args.pairwise_weight else ce * 0.0)
            zero = ce * 0.0
            regions = ((gt_regions(target, valid) if args.cluster_source == "gt"
                        else slic_segments(x[(...,) + window])) if args.cluster_weight else None)
            cluster = cluster_consistency(probs, regions, valid) if args.cluster_weight else zero
            presence = presence_bce(probs, target, valid) if args.presence_weight else zero
            prior_cost = prior.expected_cost(soft_presence(probs, valid)) if prior is not None else zero
            loss = (ce + args.pairwise_weight * pairwise + args.cluster_weight * cluster
                    + args.presence_weight * presence + args.prior_weight * prior_cost)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            totals += [ce.item(), pairwise.item(), cluster.item(), presence.item(), prior_cost.item(), loss.item()]
            steps += 1
            if step % 50 == 0:
                print(f"epoch {epoch+1} step {step}: CE={ce.item():.4f} pair={pairwise.item():.4f} "
                      f"cluster={cluster.item():.4f} presence={presence.item():.4f} prior={prior_cost.item():.4f}",
                      flush=True)
        if not steps:
            raise RuntimeError("no labeled training batches")
        sched.step()
        row = {"epoch": epoch+1, **dict(zip(("ce", "pairwise", "cluster", "presence", "prior", "loss"), totals/steps)),
               "seconds": time.perf_counter()-start, **evaluate(model, dev_loader, args, window)}
        log.append(row)
        if row["dev_mIoU"] > best:
            best = row["dev_mIoU"]
            save(out / "best.pt", epoch+1, dev=row)
        save(last, epoch+1, opt=opt.state_dict(), sched=sched.state_dict(), best=best, log=log,
             rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
             cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        with (out / "log.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(log)
        print(json.dumps(row), flush=True)
    (out / "settings.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
