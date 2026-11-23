# CONCH zero-shot on the BCSS dev tiles: does a pathology vision-language model know
# which tissue is where, without training?
#
# Each 256 x 256 tile (the U-Net/WCSP grid) is scanned with windows (default 96 px,
# stride 32). Every window is scored against text prompts for the five classes (prompt
# ensembles, CONCH paper style); window probabilities are averaged per pixel into a
# coarse (5, 256, 256) map. Reported on the same 185 dev tiles as the sweeps:
#
#   zero-shot   mIoU of the CONCH map alone
#   presence    per-class AUC for "class covers >= 1% of the tile", CONCH vs U-Net
#   fusion      argmax of log p_unet + w log p_conch, for a few w
#
# Maps are saved (float16) so they can be used as WCSP evidence later.
#
#   python experiments/bcss_wcsp/vlm/conch_zeroshot.py --checkpoint hf_hub:MahmoodLab/conch

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from wcsp import pipeline  # noqa: E402
from wcsp.config import load_config  # noqa: E402
from wcsp.data import select  # noqa: E402
from wcsp.evaluate.pixel import pixel_confusion, pixel_report  # noqa: E402
from wcsp.labels import IGNORE, NUM_LABELS  # noqa: E402

NAMES = ["Tumor", "Inflammatory", "Stroma", "Necrosis", "Other"]
CLASSES = {   # wcsp label order: tumor, inflammatory, stroma (connective), necrosis, other
    "Tumor": ["invasive ductal carcinoma", "breast carcinoma", "tumor cells", "malignant epithelium",
              "carcinoma cells"],
    "Inflammatory": ["lymphocytes", "tumor-infiltrating lymphocytes", "inflammatory infiltrate",
                     "immune cells", "lymphocytic infiltrate"],
    "Stroma": ["stroma", "fibrous stroma", "desmoplastic stroma", "collagenous connective tissue",
               "tumor-associated stroma"],
    "Necrosis": ["necrosis", "necrotic tissue", "tumor necrosis", "necrotic debris", "comedonecrosis"],
    "Other": ["adipose tissue", "blood vessels", "normal breast ducts", "red blood cells", "mucin"],
}
TEMPLATES = ["{}.", "an H&E image of {}.", "a photomicrograph showing {}.", "{} in breast tissue.",
             "an image of {}.", "histopathology of {}."]


def text_classifier(model, device):
    from conch.open_clip_custom import get_tokenizer, tokenize

    tokenizer = get_tokenizer()
    weights = []
    for name in NAMES:
        texts = [t.format(c) for c in CLASSES[name] for t in TEMPLATES]
        with torch.inference_mode():
            emb = model.encode_text(tokenize(texts=texts, tokenizer=tokenizer).to(device))
        emb = emb / emb.norm(dim=-1, keepdim=True)
        mean = emb.mean(0)
        weights.append(mean / mean.norm())
    return torch.stack(weights)                                   # (5, d)


def windows(size, win, stride):
    starts = list(range(0, size - win + 1, stride))
    if starts[-1] != size - win:
        starts.append(size - win)
    return [(y, x) for y in starts for x in starts]


def conch_map(model, preprocess, classifier, image, win, stride, device):
    from PIL import Image

    h, w = image.shape[:2]
    spots = windows(h, win, stride)
    crops = torch.stack([preprocess(Image.fromarray(image[y:y + win, x:x + win])) for y, x in spots]).to(device)
    with torch.inference_mode():
        emb = model.encode_image(crops, proj_contrast=True, normalize=True)
        logits = model.logit_scale.exp() * emb @ classifier.T
        probs = logits.softmax(-1).float().cpu().numpy()                # (n, 5)
    acc = np.zeros((NUM_LABELS, h, w))
    count = np.zeros((h, w))
    for (y, x), p in zip(spots, probs):
        acc[:, y:y + win, x:x + win] += p[:, None, None]
        count[y:y + win, x:x + win] += 1
    return acc / count, probs


def auc(y, s):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/bcss_wcsp/configs/bcss_unet.yaml")
    parser.add_argument("--checkpoint", default="hf_hub:MahmoodLab/conch")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--win", type=int, default=96)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--out", default="/scratch1/ubhuwani/wcsp/vlm/conch_dev")
    args = parser.parse_args()

    from conch.open_clip_custom import create_model_from_pretrained

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = create_model_from_pretrained("conch_ViT-B-16", args.checkpoint, device=device)
    model.eval()
    classifier = text_classifier(model, device)

    cfg = load_config(args.config)
    data = pipeline.dataset_for(cfg, "dev")
    store = pipeline.store_for(cfg, "dev")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    weights = [0.0, 0.25, 0.5, 1.0, 2.0]
    conf = {"conch": np.zeros((NUM_LABELS, NUM_LABELS)), "unet": np.zeros((NUM_LABELS, NUM_LABELS)),
            **{f"fuse_{w}": np.zeros((NUM_LABELS, NUM_LABELS)) for w in weights[1:]}}
    truth_p, conch_s, unet_s, ids, maps = [], [], [], [], []
    for i in select(len(data), limit=args.limit):
        if data.ids[i] not in store:
            continue
        sample = data[i]
        target = sample.grid_target
        if not (target != IGNORE).any():
            continue
        unet = store.load(sample.id).probs                                # (5, H, W)
        cmap, wprobs = conch_map(model, preprocess, classifier, sample.grid_image, args.win, args.stride, device)
        conf["conch"] += pixel_confusion(cmap.argmax(0), target)
        conf["unet"] += pixel_confusion(unet.argmax(0), target)
        for w in weights[1:]:
            fused = np.log(np.clip(unet, 1e-6, 1)) + w * np.log(np.clip(cmap, 1e-6, 1))
            conf[f"fuse_{w}"] += pixel_confusion(fused.argmax(0), target)
        ok = target != IGNORE
        share = np.bincount(target[ok], minlength=NUM_LABELS)[:NUM_LABELS] / ok.sum()
        truth_p.append(share >= 0.01)
        conch_s.append(wprobs.max(0))                                     # strongest window per class
        unet_s.append(unet.reshape(NUM_LABELS, -1).mean(1))               # U-Net mean probability
        ids.append(sample.id)
        maps.append(cmap.astype(np.float16))
        print(f"  {len(ids)} {sample.id}", flush=True)

    truth_p, conch_s, unet_s = map(np.array, (truth_p, conch_s, unet_s))
    report = {
        "tiles": len(ids), "win": args.win, "stride": args.stride,
        "mIoU": {k: float(pixel_report(v)["mIoU"]) for k, v in conf.items()},
        "per_class_IoU": {k: {n: float(pixel_report(v)[f"IoU_{lab}"]) for n, lab in
                              zip(NAMES, ["neoplastic", "inflammatory", "connective", "dead", "healthy"])}
                          for k, v in conf.items()},
        "presence_AUC": {n: {"conch": auc(truth_p[:, k], conch_s[:, k]), "unet": auc(truth_p[:, k], unet_s[:, k])}
                         for k, n in enumerate(NAMES)},
    }
    np.savez_compressed(out / "maps.npz", ids=np.array(ids), maps=np.stack(maps),
                        truth_present=truth_p, conch_score=conch_s, unet_score=unet_s)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
