"""Full-patch HoVer-Net instances and a patch-disjoint evaluation split."""
import ast
import importlib.util
import inspect
import json
import math
from pathlib import Path
import time
import warnings

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HIER = HERE.parent / "hierarchy_20261109"
DECAY = HERE.parent / "decay_20261114"
D5 = HERE.parent / "pannuke_d5"
CACHE = HERE / "cache"


def load(path):
    return json.loads(path.read_text())


def write(name, value):
    target = HERE / name
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(target)


def prior_module():
    spec = importlib.util.spec_from_file_location("previous_pannuke", D5 / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ground_truth(mask):
    instances = np.zeros(mask.shape[:2], np.int32)
    types = [0]
    for channel in range(5):
        values, inverse = np.unique(mask[..., channel], return_inverse=True)
        foreground = values > 0
        index = np.zeros(len(values), np.int32)
        index[foreground] = np.arange(len(types), len(types) + foreground.sum())
        instances[mask[..., channel] > 0] = index[inverse.reshape(instances.shape)][mask[..., channel] > 0]
        types.extend([channel + 1] * int(foreground.sum()))
    return instances, np.asarray(types, np.uint8)


def prepare():
    old = prior_module()
    prior = load(HIER / "split.json")
    decay = load(DECAY / "split.json")
    if not (HERE / "split.json").exists():
        excluded = set(decay["excluded_ids"])
        excluded.update(r["image_index"] for group in ("dev", "test") for r in decay[group])
        archive, member = old.members("types.npy")
        with archive, archive.open(member) as file:
            tissues = np.load(file)
        eligible = [i for i in range(len(tissues)) if i not in excluded]
        ids = np.random.default_rng(20261115).choice(eligible, 100, replace=False)
        records = [dict(image_index=int(i), tissue=str(tissues[i])) for i in sorted(ids)]
        write("split.json", dict(train=prior["train"], dev=prior["dev"], test=records,
                                 excluded_ids=sorted(excluded), seed=20261115,
                                 scope="Same fold and checkpoint; newly sampled patch IDs, patient separation unverified."))
    split = load(HERE / "split.json")
    selected = [r["image_index"] for group in ("dev", "test") for r in split[group]]
    fitting = [r["image_index"] for r in split["train"]]
    assert not set(fitting + [r["image_index"] for r in split["dev"]]) & {
        r["image_index"] for r in split["test"]}
    CACHE.mkdir(exist_ok=True)
    missing = [i for i in selected + fitting if not (CACHE / f"gt_{i:04d}.npz").exists()]
    if missing:
        for i, masks in old.stream_arrays("masks.npy", missing):
            instances, types = ground_truth(masks)
            np.savez_compressed(CACHE / f"gt_{i:04d}.npz", instances=instances, types=types)
    missing = [i for i in selected if not (CACHE / f"rgb_{i:04d}.npy").exists()]
    if missing:
        for i, image in old.stream_arrays("images.npy", missing):
            converted = image.astype(np.uint8)
            assert np.array_equal(image, converted)
            np.save(CACHE / f"rgb_{i:04d}.npy", converted)
    print("Ground truth and RGB extracted", flush=True)


def watershed_postprocessor():
    import cv2
    from scipy import ndimage
    from skimage.morphology import remove_small_objects
    from skimage.segmentation import watershed

    # Load the installed implementation without loading WSI support.
    path = ROOT / ".venv/lib/python3.12/site-packages/tiatoolbox/models/architecture/hovernet.py"
    model = next(node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.ClassDef) and node.name == "HoVerNet")
    method = next(node for node in model.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_proc_np_hv")
    method.decorator_list = []
    if "max_size" not in inspect.signature(remove_small_objects).parameters:
        original = remove_small_objects

        def remove_small_objects(labels, max_size):
            return original(labels, min_size=max_size + 1)

    scope = dict(np=np, cv2=cv2, ndimage=ndimage, remove_small_objects=remove_small_objects,
                 watershed=watershed, math=math, warnings=warnings)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope["_proc_np_hv"]


def infer(group):
    import torch

    old = prior_module()
    start = time.perf_counter()
    model, hashes = old.load_model()
    checkpoint = old.sha(old.DATA / "cache/models/hovernet_fast-pannuke.pth")
    reference = load(HIER / "inference.json")
    assert checkpoint == reference["checkpoint_sha256"]
    assert hashes == reference["architecture_sha256"]
    postprocess = watershed_postprocessor()
    print("Model ready", round(time.perf_counter() - start, 2), flush=True)
    records = load(HERE / "split.json")[group]
    for row in records:
        index = row["image_index"]
        target = CACHE / f"pred_{index:04d}.npz"
        if target.exists():
            continue
        then = time.perf_counter()
        image = np.load(CACHE / f"rgb_{index:04d}.npy")
        padded = np.pad(image, ((46, 46), (46, 46), (0, 0)), mode="reflect")
        total = np.zeros((256, 256, 9), float)
        count = np.zeros((256, 256, 1), int)
        for y in (0, 92):
            for x in (0, 92):
                tile = np.ascontiguousarray(padded[y:y + 256, x:x + 256])
                tensor = torch.from_numpy(tile.astype(np.float32)).permute(2, 0, 1)[None]
                with torch.inference_mode():
                    out = model(tensor)
                    tp = out["tp"][0].permute(1, 2, 0).numpy().astype(float)
                    tp -= tp.max(axis=-1, keepdims=True)
                    tp = np.exp(tp)
                    tp /= tp.sum(axis=-1, keepdims=True)
                    foreground = out["np"].softmax(dim=1)[0, 1, ..., None].numpy()
                    hv = out["hv"][0].permute(1, 2, 0).numpy()
                total[y:y + 164, x:x + 164] += np.concatenate([tp, foreground, hv], axis=-1)
                count[y:y + 164, x:x + 164] += 1
        fused = total / count
        prob = fused[..., :6].astype(np.float32)
        if group == "dev":
            with np.load(HIER / "cache" / f"prob_{index:04d}.npz") as old_prob:
                np.testing.assert_array_equal(prob, old_prob["p"])
        labels = postprocess(fused[..., 6:7].astype(np.float32), fused[..., 7:].astype(np.float32))
        _, ids = np.unique(labels, return_inverse=True)
        # Retain zero as background even on an entirely foreground patch.
        if not np.any(labels == 0):
            ids += 1
        temp = target.with_suffix(".tmp.npz")
        np.savez_compressed(temp, p=prob, instances=ids.reshape(labels.shape).astype(np.int32))
        temp.replace(target)
        print(group, index, round(time.perf_counter() - then, 2), flush=True)
    write(f"inference_{group}.json", dict(checkpoint_sha256=checkpoint, architecture_sha256=hashes,
                                         patch_ids=[r["image_index"] for r in records],
                                         tiles=[[0, 0], [0, 92], [92, 0], [92, 92]],
                                         reflect_padding=46, postprocess="HoVer-Net watershed after full-patch NP/HV fusion"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "dev", "test"])
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        infer(args.stage)
