"""Prepare a 300-patch replication plus a separate Dead-positive stress set."""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "presence_20261114"
sys.path.insert(0, str(PREVIOUS))
from prepare_fresh import source

CACHE = HERE / "cache"
WORKERS = 3
SEED = 20261117


def prepare():
    previous_split = source.load(PREVIOUS / "split.json")
    known = set(previous_split["excluded_ids"])
    known.update(row["image_index"] for group in ("train", "dev", "test") for row in previous_split[group])
    old = source.prior_module()
    if not (HERE / "split.json").exists():
        archive, member = old.members("types.npy")
        with archive, archive.open(member) as file:
            tissues = np.load(file)
        eligible = [i for i in range(len(tissues)) if i not in known]
        ids = np.random.default_rng(SEED).choice(eligible, 300, replace=False)
        primary = set(map(int, ids))
        other = [i for i in eligible if i not in primary]
        secondary = [i for i, mask in old.stream_arrays("masks.npy", other) if np.any(mask[..., 3] > 0)]
        records = lambda ids: [dict(image_index=i, tissue=str(tissues[i])) for i in sorted(ids)]
        source.write("split.json", dict(
            train=[], dev=[], test=records(primary) + records(secondary),
            primary=records(primary), dead_stress=records(secondary), excluded_ids=sorted(known),
            seed=SEED, scope="Primary sample drawn uniformly before viewing model metrics; secondary contains all other eligible Dead-positive patches."
        ))
        source.write("protocol.json", dict(
            prior_experiment=str(PREVIOUS.name), n_primary=300, n_dead_stress=len(secondary),
            primary="Frozen type-set WCSP versus Raw TP and fixed-mask voting; pooled five-class mIoU.",
            secondary="Paired intervals without Dead; separate selected Dead-positive cohort, not pooled into the primary result.",
            paired_bootstrap=5000, sampling_seed=SEED,
            frozen_inputs={name: hashlib.sha256((PREVIOUS / name).read_bytes()).hexdigest()
                           for name in ("frozen.json", "priors.json", "run.py")},
        ))
        print("Primary:", len(primary), "Dead-positive stress:", len(secondary), flush=True)
    source.prepare()


def worker(i):
    original_load, original_write = source.load, source.write
    path = HERE / "split.json"
    split = original_load(path)

    def load_shard(file):
        if file == path:
            return dict(split, test=split["test"][i::WORKERS])
        return original_load(file)

    def write_shard(name, payload):
        original_write(f"inference_{i}.json" if name == "inference_test.json" else name, payload)

    original_model_loader = source.prior_module

    def prior_module():
        old = original_model_loader()
        original = old.load_model

        def load_model():
            import torch
            model, hashes = original()
            torch.set_num_threads(3)
            return model, hashes

        old.load_model = load_model
        return old

    source.load, source.write, source.prior_module = load_shard, write_shard, prior_module
    source.infer("test")


def accelerate():
    """Finish pending patches with the checked Apple GPU inference path."""
    original = source.prior_module
    records = source.load(HERE / "split.json")["test"]
    existing = [r["image_index"] for r in records
                if (CACHE / f'pred_{r["image_index"]:04d}.npz').exists()]
    source.write("inference_backends.json", dict(cpu_patch_ids=existing,
        mps_patch_ids=[r["image_index"] for r in records if r["image_index"] not in existing]))

    class GPUModel:
        def __init__(self, model):
            self.model = model.to("mps")

        def __call__(self, tensor):
            return {key: value.cpu() for key, value in self.model(tensor.to("mps")).items()}

    def prior_module():
        old = original()
        original_load = old.load_model

        def load_model():
            model, hashes = original_load()
            return GPUModel(model), hashes

        old.load_model = load_model
        return old

    source.prior_module = prior_module
    source.infer("test")


if __name__ == "__main__":
    source.HERE, source.CACHE = HERE, CACHE
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--worker", type=int, choices=range(WORKERS))
    mode.add_argument("--gpu", action="store_true")
    options = parser.parse_args()
    if options.gpu:
        accelerate()
    elif options.worker is None:
        prepare()
    else:
        worker(options.worker)
