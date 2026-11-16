"""Reuse full-patch inference on a fresh sample with a larger development set."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "instances_20261114"
sys.path.insert(0, str(PREVIOUS))
import prepare as source

CACHE = HERE / "cache"


def main():
    before = source.load(PREVIOUS / "split.json")
    used = set(before["excluded_ids"])
    for group in ("train", "dev", "test"):
        used.update(r["image_index"] for r in before[group])
    old = source.prior_module()
    source.HERE = HERE
    source.CACHE = CACHE
    CACHE.mkdir(exist_ok=True)
    if not (HERE / "split.json").exists():
        archive, member = old.members("types.npy")
        with archive, archive.open(member) as file:
            tissues = np.load(file)
        eligible = [i for i in range(len(tissues)) if i not in used]
        chosen = np.random.default_rng(20261116).choice(eligible, 100, replace=False)
        test = [dict(image_index=int(i), tissue=str(tissues[i])) for i in sorted(chosen)]
        source.write("split.json", dict(train=before["train"], dev=before["dev"] + before["test"],
                                      test=test, excluded_ids=sorted(used), seed=20261116))
    split = source.load(HERE / "split.json")
    for group, prefixes in (("train", ("gt",)), ("dev", ("gt", "pred"))):
        for rec in split[group]:
            for prefix in prefixes:
                name = f'{prefix}_{rec["image_index"]:04d}.npz'
                target = CACHE / name
                if not target.exists():
                    target.symlink_to(PREVIOUS / "cache" / name)
    source.prepare()
    source.infer("test")


if __name__ == "__main__":
    main()
