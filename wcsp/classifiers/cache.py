# Classifier outputs saved once and read by every WCSP run.
#
# Layout: <root>/<dataset>/<classifier>/<split>/shard_<i>_of_<n>.npz, each
# holding the ids, float16 probabilities and (optionally) instance maps of
# the samples in that shard. One file per shard keeps the inode count low on
# the shared filesystems.

import json
from pathlib import Path

import numpy as np

from .base import Prediction


class ProbStore:
    def __init__(self, root, dataset, classifier, split):
        self.dir = Path(root) / dataset / classifier / split
        self._index = None
        self._open = {}

    def shard_path(self, index, count):
        return self.dir / f"shard_{index:04d}_of_{count:04d}.npz"

    def write(self, index, count, ids, predictions):
        self.dir.mkdir(parents=True, exist_ok=True)
        arrays = {
            "ids": np.array(ids),
            "probs": np.stack([p.probs for p in predictions]).astype(np.float16),
        }
        if predictions and predictions[0].instances is not None:
            arrays["instances"] = np.stack([p.instances for p in predictions]).astype(np.int32)
        path = self.shard_path(index, count)
        tmp = path.with_suffix(".partial.npz")
        np.savez(tmp, **arrays)
        tmp.replace(path)

    def _build_index(self):
        self._index = {}
        for path in sorted(self.dir.glob("shard_*_of_*.npz")):
            if path.name.endswith(".partial.npz"):
                continue
            with np.load(path) as z:
                for row, sample_id in enumerate(z["ids"].tolist()):
                    self._index[sample_id] = (path, row)

    def ids(self):
        if self._index is None:
            self._build_index()
        return list(self._index)

    def __contains__(self, sample_id):
        if self._index is None:
            self._build_index()
        return sample_id in self._index

    def load(self, sample_id):
        if self._index is None:
            self._build_index()
        path, row = self._index[sample_id]
        if path not in self._open:
            self._open = {path: dict(np.load(path))}   # keep one shard in memory
        z = self._open[path]
        instances = z["instances"][row] if "instances" in z else None
        return Prediction(z["probs"][row].astype(np.float32), instances)

    # Fitted temperature (see wcsp.classifiers.calibration).
    def save_temperature(self, value):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "temperature.json").write_text(json.dumps({"temperature": value}))

    def temperature(self):
        path = self.dir / "temperature.json"
        if not path.exists():
            raise FileNotFoundError(f"{path}: run `python -m wcsp calibrate` first.")
        return json.loads(path.read_text())["temperature"]
