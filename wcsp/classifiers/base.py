# What every pixel classifier returns.

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Prediction:
    probs: np.ndarray                    # (K, h, w) float, wcsp label order, over sample.window
    instances: np.ndarray | None = None  # (h, w) int32 detected nuclei, 0 = background

    @property
    def informative(self):
        """Pixels whose probabilities say something about cell type."""
        return None if self.instances is None else self.instances > 0


class PixelClassifier(Protocol):
    name: str

    def predict(self, samples) -> list[Prediction]:
        """One Prediction per wcsp.data.Sample, covering its grid window."""
        ...
