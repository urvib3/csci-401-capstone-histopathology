# PyTorch adapter for BCSSDataset. Kept separate so the rest of the package
# only needs numpy and Pillow.

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .classes import IGNORE_INDEX


class TorchBCSS(Dataset):
    """
    Wraps a BCSSDataset for torch.

    Each item is (image, mask, index):
        image: float32 tensor (3, H, W) in [0, 1]
        mask:  int64 tensor (H, W); all IGNORE_INDEX for unlabeled tiles
        index: position in the underlying BCSSDataset, for looking up
               bcss_dataset.tiles[index] metadata

    transform:
        Optional callable (image, mask) -> (image, mask) applied to the
        numpy arrays before conversion, e.g. for flips or rotations.
    """

    def __init__(self, bcss_dataset, transform=None):
        self.data = bcss_dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image, mask = sample.image, sample.mask
        if mask is None:
            mask = np.full(image.shape[:2], IGNORE_INDEX, dtype=np.uint8)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        mask = torch.from_numpy(np.ascontiguousarray(mask).astype(np.int64))
        return image.float() / 255.0, mask, index


def make_loader(bcss_dataset, batch_size=16, shuffle=True, num_workers=0,
                transform=None):
    return DataLoader(
        TorchBCSS(bcss_dataset, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
