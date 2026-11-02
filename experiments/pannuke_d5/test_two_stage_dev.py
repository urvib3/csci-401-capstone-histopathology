import io
import zipfile

import numpy as np
import pytest

from experiments.pannuke_d5.run_two_stage_dev import ground_truth_from_zip


def archive_masks(tmp_path, masks):
    data = io.BytesIO()
    np.save(data, masks)
    path = tmp_path / 'fold.zip'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('Fold 2/masks/fold2/masks.npy', data.getvalue())
    return path


def test_streamed_ground_truth_keeps_indices_crop_and_class_labels(tmp_path):
    masks = np.zeros((4, 256, 256, 6), dtype=np.uint8)
    masks[1, 46, 46, 2] = 17  # Instance ID becomes class 3 at cropped (0,0).
    masks[3, 209, 209, 4] = 9
    masks[3, 210, 210, 1] = 12  # Outside the valid region.
    masks[..., 5] = 1  # Background channel is not a biological class.
    result = ground_truth_from_zip(archive_masks(tmp_path, masks), [3, 1, 3])
    assert set(result) == {1, 3}
    assert result[1].shape == (164, 164)
    assert result[1][0, 0] == 3
    assert result[3][-1, -1] == 5
    assert np.count_nonzero(result[3]) == 1


def test_bad_archive_index_rejected(tmp_path):
    masks = np.zeros((1, 256, 256, 6), dtype=np.uint8)
    with pytest.raises(ValueError, match='outside'):
        ground_truth_from_zip(archive_masks(tmp_path, masks), [1])
