import io
import zipfile

import numpy as np

from experiments.pannuke_d5.export_hovernet_probabilities import image_patches


def test_export_reads_rgb_indices_without_masks_or_ground_truth(tmp_path):
    images=np.zeros((3,256,256,3),dtype=np.float64)
    images[1]=37
    images[2]=125
    buffer=io.BytesIO()
    np.save(buffer,images)
    path=tmp_path/'fold.zip'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('Fold 2/images/fold2/images.npy',buffer.getvalue())
    result=dict(image_patches(path,[2,1]))
    assert set(result)=={1,2}
    assert result[1].shape==(256,256,3)
    assert result[1].dtype==np.float32
    assert (result[1]==37).all()
    assert (result[2]==125).all()
