# Tests BCSSDataset against a tiny synthetic tree that mirrors the Kaggle
# layout and file naming, so no download is needed.

import numpy as np
import pytest
from PIL import Image

from bcss import IGNORE_INDEX, BCSSDataset, coarse_lut
from bcss.dataset import parse_tile_name

ROI = "TCGA-A1-A0SK-DX1_xmin45749_ymin25055"


def tile_name(r, c, size):
    return f"{ROI}_MPP-0_{r}_{c}_size{size}.png"


def write_tile(directory, name, size, value):
    directory.mkdir(parents=True, exist_ok=True)
    image = np.full((size, size, 3), value, dtype=np.uint8)
    Image.fromarray(image).save(directory / name)


def write_mask(directory, name, size, value):
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size, size), value, dtype=np.uint8)).save(directory / name)


@pytest.fixture
def root(tmp_path):
    base = tmp_path / "BCSS"
    # 2 x 2 tiles of one ROI split across train / val / test.
    for (r, c), split, label in [
        ((0, 0), "train", 0),
        ((0, 224), "train", 1),
        ((224, 0), "val", 2),
        ((224, 224), "test", None),
    ]:
        name = tile_name(r, c, 224)
        write_tile(base / split, name, 224, 10 * (label or 0))
        if label is not None:
            write_mask(base / f"{split}_mask", name, 224, label)

    base512 = tmp_path / "BCSS_512"
    name = tile_name(0, 512, 512)
    write_tile(base512 / "train_512", name, 512, 50)
    write_mask(base512 / "train_mask_512", name, 512, 20)  # dcis -> tumor
    return tmp_path


def test_parse_tile_name():
    m = parse_tile_name("TCGA-A2-A0D0-DX1_xmin68482_ymin39071_MPP-0_5152_2240_size224.png")
    assert m["slide"] == "TCGA-A2-A0D0-DX1"
    assert (m["xmin"], m["ymin"], m["off0"], m["off1"], m["size"]) == (
        "68482", "39071", "5152", "2240", "224")
    with pytest.raises(ValueError):
        parse_tile_name("not_a_tile.png")


def test_load_split(root):
    data = BCSSDataset(root, "224", "train")
    assert len(data) == 2
    sample = data[1]
    assert sample.image.shape == (224, 224, 3) and sample.image.dtype == np.uint8
    assert sample.mask.shape == (224, 224)
    assert sample.info.roi_id == ROI
    assert sample.info.offset == (0, 224)
    assert np.all(sample.mask == 1)


def test_test_split_has_no_masks(root):
    data = BCSSDataset(root, "224", "test")
    assert data[0].mask is None
    assert list(data.masks()) == []


def test_unknown_split_and_missing_dir(root, tmp_path):
    with pytest.raises(ValueError):
        BCSSDataset(root, "512", "test")
    with pytest.raises(FileNotFoundError):
        BCSSDataset(tmp_path / "nowhere", "224", "train")


def test_stitch_across_splits(root):
    data = BCSSDataset(root, "224", ["train", "val", "test"])
    assert data.rois() == {ROI: [0, 1, 2, 3]}
    image, mask = data.stitch_roi(ROI)
    assert image.shape == (448, 448, 3)
    assert mask[0, 0] == 0 and mask[0, 300] == 1 and mask[300, 0] == 2
    assert mask[300, 300] == IGNORE_INDEX  # test tile has no mask


def test_stitch_leaves_holes_for_missing_tiles(root):
    image, mask = BCSSDataset(root, "224", "train").stitch_roi(ROI)
    assert image.shape == (224, 448, 3)
    assert mask.shape == (224, 448)


def test_lut_applied_to_512_masks(root):
    lut, names = coarse_lut()
    sample = BCSSDataset(root, "512", "train", lut=lut)[0]
    assert names[sample.mask[0, 0]] == "tumor"


def test_torch_adapter(root):
    torch = pytest.importorskip("torch")
    from bcss.torch_data import make_loader

    loader = make_loader(BCSSDataset(root, "224", ["train", "test"]),
                         batch_size=3, shuffle=False)
    images, masks, index = next(iter(loader))
    assert images.shape == (3, 3, 224, 224) and images.dtype == torch.float32
    assert masks.shape == (3, 224, 224) and masks.dtype == torch.int64
    assert torch.all(masks[2] == IGNORE_INDEX)
    assert index.tolist() == [0, 1, 2]
