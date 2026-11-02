import numpy as np
from PIL import Image
from scripts.prepare_bcss_upstream import label_image


def test_16_bit_mask_preserves_both_axes_and_label_values():
    labels = np.array([[1, 2, 3], [4, 9, 0]], dtype=np.uint16)
    mask = Image.fromarray(labels)
    assert mask.mode == 'I;16'
    converted = label_image(mask)
    assert converted.size == (3, 2)
    np.testing.assert_array_equal(np.asarray(converted), labels)


def test_rgb_label_mask_uses_first_channel():
    labels = np.array([[1, 2, 3], [4, 9, 0]], dtype=np.uint8)
    mask = Image.fromarray(np.repeat(labels[..., None], 3, axis=-1))
    np.testing.assert_array_equal(np.asarray(label_image(mask)), labels)
