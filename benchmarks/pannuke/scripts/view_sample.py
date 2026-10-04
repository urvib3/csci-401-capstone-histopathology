import numpy as np
import matplotlib.pyplot as plt

images_data=np.load('../data/images/fold2/images.npy')
types_data=np.load('../data/images/fold2/types.npy')
masks_data=np.load('../data/masks/fold2/masks.npy')
first_mask=masks_data[0]
first_image=images_data[0]
first_type=types_data[0]
arr_clipped=np.clip(first_image, 0.0, 255.0)
arr_clipped = arr_clipped.astype(np.uint8)
# plt.imsave('../results/sample_predictions/input_image_0.png', arr_clipped)

for i in range(len(images_data)):
    ith_image=images_data[i]
    image_clipped=np.clip(ith_image, 0.0, 255.0)
    image_clipped=image_clipped.astype(np.uint8)
    plt.imsave(f'../hovernet_inputs_fold2/image_00{i}.png', image_clipped)

# channels_to_check = first_mask[:, :, :5]
# nucleus_mask = np.any(channels_to_check > 0, axis=2)
# binary_map = nucleus_mask.astype(np.uint8) * 255
# plt.imsave('../results/sample_predictions/ground_truth_mask_0.png', binary_map)
