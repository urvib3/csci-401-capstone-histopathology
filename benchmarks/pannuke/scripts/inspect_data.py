import numpy as np

images_data=np.load('../data/images/fold2/images.npy')
types_data=np.load('../data/images/fold2/types.npy')
masks_data=np.load('../data/masks/fold2/masks.npy')

# print("Images shape: \n")
# print(images_data.shape)
# print(images_data.dtype)

# print("Types shape: \n")
# print(types_data.shape)
# print(types_data.dtype)

# print("Masks shape: \n")
# print(masks_data.shape)
# print(masks_data.dtype)

first_image=images_data[0]
print("Image shape:", first_image.shape)
print("Tissue type: ", types_data[0])
print("Minimum: ", np.min(first_image))
print("Maximum: ", np.max(first_image))

print("\nFirst Image Mask Channels Breakdown:")
first_mask=masks_data[0]
for i in range(6):
    channel=first_mask[..., i]
    c_min=np.min(channel)
    c_max=np.max(channel)
    c_unique=len(np.unique(channel))
    print(f"Channel {i} -> Min: {c_min}, Max: {c_max}, Unique values: {c_unique}")
