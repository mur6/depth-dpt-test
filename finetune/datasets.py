# import os
from pathlib import Path
import numpy as np
from skimage.transform import warp, AffineTransform

import torch
import torch.utils.data as data
import torchvision.utils
from torchvision.transforms import Lambda, Normalize, ToTensor


NYUD_MEAN = [0.48056951, 0.41091299, 0.39225179]
NYUD_STD = [0.28918225, 0.29590312, 0.3093034]



class CustomDepthDataset(data.Dataset):
    def __init__(self, root, split="test", transform=None, limit=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.limit = limit

        if split == "test":
            folder = Path(root), "nyu_depth_v2", "labeled", "npy")
            self.images = np.load(os.path.join(folder, "images.npy"))
            self.depths = np.load(os.path.join(folder, "depths.npy"))
        else:
            folder = os.path.join(root, "nyu_depth_v2", "npy")
            self.file_paths = [
                os.path.join(folder, n) for n in sorted(os.listdir(folder))
            ]

    def __len__(self):
        if hasattr(self, "images"):
            length = len(self.images)
        else:
            length = len(self.file_paths)
        if self.limit is not None:
            length = np.minimum(self.limit, length)
        return length

    def __getitem__(self, index):
        if self.split == "test" or self.debug:
            image = self.images[index]
            depth = self.depths[index]
        else:
            stacked = np.load(self.file_paths[index])
            image = stacked[0:3]
            depth = stacked[3:5]

        if self.transform is not None:
            image, depth = transform_chw(self.transform, [image, depth])

        return image, depth

    def compute_image_mean(self):
        return np.mean(self.images / 255, axis=(0, 2, 3))

    def compute_image_std(self):
        return np.std(self.images / 255, axis=(0, 2, 3))


# def get_transform(training=True, size=(256, 192), normalize=True):
#     if training:
#         transforms = [
#             Merge(),
#             RandomFlipHorizontal(),
#             RandomRotate(angle_range=(-5, 5), mode="constant"),
#             RandomCropNumpy(size=size),
#             RandomAffineZoom(scale_range=(1.0, 1.5)),
#             Split([0, 3], [3, 5]),  #
#             # Note: ToTensor maps from [0, 255] to [0, 1] while to_tensor does not
#             [RandomColor(multiplier_range=(0.8, 1.2)), None],
#         ]
#     else:
#         transforms = [
#             [BilinearResize(0.5), None],
#         ]

#     transforms.extend(
#         [
#             # Note: ToTensor maps from [0, 255] to [0, 1] while to_tensor does not
#             [ToTensor(), Lambda(to_tensor)],
#             [Normalize(mean=NYUD_MEAN, std=NYUD_STD), None] if normalize else None,
#         ]
#     )

