from pathlib import Path
import numpy as np


root_dir = Path("./train")


def check_length(path):
    files = tuple(p.stem for p in (path).glob("*"))
    return len(files)


dataset_num = 800
# 000319.npy
# print(np.load('data/temp/np_save.npy'))
# assert (
#     check_length(root_dir / "train_images")
#     == check_length(root_dir / "train_depth")
#     == check_length(root_dir / "train_masks")
# )

from sklearn.model_selection import train_test_split


def split_data(dataset_num):
    X_trainval, X_test = train_test_split(
        range(dataset_num), test_size=0.1, random_state=19
    )
    X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=19)
    return X_train, X_val, X_test


X_train, X_val, X_test = split_data(dataset_num)
print("Train Size   : ", len(X_train))
print("Val Size     : ", len(X_val))
print("Test Size    : ", len(X_test))

from torch.utils.data import Dataset, DataLoader
import cv2


class DroneDataset(Dataset):
    def __init__(self, root_dir, X):
        self.img_path = root_dir / "train_images"
        self.depth_path = root_dir / "train_depth"
        self.mask_path = root_dir / "train_masks"
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        id = self.X[idx]
        img = cv2.imread(self.img_path / f"image_{id:06}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path / f"image_{id:06}.png", cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug["image"])
            mask = aug["mask"]

        if self.transform is None:
            img = Image.fromarray(img)

        # t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        t = T.ToTensor()
        img = t(img)
        mask = torch.from_numpy(mask).long()
        return img, mask


# datasets
train_set = DroneDataset(root_dir, X_train)
val_set = DroneDataset(root_dir, X_val)

# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
