import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

IGNORE_INDEX = 255

CAMVID_COLOR_TO_CLASS = {
    (128,128,128): 0,   # Sky
    (128,0,0):     1,   # Building
    (192,192,128): 2,   # Pole
    (128,64,128):  3,   # Road
    (0,0,192):     4,   # Sidewalk
    (128,128,0):   5,   # Tree
    (192,128,128): 6,   # SignSymbol
    (64,64,128):   7,   # Fence
    (64,0,128):    8,   # Car
    (64,64,0):     9,   # Pedestrian
    (0,128,192):   10   # Bicyclist
}

class CamVidDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        # ---- IMAGE ----
        img = cv2.imread(os.path.join(self.img_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = img / 255.0
        img = torch.tensor(img).permute(2, 0, 1).float()

        # ---- MASK (RGB → class index) ----
        mask_rgb = cv2.imread(os.path.join(self.mask_dir, name))
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.resize(
            mask_rgb, (256, 256),
            interpolation=cv2.INTER_NEAREST
        )

        mask = np.ones((256,256), dtype=np.uint8) * IGNORE_INDEX

        for color, class_id in CAMVID_COLOR_TO_CLASS.items():
            matches = np.all(mask_rgb == color, axis=-1)
            mask[matches] = class_id

        mask = torch.tensor(mask).long()
        print(torch.unique(mask))


        return img, mask
