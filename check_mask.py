import cv2
import numpy as np

mask = cv2.imread("dataset/train/masks/Seq05VD_f01770.png", 0)

if mask is None:
    print("Mask not found")
else:
    print("Unique values in mask:", np.unique(mask))
