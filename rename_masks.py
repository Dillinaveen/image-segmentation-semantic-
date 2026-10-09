import os

mask_dir = "dataset/train/masks"

for fname in os.listdir(mask_dir):
    if fname.endswith("_L.png"):
        new_name = fname.replace("_L", "")
        old_path = os.path.join(mask_dir, fname)
        new_path = os.path.join(mask_dir, new_name)
        os.rename(old_path, new_path)
import os

mask_dir = "dataset/test/masks"

for fname in os.listdir(mask_dir):
    if fname.endswith("_L.png"):
        new_name = fname.replace("_L", "")
        old_path = os.path.join(mask_dir, fname)
        new_path = os.path.join(mask_dir, new_name)
        os.rename(old_path, new_path)
import os

mask_dir = "dataset/val/masks"

for fname in os.listdir(mask_dir):
    if fname.endswith("_L.png"):
        new_name = fname.replace("_L", "")
        old_path = os.path.join(mask_dir, fname)
        new_path = os.path.join(mask_dir, new_name)
        os.rename(old_path, new_path)

