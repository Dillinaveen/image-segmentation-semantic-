"""import os
import random
import shutil

# =========================
# PATHS (DO NOT CHANGE)
# =========================
IMAGE_DIR = "images"                      # Kaggle images folder
MASK_DIR = "annotations/trimaps"          # Kaggle masks folder
OUTPUT_DIR = "dataset"                    # Output dataset

# =========================
# SPLIT RATIOS
# =========================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# =========================
# CREATE OUTPUT FOLDERS
# =========================
for split in ["train", "val", "test"]:
    os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/{split}/masks", exist_ok=True)

# =========================
# GET IMAGE FILES
# =========================
images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
images.sort()          # keep deterministic
random.seed(42)
random.shuffle(images)

total = len(images)

# =========================
# COMPUTE SPLITS
# =========================
train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_images = images[:train_end]
val_images = images[train_end:val_end]
test_images = images[val_end:]

# =========================
# COPY FUNCTION
# =========================
def copy_split(file_list, split_name):
    for img_name in file_list:
        mask_name = img_name.replace(".jpg", ".png")

        img_src = os.path.join(IMAGE_DIR, img_name)
        mask_src = os.path.join(MASK_DIR, mask_name)

        img_dst = f"{OUTPUT_DIR}/{split_name}/images/{img_name}"
        mask_dst = f"{OUTPUT_DIR}/{split_name}/masks/{img_name}"

        if not os.path.exists(mask_src):
            print(f"Mask missing for {img_name}, skipping...")
            continue

        shutil.copy(img_src, img_dst)
        shutil.copy(mask_src, mask_dst)

# =========================
# PERFORM SPLIT
# =========================
copy_split(train_images, "train")
copy_split(val_images, "val")
copy_split(test_images, "test")

# =========================
# DONE
# =========================
print("Dataset split completed successfully!")
print(f"Train images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
print(f"Test images: {len(test_images)}")"""
