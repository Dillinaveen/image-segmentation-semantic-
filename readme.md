DeepLabV3 Image Segmentation

Overview

This project implements semantic image segmentation using the DeepLabV3 deep learning architecture with a custom dataset. The model predicts a class label for each pixel in an input image, enabling pixel-level understanding of visual scenes.

Features

- Semantic image segmentation using DeepLabV3

- Custom dataset support with training, validation, and test splits

- Model training and evaluation

- Mask processing and dataset preparation utilities

- Performance evaluation using IoU, Mean IoU (mIoU), and Dice coefficient

Technologies Used

Python

PyTorch

Torchvision

NumPy

OpenCV / Pillow

Matplotlib

Dataset

The dataset is organized into three splits:

Train: Used to train the segmentation model.

Validation: Used to monitor model performance during training and tune parameters.

Test: Used for final evaluation on unseen data.

Ensure that each image is paired with its corresponding segmentation mask and that the image-mask dimensions and filenames follow the format expected by the data-loading code.