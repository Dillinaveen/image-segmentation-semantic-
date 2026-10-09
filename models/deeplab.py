import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

def get_model(num_classes=11):
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, num_classes, 1)
    return model
