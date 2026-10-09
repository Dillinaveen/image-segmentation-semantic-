import torch
from torch.utils.data import DataLoader
from data.dataset import CamVidDataset
from models.deeplab import get_model
from metrics import mean_iou
DEVICE="cpu"
NUM_CLASSES = 11

model = get_model(NUM_CLASSES)
model.load_state_dict(torch.load("deeplab_camvid.pth", map_location="cpu"))
model.eval()

test_ds = CamVidDataset(
    "dataset/test/images",
    "dataset/test/masks"
)
test_loader = DataLoader(test_ds, batch_size=1)

total_iou = 0
with torch.no_grad():
    for img, mask in test_loader:
        img, mask = img.to(DEVICE), mask.to(DEVICE)
        output = model(img)["out"]
        preds = torch.argmax(output, dim=1) 
        total_iou += mean_iou(preds, mask, NUM_CLASSES)
 
print("Test Mean IoU:", total_iou / len(test_loader))
