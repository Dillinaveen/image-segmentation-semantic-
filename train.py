import torch
from torch.utils.data import DataLoader
from data.dataset import CamVidDataset
from models.deeplab import get_model
from metrics import mean_iou
from tqdm import tqdm

DEVICE = "cpu"
NUM_CLASSES = 11

train_ds = CamVidDataset(
    "dataset/train/images",
    "dataset/train/masks"
)
val_ds = CamVidDataset(
    "dataset/val/images",
    "dataset/val/masks"
)

train_loader = DataLoader(
    train_ds,
    batch_size=2,
    shuffle=True,
    drop_last=True,
    num_workers=0
)

val_loader = DataLoader(
    val_ds,
    batch_size=2,
    num_workers=0
)

model = get_model(NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

for epoch in range(10):
    model.train()

    # 🔒 Freeze BatchNorm every epoch
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()

    train_loss = 0.0

    for imgs, masks in tqdm(train_loader):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)["out"]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)

    model.eval()
    val_iou = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)["out"]
            val_iou += mean_iou(outputs, masks, NUM_CLASSES)

    val_iou /= len(val_loader)

    print(
        f"Epoch {epoch+1}/10 | "
        f"Loss: {avg_loss:.4f} | "
        f"Val mIoU: {val_iou:.4f}"
    )

torch.save(model.state_dict(), "deeplab_camvid.pth")
