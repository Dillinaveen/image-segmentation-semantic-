
import torch

def mean_iou(pred, target, num_classes, eps=1e-6):
    pred = torch.argmax(pred, dim=1)
    iou = 0.0

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union > 0:
            iou += (intersection + eps) / (union + eps)

    return iou / num_classes

