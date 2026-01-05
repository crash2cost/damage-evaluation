import torch
import torch.nn as nn
import torch.nn.functional as F
class YOLOLoss(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, predictions, targets):
        box_loss = torch.tensor(0.0, device=predictions[0].device)
        cls_loss = torch.tensor(0.0, device=predictions[0].device)
        dfl_loss = torch.tensor(0.0, device=predictions[0].device)
        for pred in predictions:
            batch_size = pred.shape[0]
            box_pred = pred[:, :64, :, :]
            cls_pred = pred[:, 64:, :, :]
            if targets is not None and len(targets) > 0:
                cls_loss += self.bce(cls_pred, torch.zeros_like(cls_pred)).mean()
                box_loss += torch.abs(box_pred).mean() * 0.05
        total_loss = box_loss + cls_loss + dfl_loss
        return {
            'total': total_loss,
            'box': box_loss,
            'cls': cls_loss,
            'dfl': dfl_loss
        }
def bbox_iou(box1, box2, eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    inter_x1 = torch.max(b1_x1, b2_x1.T)
    inter_y1 = torch.max(b1_y1, b2_y1.T)
    inter_x2 = torch.min(b1_x2, b2_x2.T)
    inter_y2 = torch.min(b1_y2, b2_y2.T)
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area.T - inter_area + eps
    iou = inter_area / union_area
    return iou
if __name__ == "__main__":
    loss_fn = YOLOLoss(num_classes=1)
    pred1 = torch.randn(2, 65, 80, 80)
    pred2 = torch.randn(2, 65, 40, 40)
    pred3 = torch.randn(2, 65, 20, 20)
    predictions = [pred1, pred2, pred3]
    targets = []
    losses = loss_fn(predictions, targets)
    print("Loss Function Test:")
    print(f"Total Loss: {losses['total']:.4f}")
    print(f"Box Loss: {losses['box']:.4f}")
    print(f"Class Loss: {losses['cls']:.4f}")
    print(f"DFL Loss: {losses['dfl']:.4f}")