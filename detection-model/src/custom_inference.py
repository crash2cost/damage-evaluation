import torch
import torch.nn as nn
import torchvision
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import numpy as np
from custom_yolo import CustomYOLO
class CustomYOLOInference:
    def __init__(self, weights_path, num_classes=1, device=None):
        self.num_classes = num_classes
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CustomYOLO(num_classes=num_classes)
        checkpoint = torch.load(weights_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Loaded custom YOLO model from {weights_path}")
    def preprocess(self, img, img_size=640):
        if isinstance(img, str) or isinstance(img, Path):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        original_size = img.size
        transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0)
        return img_tensor.to(self.device), original_size
    def decode_predictions(self, predictions, conf_threshold=0.25, iou_threshold=0.45, original_size=None, img_size=640):
        all_boxes = []
        all_scores = []
        all_classes = []
        for pred in predictions:
            batch_size, channels, h, w = pred.shape
            pred = pred.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)
            box_pred = pred[..., :4]
            obj_pred = pred[..., 4:5]
            if self.num_classes > 1:
                cls_pred = pred[..., 5:]
            else:
                cls_pred = torch.ones_like(obj_pred)
            obj_scores = torch.sigmoid(obj_pred)
            if self.num_classes > 1:
                cls_scores, cls_indices = torch.max(torch.sigmoid(cls_pred), dim=-1, keepdim=True)
            else:
                cls_scores = torch.ones_like(obj_scores)
                cls_indices = torch.zeros_like(obj_scores).long()
            scores = obj_scores * cls_scores
            mask = scores.squeeze(-1) > conf_threshold
            if mask.sum() > 0:
                valid_boxes = box_pred[mask]
                valid_scores = scores[mask]
                valid_classes = cls_indices[mask]
                cx, cy, w, h = valid_boxes[..., 0], valid_boxes[..., 1], valid_boxes[..., 2], valid_boxes[..., 3]
                cx = torch.sigmoid(cx)
                cy = torch.sigmoid(cy)
                w = torch.sigmoid(w)
                h = torch.sigmoid(h)
                x1 = (cx - w/2)
                y1 = (cy - h/2)
                x2 = (cx + w/2)
                y2 = (cy + h/2)
                boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                all_boxes.append(boxes)
                all_scores.append(valid_scores)
                all_classes.append(valid_classes)
        if len(all_boxes) == 0:
            return []
        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0).squeeze(-1)
        all_classes = torch.cat(all_classes, dim=0).squeeze(-1)
        keep_indices = torchvision.ops.nms(all_boxes, all_scores, iou_threshold)
        final_boxes = all_boxes[keep_indices]
        final_scores = all_scores[keep_indices]
        final_classes = all_classes[keep_indices]
        if original_size is not None:
            orig_w, orig_h = original_size
            final_boxes[:, [0, 2]] *= orig_w
            final_boxes[:, [1, 3]] *= orig_h
        else:
            final_boxes *= img_size
        detections = []
        for box, score, cls in zip(final_boxes, final_scores, final_classes):
            detections.append([
                box[0].item(), box[1].item(), box[2].item(), box[3].item(),
                score.item(), cls.item()
            ])
        return detections
    def predict(self, source, conf=0.25, iou=0.45, img_size=640):
        img_tensor, original_size = self.preprocess(source, img_size)
        with torch.no_grad():
            predictions = self.model(img_tensor)
        detections = self.decode_predictions(
            predictions, conf, iou, original_size, img_size
        )
        return CustomResults(detections, original_size, source)
    def __call__(self, source, **kwargs):
        return [self.predict(source, **kwargs)]
class CustomResults:
    def __init__(self, detections, original_size, source):
        self.detections = detections
        self.original_size = original_size
        self.source = source
        if len(detections) > 0:
            self.boxes = CustomBoxes(detections)
        else:
            self.boxes = CustomBoxes([])
    def __len__(self):
        return len(self.detections)
class CustomBoxes:
    def __init__(self, detections):
        self.detections = detections
        if len(detections) > 0:
            boxes_list = [[d[0], d[1], d[2], d[3]] for d in detections]
            conf_list = [d[4] for d in detections]
            cls_list = [d[5] for d in detections]
            self.xyxy = torch.tensor(boxes_list)
            self.conf = torch.tensor(conf_list)
            self.cls = torch.tensor(cls_list)
            x1, y1, x2, y2 = self.xyxy.T
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            self.xywh = torch.stack([cx, cy, w, h], dim=1)
        else:
            self.xyxy = torch.zeros((0, 4))
            self.conf = torch.zeros(0)
            self.cls = torch.zeros(0)
            self.xywh = torch.zeros((0, 4))
    def __len__(self):
        return len(self.xyxy)
def load_custom_yolo(weights_path, num_classes=1):
    return CustomYOLOInference(weights_path, num_classes)