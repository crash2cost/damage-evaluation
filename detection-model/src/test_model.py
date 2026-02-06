import torch
from pathlib import Path
from PIL import Image, ImageDraw
import torchvision.transforms as T
from custom_yolo import CustomYOLO
import sys

def load_model(checkpoint_path, num_classes=1):
    """Load trained model from checkpoint"""
    model = CustomYOLO(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict_image(model, image_path, device='mps', conf_threshold=0.25):
    """Run inference on a single image"""
    img = Image.open(image_path).convert('RGB')
    
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(img_tensor)
    
    # predictions format: [batch, num_anchors, (x, y, w, h, objectness, class_probs...)]
    # Filter by confidence
    detections = []
    for pred in predictions:
        # pred shape: [num_anchors, 5 + num_classes]
        objectness = pred[:, 4]
        mask = objectness > conf_threshold
        filtered = pred[mask]
        
        for detection in filtered:
            x, y, w, h = detection[:4]
            conf = detection[4].item()
            detections.append({
                'bbox': [x.item(), y.item(), w.item(), h.item()],
                'confidence': conf
            })
    
    return detections, img

def draw_detections(image, detections):
    """Draw bounding boxes on image"""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    for det in detections:
        x, y, w, h = det['bbox']
        # Convert from normalized YOLO format to pixel coordinates
        x_center = x * width
        y_center = y * height
        box_width = w * width
        box_height = h * height
        
        x1 = x_center - box_width / 2
        y1 = y_center - box_height / 2
        x2 = x_center + box_width / 2
        y2 = y_center + box_height / 2
        
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        draw.text((x1, y1 - 10), f"Damage {det['confidence']:.2f}", fill='red')
    
    return image

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <image_path> [checkpoint_path]")
        print("Example: python test_model.py ../dataset-final/val/images/000001.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else '../runs/custom-yolo/best.pt'
    
    print(f"Loading model from: {checkpoint_path}")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = load_model(checkpoint_path, num_classes=1)
    model.to(device)
    
    print(f"Running inference on: {image_path}")
    detections, img = predict_image(model, image_path, device=device)
    
    print(f"\nFound {len(detections)} damage detection(s):")
    for i, det in enumerate(detections, 1):
        print(f"  {i}. Confidence: {det['confidence']:.3f}, BBox: {det['bbox']}")
    
    # Save result
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_detected.jpg"
    img_with_boxes = draw_detections(img, detections)
    img_with_boxes.save(output_path)
    print(f"\nSaved visualization to: {output_path}")

if __name__ == "__main__":
    main()
