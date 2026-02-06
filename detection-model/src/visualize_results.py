import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from custom_yolo import CustomYOLO
import argparse
import matplotlib.pyplot as plt
import numpy as np
import yaml


def nms(boxes, scores, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove overlapping boxes.
    boxes: list of [x1, y1, x2, y2] (pixel coordinates)
    scores: list of confidence scores
    Returns: indices of boxes to keep
    """
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


def load_model(checkpoint_path, num_classes=1, device='mps'):
    """Load trained model from checkpoint"""
    model = CustomYOLO(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    print(f" Loaded model from {checkpoint_path}")
    print(f"  Best validation loss: {checkpoint['best_loss']:.4f}")
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    return model


def predict_image(model, image_path, device='mps', conf_threshold=0.25, img_size=640, num_classes=1):
    """Run inference on a single image"""
    img_original = Image.open(image_path).convert('RGB')
    
    # Transform for model input
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])
    
    img_tensor = transform(img_original).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(img_tensor)
    
    # Parse predictions - format depends on your model output
    # Model returns 3 prediction heads (small, medium, large)
    detections = []
    
    # Handle different prediction formats - predictions is a list of 3 tensors
    if isinstance(predictions, (list, tuple)):
        # Process each prediction head separately
        for pred in predictions:
            # pred shape: [batch, num_anchors, features]
            pred = pred.squeeze(0)  # Remove batch dimension: [num_anchors, features]
            
            # Get objectness scores (5th element)
            if pred.shape[-1] >= 5:
                objectness = pred[:, 4]
                mask = objectness > conf_threshold
                filtered = pred[mask]
                
                for detection in filtered:
                    x, y, w, h = detection[:4].cpu().numpy()
                    conf = detection[4].cpu().item()
                    # Get class (if multi-class)
                    if num_classes > 1 and detection.shape[0] > 5:
                        class_scores = detection[5:5+num_classes].cpu().numpy()
                        class_id = int(class_scores.argmax())
                    else:
                        class_id = 0
                    detections.append({
                        'bbox': [float(x), float(y), float(w), float(h)],
                        'confidence': conf,
                        'class': class_id
                    })
    else:
        # Single tensor output
        all_preds = predictions.view(-1, predictions.shape[-1])
        
        if all_preds.shape[-1] >= 5:
            objectness = all_preds[:, 4]
            mask = objectness > conf_threshold
            filtered = all_preds[mask]
            
            for detection in filtered:
                x, y, w, h = detection[:4].cpu().numpy()
                conf = detection[4].cpu().item()
                # Get class (if multi-class)
                if num_classes > 1 and detection.shape[0] > 5:
                    class_scores = detection[5:5+num_classes].cpu().numpy()
                    class_id = int(class_scores.argmax())
                else:
                    class_id = 0
                detections.append({
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'confidence': conf,
                    'class': class_id
                })
    
    return detections, img_original


def draw_detections(image, detections, class_names=None, color='red', thickness=3, iou_threshold=0.5):
    """Draw bounding boxes on image with NMS applied"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    print(f"Raw detections: {len(detections)}")
    
    if len(detections) == 0:
        return img
    
    # Convert to pixel coordinates for NMS
    boxes = []
    scores = []
    for det in detections:
        x, y, w, h = det['bbox']
        x_center = x * width
        y_center = y * height
        box_width = w * width
        box_height = h * height
        
        x1 = x_center - box_width / 2
        y1 = y_center - box_height / 2
        x2 = x_center + box_width / 2
        y2 = y_center + box_height / 2
        
        boxes.append([x1, y1, x2, y2])
        scores.append(det['confidence'])
    
    # Apply NMS
    keep_indices = nms(boxes, scores, iou_threshold)
    print(f"After NMS (IoU={iou_threshold}): {len(keep_indices)} detections\n")
    
    # Try to use a nicer font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    for i, idx in enumerate(keep_indices):
        x1, y1, x2, y2 = boxes[idx]
        conf = scores[idx]
        det = detections[idx]
        class_id = det.get('class', 0)
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        
        # Draw label with class name and confidence
        if class_names and class_id < len(class_names):
            class_name = class_names[class_id]
            label = f"{class_name}: {conf*100:.0f}%"
        else:
            label = f"Damage #{i+1}: {conf:.2f}"
        
        # Draw background for text
        bbox = draw.textbbox((x1, y1 - 25), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 25), label, fill='white', font=font)
        
        print(f"  {i+1}. Confidence: {conf:.3f} | Box: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
    
    return img


def visualize_before_after(original_img, detected_img, detections, image_name):
    """Display before and after images side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Before (original)
    ax1.imshow(np.array(original_img))
    ax1.set_title('BEFORE - Original Image', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # After (with detections)
    ax2.imshow(np.array(detected_img))
    ax2.set_title(f'AFTER - Detected {len(detections)} Damage(s)', fontsize=16, fontweight='bold', color='red')
    ax2.axis('off')
    
    # Add overall title
    fig.suptitle(f'Car Damage Detection: {image_name}', fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Test trained YOLO model and visualize results')
    parser.add_argument('--checkpoint', type=str, default='../runs/custom-yolo/best.pt', 
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, 
                        help='Path to test image or directory of images')
    parser.add_argument('--conf-threshold', type=float, default=0.25, 
                        help='Confidence threshold for detections')
    parser.add_argument('--iou-threshold', type=float, default=0.5, 
                        help='IoU threshold for NMS')
    parser.add_argument('--output-dir', type=str, default='../test_results', 
                        help='Directory to save results')
    parser.add_argument('--num-classes', type=int, default=1, 
                        help='Number of classes')
    parser.add_argument('--data', type=str, default=None, 
                        help='Path to data.yaml file with class names')
    parser.add_argument('--show', action='store_true', 
                        help='Display images in matplotlib window')
    
    args = parser.parse_args()
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print("="*70)
    print("Car Damage Detection - Model Testing")
    print("="*70)
    print(f"Device: {device}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print("="*70 + "\n")
    
    # Load class names from data.yaml if provided
    class_names = None
    if args.data:
        data_yaml_path = Path(args.data)
        if data_yaml_path.exists():
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
                class_names = data_config.get('names', None)
                print(f" Loaded {len(class_names)} class names from {args.data}")
                print(f"  Classes: {', '.join(class_names)}\n")
    
    # Load model
    model = load_model(args.checkpoint, num_classes=args.num_classes, device=device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get image(s) to test
    image_path = Path(args.image)
    if image_path.is_dir():
        image_files = list(image_path.glob('*.jpg')) + \
                     list(image_path.glob('*.jpeg')) + \
                     list(image_path.glob('*.png'))
    else:
        image_files = [image_path]
    
    print(f"\nTesting on {len(image_files)} image(s)...\n")
    
    # Process each image
    for img_file in image_files:
        print(f"\n{'='*70}")
        print(f"Processing: {img_file.name}")
        print(f"{'='*70}")
        
        # Run inference
        detections, original_img = predict_image(
            model, img_file, device=device, conf_threshold=args.conf_threshold, num_classes=args.num_classes
        )
        
        # Print results
        print(f"\n Found {len(detections)} damage detection(s):")
        if len(detections) == 0:
            print("  No damage detected above confidence threshold")
        else:
            for i, det in enumerate(detections, 1):
                bbox = det['bbox']
                class_id = det.get('class', 0)
                class_label = class_names[class_id] if class_names and class_id < len(class_names) else f"Class {class_id}"
                print(f"  {i}. {class_label} - Confidence: {det['confidence']:.3f} | "
                      f"BBox: [x={bbox[0]:.3f}, y={bbox[1]:.3f}, w={bbox[2]:.3f}, h={bbox[3]:.3f}]")
        
        # Draw detections with NMS
        detected_img = draw_detections(original_img, detections, class_names=class_names, iou_threshold=args.iou_threshold)
        
        # Print BEFORE
        print("\n" + "="*70)
        print("BEFORE - Original Image")
        print("="*70)
        print(f"  Image size: {original_img.size}")
        print(f"  Format: {original_img.format}")
        print(f"  Mode: {original_img.mode}")
        
        # Print AFTER
        print("\n" + "="*70)
        print(f"AFTER - With {len(detections)} Detection(s)")
        print("="*70)
        print(f"  Bounding boxes drawn: {len(detections)}")
        print(f"  Color: Red")
        print(f"  Line thickness: 3px")
        
        # Save comparison
        comparison_path = output_dir / f"{img_file.stem}_comparison.png"
        fig = visualize_before_after(original_img, detected_img, detections, img_file.name)
        fig.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"\n Saved comparison to: {comparison_path}")
        
        # Save individual images
        before_path = output_dir / f"{img_file.stem}_before.jpg"
        after_path = output_dir / f"{img_file.stem}_after.jpg"
        original_img.save(before_path)
        detected_img.save(after_path)
        print(f" Saved BEFORE to: {before_path}")
        print(f" Saved AFTER to: {after_path}")
        
        # Show if requested
        if args.show:
            plt.show()
        else:
            plt.close(fig)
    
    print("\n" + "="*70)
    print("Testing Complete!")
    print(f"Results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
