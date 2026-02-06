#!/usr/bin/env python3
"""
Generate confusion matrix and detailed metrics for YOLO detection model
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection area
    intersect_min_x = max(x1_min, x2_min)
    intersect_min_y = max(y1_min, y2_min)
    intersect_max_x = min(x1_max, x2_max)
    intersect_max_y = min(y1_max, y2_max)
    
    if intersect_max_x < intersect_min_x or intersect_max_y < intersect_min_y:
        return 0.0
    
    intersect_area = (intersect_max_x - intersect_min_x) * (intersect_max_y - intersect_min_y)
    
    # Union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersect_area
    
    return intersect_area / union_area if union_area > 0 else 0.0

def generate_confusion_matrix(model_path, test_images_dir, labels_dir, iou_threshold=0.5, conf_threshold=0.25):
    """Generate confusion matrix for object detection"""
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Get all test images
    test_dir = Path(test_images_dir)
    image_files = sorted(test_dir.glob('*.jpg'))
    
    print(f"Processing {len(image_files)} test images...")
    
    # Counters
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_gt_boxes = 0
    total_pred_boxes = 0
    
    # Per-image stats
    images_with_damage = 0
    images_without_damage = 0
    correctly_detected_images = 0
    
    iou_scores = []
    
    for idx, img_path in enumerate(image_files):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(image_files)} images...")
        
        # Load ground truth labels
        label_file = Path(labels_dir) / (img_path.stem + '.txt')
        gt_boxes = []
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO format: class x_center y_center width height (normalized)
                        _, x_c, y_c, w, h = map(float, parts[:5])
                        # Convert to absolute coordinates (assuming we know image size)
                        # We'll use normalized coords for IoU calculation
                        x_min = x_c - w/2
                        y_min = y_c - h/2
                        x_max = x_c + w/2
                        y_max = y_c + h/2
                        gt_boxes.append([x_min, y_min, x_max, y_max])
        
        total_gt_boxes += len(gt_boxes)
        if len(gt_boxes) > 0:
            images_with_damage += 1
        else:
            images_without_damage += 1
        
        # Run inference
        results = model(str(img_path), verbose=False, conf=conf_threshold)
        
        pred_boxes = []
        if len(results[0].boxes) > 0:
            # Get normalized coordinates
            boxes_xyxy = results[0].boxes.xyxyn.cpu().numpy()
            pred_boxes = boxes_xyxy.tolist()
        
        total_pred_boxes += len(pred_boxes)
        
        # Match predictions with ground truth
        matched_gt = set()
        matched_pred = set()
        
        for pred_idx, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx != -1:
                # True positive
                true_positives += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
                iou_scores.append(best_iou)
            else:
                # False positive
                false_positives += 1
        
        # Unmatched ground truth boxes are false negatives
        false_negatives += len(gt_boxes) - len(matched_gt)
        
        # Check if image was correctly detected
        if len(gt_boxes) > 0 and len(matched_gt) == len(gt_boxes) and len(pred_boxes) == len(gt_boxes):
            correctly_detected_images += 1
    
    print(f"\n{'='*60}")
    print("Detection Statistics:")
    print(f"{'='*60}")
    print(f"Total images:              {len(image_files)}")
    print(f"Images with damage:        {images_with_damage}")
    print(f"Images without damage:     {images_without_damage}")
    print(f"Correctly detected images: {correctly_detected_images}")
    print(f"\nTotal ground truth boxes:  {total_gt_boxes}")
    print(f"Total predicted boxes:     {total_pred_boxes}")
    print(f"\nTrue Positives:            {true_positives}")
    print(f"False Positives:           {false_positives}")
    print(f"False Negatives:           {false_negatives}")
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_iou = np.mean(iou_scores) if iou_scores else 0
    
    print(f"\nPrecision:                 {precision:.4f}")
    print(f"Recall:                    {recall:.4f}")
    print(f"F1 Score:                  {f1_score:.4f}")
    print(f"Average IoU (TP):          {avg_iou:.4f}")
    print(f"{'='*60}\n")
    
    # Create confusion matrix visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box-level confusion matrix
    cm_boxes = np.array([
        [true_positives, false_positives],
        [false_negatives, 0]
    ])
    
    # Image-level confusion matrix
    cm_images = np.array([
        [correctly_detected_images, images_with_damage - correctly_detected_images],
        [0, images_without_damage]
    ])
    
    # Plot box-level
    sns.heatmap(cm_boxes, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Damage', 'Background'],
                yticklabels=['Damage', 'Background'],
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title(f'Box-Level Confusion Matrix\n(IoU threshold: {iou_threshold})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=11)
    axes[0].set_ylabel('Actual', fontsize=11)
    
    # Add metrics text
    metrics_text = f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1 Score: {f1_score:.3f}\nAvg IoU: {avg_iou:.3f}'
    axes[0].text(1.05, 0.5, metrics_text, transform=axes[0].transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot image-level
    sns.heatmap(cm_images, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Correct', 'Incorrect'],
                yticklabels=['With Damage', 'No Damage'],
                ax=axes[1], cbar_kws={'label': 'Count'})
    axes[1].set_title('Image-Level Detection Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Detection Result', fontsize=11)
    axes[1].set_ylabel('Ground Truth', fontsize=11)
    
    # Add accuracy text
    img_accuracy = correctly_detected_images / images_with_damage if images_with_damage > 0 else 0
    accuracy_text = f'Accuracy: {img_accuracy:.3f}\n({correctly_detected_images}/{images_with_damage})'
    axes[1].text(1.05, 0.5, accuracy_text, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print(f" Confusion matrix saved to: confusion_matrix.png")
    plt.show()
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_iou': avg_iou
    }

if __name__ == "__main__":
    # Paths
    workspace = Path(__file__).parent.parent
    model_path = workspace / "detection-model/runs/roboflow-continuous/weights/best.pt"
    test_images_dir = workspace / "car-damage-detector-1/test/images"
    test_labels_dir = workspace / "car-damage-detector-1/test/labels"
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  Confusion Matrix for Car Damage Detection              ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    results = generate_confusion_matrix(
        model_path,
        test_images_dir,
        test_labels_dir,
        iou_threshold=0.5,
        conf_threshold=0.25
    )
