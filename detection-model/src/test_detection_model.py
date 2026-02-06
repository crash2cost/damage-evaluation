#!/usr/bin/env python3
"""
Test the trained YOLO detection model
Shows detection results on test images
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_model(model_path, test_images_dir, num_samples=6):
    """Test the model and display results"""
    
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Get test images
    test_dir = Path(test_images_dir)
    image_files = list(test_dir.glob('*.jpg'))[:num_samples]
    
    if not image_files:
        print(f"No images found in {test_images_dir}")
        return
    
    print(f"\nTesting on {len(image_files)} images...")
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, img_path in enumerate(image_files):
        # Run inference
        results = model(str(img_path), verbose=False)
        
        # Load image
        img = Image.open(img_path)
        
        # Display
        axes[idx].imshow(img)
        axes[idx].set_title(f"{img_path.name}\n{len(results[0].boxes)} detections")
        axes[idx].axis('off')
        
        # Draw boxes
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none'
                )
                axes[idx].add_patch(rect)
                
                # Add confidence score
                axes[idx].text(
                    x1, y1 - 5,
                    f'{conf:.2f}',
                    color='red',
                    fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )
        
        print(f"  {img_path.name}: {len(results[0].boxes)} damage areas detected")
    
    plt.tight_layout()
    plt.savefig('detection_results.png', dpi=150, bbox_inches='tight')
    print(f"\n Results saved to: detection_results.png")
    plt.show()

def evaluate_on_test_set(model_path, data_yaml):
    """Run full evaluation on test set"""
    print(f"\n{'='*60}")
    print("Running full test set evaluation...")
    print(f"{'='*60}\n")
    
    model = YOLO(model_path)
    
    # Run validation on test set
    metrics = model.val(
        data=data_yaml,
        split='test',
        batch=32,
        imgsz=416,
        device='mps',
        plots=True,
        save_json=False
    )
    
    print(f"\n{'='*60}")
    print("Test Set Results:")
    print(f"{'='*60}")
    print(f"Precision: {metrics.box.p[0]:.4f}")
    print(f"Recall:    {metrics.box.r[0]:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"{'='*60}\n")
    
    return metrics

if __name__ == "__main__":
    # Paths
    workspace = Path(__file__).parent.parent
    model_path = workspace / "detection-model/runs/roboflow-continuous/weights/best.pt"
    test_images_dir = workspace / "car-damage-detector-1/test/images"
    data_yaml = workspace / "car-damage-detector-1/data.yaml"
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  Testing YOLOv8s Car Damage Detection Model             ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Test on sample images with visualization
    test_model(model_path, test_images_dir, num_samples=6)
    
    # Full test set evaluation
    evaluate_on_test_set(model_path, data_yaml)
