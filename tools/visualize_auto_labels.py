"""
Visualize auto-labeled images to verify quality
Shows images with bounding boxes for manual inspection
"""

import os
import sys
from pathlib import Path
import cv2
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def visualize_labels(dataset_dir, num_samples=20, split='train'):
    """
    Visualize random samples from auto-labeled dataset
    """
    images_dir = Path(dataset_dir) / split / 'images'
    labels_dir = Path(dataset_dir) / split / 'labels'
    
    # Get all images
    all_images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    
    if len(all_images) == 0:
        print(f"No images found in {images_dir}")
        return
    
    # Sample random images
    samples = random.sample(all_images, min(num_samples, len(all_images)))
    
    print(f"📸 Visualizing {len(samples)} random samples from {split} split...")
    print("Press any key to see next image, 'q' to quit\n")
    
    for img_path in samples:
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Load corresponding label
        label_path = labels_dir / img_path.name.replace('.jpg', '.txt').replace('.png', '.txt')
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Draw bounding boxes
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                cls, x_center, y_center, width, height = map(float, parts)
                
                # Convert from YOLO format to pixel coordinates
                x_center *= w
                y_center *= h
                width *= w
                height *= h
                
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label_text = "damage"
                cv2.putText(img, label_text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add filename
        cv2.putText(img, img_path.name, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Resize if too large
        max_height = 800
        if h > max_height:
            scale = max_height / h
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
        
        # Show
        cv2.imshow('Auto-Labeled Dataset - Press any key for next, q to quit', img)
        key = cv2.waitKey(0)
        
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("✓ Visualization complete")

if __name__ == '__main__':
    dataset_dir = '/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/ml-service/detection-model/dataset-expanded'
    
    print("Choose what to visualize:")
    print("1. Train split (auto-labeled)")
    print("2. Val split (auto-labeled)")
    print("3. Both")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        visualize_labels(dataset_dir, num_samples=30, split='train')
    elif choice == '2':
        visualize_labels(dataset_dir, num_samples=20, split='val')
    elif choice == '3':
        visualize_labels(dataset_dir, num_samples=20, split='train')
        visualize_labels(dataset_dir, num_samples=20, split='val')
    else:
        print("Invalid choice")
