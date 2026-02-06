"""
Semi-Automated Dataset Labeling
Uses trained custom YOLO model to generate labels for classification images
"""

import os
import sys
import torch
from pathlib import Path
from PIL import Image
import shutil
from tqdm import tqdm

# Add detection model to path
sys.path.append('/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/machines/detection-model/src')
from custom_inference import CustomYOLOInference

class AutoLabeler:
    def __init__(self, model_path, confidence_threshold=0.3):
        """
        Args:
            model_path: Path to trained YOLO weights
            confidence_threshold: Minimum confidence to accept detection (lower = more boxes)
        """
        self.model = CustomYOLOInference(model_path)
        self.conf_threshold = confidence_threshold
        
    def label_image(self, image_path):
        """
        Generate YOLO format label for an image
        Returns: List of label lines in YOLO format: [class_id, x_center, y_center, width, height]
        """
        results = self.model.predict(image_path, conf=self.conf_threshold)
        
        if len(results) == 0:
            return []
        
        result = results[0]
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            return []
        
        labels = []
        # Get image dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        for i in range(len(boxes)):
            # Get box coordinates (xyxy format)
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            conf = boxes.conf[i].item()
            cls = int(boxes.cls[i].item())
            
            # Convert to YOLO format (normalized center coordinates + width/height)
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            # YOLO format: class_id x_center y_center width height
            labels.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        return labels

def main():
    # Paths
    classification_dataset = Path('/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/machines/regression-model/dataset')
    model_path = '/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/machines/detection-model/runs/custom-yolo/best.pt'
    output_dir = Path('/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/machines/detection-model/dataset-expanded')
    
    # Create output directories
    output_train_images = output_dir / 'train' / 'images'
    output_train_labels = output_dir / 'train' / 'labels'
    output_val_images = output_dir / 'val' / 'images'
    output_val_labels = output_dir / 'val' / 'labels'
    
    for dir_path in [output_train_images, output_train_labels, output_val_images, output_val_labels]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting Semi-Automated Labeling...")
    print(f"Model: {model_path}")
    print(f"Source: {classification_dataset}")
    print(f"Output: {output_dir}\n")
    
    # Initialize labeler
    labeler = AutoLabeler(model_path, confidence_threshold=0.25)
    
    # Get all damage type folders
    damage_types = ['bumper_dent', 'bumper_scratch', 'door_dent', 'door_scratch', 
                    'glass_shatter', 'head_lamp', 'tail_lamp']
    
    # Stats
    stats = {
        'total_images': 0,
        'labeled': 0,
        'no_detection': 0,
        'train': 0,
        'val': 0
    }
    
    # Process train split
    print("📂 Processing train images...")
    train_dir = classification_dataset / 'train'
    for damage_type in damage_types:
        damage_folder = train_dir / damage_type
        if not damage_folder.exists():
            continue
        
        images = list(damage_folder.glob('*.jpg')) + list(damage_folder.glob('*.png')) + list(damage_folder.glob('*.jpeg'))
        
        for img_path in tqdm(images, desc=f"  {damage_type}"):
            stats['total_images'] += 1
            
            # Generate labels
            labels = labeler.label_image(str(img_path))
            
            # Save image and label
            output_name = f"{damage_type}_{img_path.name}"
            
            # Copy image
            shutil.copy(img_path, output_train_images / output_name)
            
            # Save label
            label_file = output_train_labels / output_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
            if labels:
                with open(label_file, 'w') as f:
                    f.write('\n'.join(labels))
                stats['labeled'] += 1
            else:
                # Create empty label file (no detections)
                label_file.touch()
                stats['no_detection'] += 1
            
            stats['train'] += 1
    
    # Process val split
    print("\n📂 Processing val images...")
    val_dir = classification_dataset / 'val'
    for damage_type in damage_types:
        damage_folder = val_dir / damage_type
        if not damage_folder.exists():
            continue
        
        images = list(damage_folder.glob('*.jpg')) + list(damage_folder.glob('*.png')) + list(damage_folder.glob('*.jpeg'))
        
        for img_path in tqdm(images, desc=f"  {damage_type}"):
            stats['total_images'] += 1
            
            # Generate labels
            labels = labeler.label_image(str(img_path))
            
            # Save image and label
            output_name = f"{damage_type}_{img_path.name}"
            
            # Copy image
            shutil.copy(img_path, output_val_images / output_name)
            
            # Save label
            label_file = output_val_labels / output_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
            if labels:
                with open(label_file, 'w') as f:
                    f.write('\n'.join(labels))
                stats['labeled'] += 1
            else:
                label_file.touch()
                stats['no_detection'] += 1
            
            stats['val'] += 1
    
    # Copy existing labeled data
    print("\n📋 Merging with existing labeled dataset...")
    existing_dataset = Path('/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/machines/detection-model/dataset')
    
    # Copy existing train data
    existing_train_imgs = existing_dataset / 'train' / 'images'
    existing_train_labels = existing_dataset / 'train' / 'labels'
    if existing_train_imgs.exists():
        for img in existing_train_imgs.glob('*'):
            shutil.copy(img, output_train_images / f"original_{img.name}")
        for lbl in existing_train_labels.glob('*'):
            shutil.copy(lbl, output_train_labels / f"original_{lbl.name}")
        print(f"  ✓ Copied {len(list(existing_train_imgs.glob('*')))} original train images")
    
    # Copy existing val data
    existing_val_imgs = existing_dataset / 'valid' / 'images'
    existing_val_labels = existing_dataset / 'valid' / 'labels'
    if existing_val_imgs.exists():
        for img in existing_val_imgs.glob('*'):
            shutil.copy(img, output_val_images / f"original_{img.name}")
        for lbl in existing_val_labels.glob('*'):
            shutil.copy(lbl, output_val_labels / f"original_{lbl.name}")
        print(f"  ✓ Copied {len(list(existing_val_imgs.glob('*')))} original val images")
    
    # Create data.yaml
    yaml_content = f"""# Expanded Car Damage Detection Dataset
# Auto-labeled from classification dataset + original labeled data

path: {output_dir}
train: train/images
val: val/images

# Classes
nc: 1  # number of classes
names: ['damage']  # class names
"""
    
    with open(output_dir / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    
    # Print summary
    print("\n" + "="*60)
    print("✅ LABELING COMPLETE!")
    print("="*60)
    print(f"Total images processed: {stats['total_images']}")
    if stats['total_images'] > 0:
        print(f"  - With detections: {stats['labeled']} ({stats['labeled']/stats['total_images']*100:.1f}%)")
        print(f"  - No detections: {stats['no_detection']} ({stats['no_detection']/stats['total_images']*100:.1f}%)")
    else:
        print(f"  - With detections: {stats['labeled']}")
        print(f"  - No detections: {stats['no_detection']}")
    print(f"\nSplit:")
    print(f"  - Train: {stats['train']} images")
    print(f"  - Val: {stats['val']} images")
    print(f"\nOriginal labeled data merged: Yes")
    print(f"\nDataset saved to: {output_dir}")
    print(f"Config file: {output_dir / 'data.yaml'}")
    print("\n⚠️  IMPORTANT: Review a sample of auto-labeled images before training!")
    print("   Some labels may be incorrect and need manual adjustment.")

if __name__ == '__main__':
    main()
