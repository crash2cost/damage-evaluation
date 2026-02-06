"""
Auto-label and organize CarDD dataset + classification images
Generates YOLO labels using trained model and creates expanded dataset
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
    def __init__(self, model_path, confidence_threshold=0.05):  # Lowered from 0.25 to 0.05
        self.model = CustomYOLOInference(model_path)
        self.conf_threshold = confidence_threshold
        
    def label_image(self, image_path):
        """Generate YOLO format label for an image"""
        try:
            # predict() returns CustomResults directly, not a list
            result = self.model.predict(image_path, conf=self.conf_threshold)
            
            if result is None:
                return []
            
            # Access boxes attribute directly
            if not hasattr(result, 'boxes') or result.boxes is None:
                return []
            
            boxes = result.boxes
            
            if boxes.xyxy is None or len(boxes.xyxy) == 0:
                return []
            
            labels = []
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            for i in range(len(boxes.xyxy)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = boxes.conf[i].item()
                cls = int(boxes.cls[i].item())
                
                # Convert to YOLO format (normalized)
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                labels.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            return labels
        except Exception as e:
            # Return empty labels if anything fails
            return []

def main():
    # Paths
    cardd_cache = Path('/Users/idolevi/.cache/huggingface/hub/datasets--harpreetsahota--CarDD/snapshots/56900bde8dddfe00eb7c03114a1d46e9105e3cdb/data')
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
    
    print("🚀 Auto-Labeling CarDD + Classification Images...")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}\n")
    
    # Initialize labeler
    labeler = AutoLabeler(model_path, confidence_threshold=0.25)
    
    stats = {
        'total_images': 0,
        'labeled': 0,
        'no_detection': 0,
        'train': 0,
        'val': 0,
        'cardd': 0,
        'classification': 0
    }
    
    # ===== Process CarDD images (80% train, 20% val) =====
    print("📸 Processing CarDD images (2,817 unlabeled)...")
    if cardd_cache.exists():
        cardd_images = list(cardd_cache.glob('*.jpg'))
        
        # Split 80/20
        train_split = int(len(cardd_images) * 0.8)
        
        for idx, img_path in enumerate(tqdm(cardd_images, desc="  CarDD")):
            stats['total_images'] += 1
            stats['cardd'] += 1
            
            # Determine split
            is_train = idx < train_split
            
            # Generate labels
            try:
                labels = labeler.label_image(str(img_path))
            except Exception as e:
                print(f"\n⚠️  Error labeling {img_path.name}: {e}")
                continue
            
            # Output paths
            output_name = f"cardd_{img_path.name}"
            if is_train:
                out_img = output_train_images / output_name
                out_lbl = output_train_labels / output_name.replace('.jpg', '.txt')
                stats['train'] += 1
            else:
                out_img = output_val_images / output_name
                out_lbl = output_val_labels / output_name.replace('.jpg', '.txt')
                stats['val'] += 1
            
            # Copy image
            shutil.copy(img_path, out_img)
            
            # Save label
            if labels:
                with open(out_lbl, 'w') as f:
                    f.write('\n'.join(labels))
                stats['labeled'] += 1
            else:
                out_lbl.touch()
                stats['no_detection'] += 1
    else:
        print(f"  ⚠️  CarDD cache not found at {cardd_cache}")
    
    # ===== Process Classification Dataset =====
    print("\n📂 Processing classification images (1,048 images)...")
    damage_types = ['bumper_dent', 'bumper_scratch', 'door_dent', 'door_scratch', 
                    'glass_shatter', 'head_lamp', 'tail_lamp']
    
    # Train split
    train_dir = classification_dataset / 'train'
    for damage_type in damage_types:
        damage_folder = train_dir / damage_type
        if not damage_folder.exists():
            continue
        
        images = list(damage_folder.glob('*.jpg')) + list(damage_folder.glob('*.png')) + list(damage_folder.glob('*.jpeg'))
        
        for img_path in tqdm(images, desc=f"  {damage_type}"):
            stats['total_images'] += 1
            stats['classification'] += 1
            
            try:
                labels = labeler.label_image(str(img_path))
            except Exception as e:
                print(f"\n⚠️  Error labeling {img_path.name}: {e}")
                continue
            
            output_name = f"{damage_type}_{img_path.name}"
            shutil.copy(img_path, output_train_images / output_name)
            
            label_file = output_train_labels / output_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
            if labels:
                with open(label_file, 'w') as f:
                    f.write('\n'.join(labels))
                stats['labeled'] += 1
            else:
                label_file.touch()
                stats['no_detection'] += 1
            
            stats['train'] += 1
    
    # Val split
    val_dir = classification_dataset / 'val'
    for damage_type in damage_types:
        damage_folder = val_dir / damage_type
        if not damage_folder.exists():
            continue
        
        images = list(damage_folder.glob('*.jpg')) + list(damage_folder.glob('*.png')) + list(damage_folder.glob('*.jpeg'))
        
        for img_path in tqdm(images, desc=f"  {damage_type}"):
            stats['total_images'] += 1
            stats['classification'] += 1
            
            try:
                labels = labeler.label_image(str(img_path))
            except Exception as e:
                print(f"\n⚠️  Error labeling {img_path.name}: {e}")
                continue
            
            output_name = f"{damage_type}_{img_path.name}"
            shutil.copy(img_path, output_val_images / output_name)
            
            label_file = output_val_labels / output_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
            if labels:
                with open(label_file, 'w') as f:
                    f.write('\n'.join(labels))
                stats['labeled'] += 1
            else:
                label_file.touch()
                stats['no_detection'] += 1
            
            stats['val'] += 1
    
    # ===== Merge original labeled data =====
    print("\n📋 Merging original labeled dataset...")
    existing_dataset = Path('/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/machines/detection-model/dataset')
    
    existing_train_imgs = existing_dataset / 'train' / 'images'
    existing_train_labels = existing_dataset / 'train' / 'labels'
    if existing_train_imgs.exists():
        for img in existing_train_imgs.glob('*'):
            shutil.copy(img, output_train_images / f"original_{img.name}")
            stats['train'] += 1
            stats['total_images'] += 1
        for lbl in existing_train_labels.glob('*'):
            shutil.copy(lbl, output_train_labels / f"original_{lbl.name}")
        print(f"  ✓ Added {len(list(existing_train_imgs.glob('*')))} original train images")
    
    existing_val_imgs = existing_dataset / 'valid' / 'images'
    existing_val_labels = existing_dataset / 'valid' / 'labels'
    if existing_val_imgs.exists():
        for img in existing_val_imgs.glob('*'):
            shutil.copy(img, output_val_images / f"original_{img.name}")
            stats['val'] += 1
            stats['total_images'] += 1
        for lbl in existing_val_labels.glob('*'):
            shutil.copy(lbl, output_val_labels / f"original_{lbl.name}")
        print(f"  ✓ Added {len(list(existing_val_imgs.glob('*')))} original val images")
    
    # Create data.yaml
    yaml_content = f"""# Expanded Car Damage Detection Dataset
# Auto-labeled CarDD (2,817) + Classification (1,048) + Original (58)

path: {output_dir}
train: train/images
val: val/images

nc: 1
names: ['damage']
"""
    
    with open(output_dir / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    
    # Summary
    print("\n" + "="*70)
    print("✅ AUTO-LABELING COMPLETE!")
    print("="*70)
    print(f"Total images: {stats['total_images']}")
    print(f"  - CarDD dataset: {stats['cardd']}")
    print(f"  - Classification dataset: {stats['classification']}")
    print(f"  - Original labeled: 69")
    print(f"\nDetection results:")
    if stats['total_images'] > 0:
        print(f"  - With detections: {stats['labeled']} ({stats['labeled']/stats['total_images']*100:.1f}%)")
        print(f"  - No detections: {stats['no_detection']} ({stats['no_detection']/stats['total_images']*100:.1f}%)")
    print(f"\nSplit:")
    print(f"  - Train: {stats['train']} images")
    print(f"  - Val: {stats['val']} images")
    print(f"\n📁 Dataset location: {output_dir}")
    print(f"📄 Config: {output_dir / 'data.yaml'}")
    print("\n💡 Ready to train with:")
    print(f"   cd detection-model/src")
    print(f"   python3 custom_train.py --data ../../detection-model/dataset-expanded/data.yaml --epochs 100 --batch-size 8 --lr 0.0005 --num-classes 1 --workers 0")

if __name__ == '__main__':
    main()
