"""
Auto-Label Remaining Images using YOLO
Train on your 100+ labeled images, then predict on the rest
"""

import json
from pathlib import Path
import shutil
from ultralytics import YOLO
import torch

# Configuration
IMAGE_DIR = Path("detection-model/dataset-multiclass/train/images")
LABEL_DIR = Path("detection-model/real_labels")
OUTPUT_DIR = Path("detection-model/auto_predicted_labels")
TRAINING_DIR = Path("detection-model/training_temp")

CLASSES = [
    "hood_dent", "hood_scratch", "hood_crack",
    "bumper_dent", "bumper_scratch", "bumper_crack", 
    "door_dent", "door_scratch", "door_crack",
    "fender_dent", "fender_scratch", "fender_crack",
    "trunk_dent", "trunk_scratch", "trunk_crack",
    "glass_crack",
    "headlight_crack", "headlight_broken",
    "taillight_crack", "taillight_broken",
    "mirror_crack", "mirror_broken"
]

def prepare_training_data():
    """Convert your JSON labels to YOLO format for training"""
    print("📦 Preparing training data...")
    
    # Create training directory structure
    train_images = TRAINING_DIR / "images" / "train"
    train_labels = TRAINING_DIR / "labels" / "train"
    val_images = TRAINING_DIR / "images" / "val"
    val_labels = TRAINING_DIR / "labels" / "val"
    
    for d in [train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Get all labeled images
    labeled_files = []
    for json_file in LABEL_DIR.glob("*.json"):
        img_name = json_file.stem + ".jpg"  # Keep the full name including auto_ prefix
        img_path = IMAGE_DIR / img_name
        if img_path.exists():
            labeled_files.append((img_path, json_file))
    
    print(f"Found {len(labeled_files)} labeled images")
    
    # Split 80/20 train/val
    split_idx = int(len(labeled_files) * 0.8)
    train_files = labeled_files[:split_idx]
    val_files = labeled_files[split_idx:]
    
    # Process training files
    for img_path, json_path in train_files:
        # Copy image
        shutil.copy(img_path, train_images / img_path.name)
        
        # Convert JSON to YOLO format
        with open(json_path) as f:
            data = json.load(f)
        
        # Write YOLO label file
        label_path = train_labels / (img_path.stem + ".txt")
        with open(label_path, 'w') as f:
            for box in data.get('boxes', []):
                class_id = box['class']
                x, y, w, h = box['x'], box['y'], box['w'], box['h']
                f.write(f"{class_id} {x} {y} {w} {h}\n")
    
    # Process validation files
    for img_path, json_path in val_files:
        shutil.copy(img_path, val_images / img_path.name)
        
        with open(json_path) as f:
            data = json.load(f)
        
        label_path = val_labels / (img_path.stem + ".txt")
        with open(label_path, 'w') as f:
            for box in data.get('boxes', []):
                class_id = box['class']
                x, y, w, h = box['x'], box['y'], box['w'], box['h']
                f.write(f"{class_id} {x} {y} {w} {h}\n")
    
    # Create data.yaml
    yaml_content = f"""# Auto-generated training config
path: {TRAINING_DIR.absolute()}
train: images/train
val: images/val

nc: {len(CLASSES)}
names: {CLASSES}
"""
    
    yaml_path = TRAINING_DIR / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Training: {len(train_files)} images")
    print(f"✅ Validation: {len(val_files)} images")
    
    return yaml_path


def train_model():
    """Train YOLO model on your labeled data"""
    print("\n🚀 Training YOLO model on your labels...")
    
    yaml_path = prepare_training_data()
    
    # Load pretrained YOLO model
    model = YOLO('yolov8n.pt')  # Start with nano model (fast training)
    
    # Train
    results = model.train(
        data=str(yaml_path),
        epochs=50,  # Can increase if you want better results
        imgsz=640,
        batch=8,
        patience=10,  # Early stopping
        save=True,
        device='mps' if torch.backends.mps.is_available() else 'cpu',
        project='detection-model/runs',
        name='auto_label_model',
        exist_ok=True
    )
    
    print("✅ Training complete!")
    return model


def predict_unlabeled():
    """Use trained model to predict labels for remaining images"""
    print("\n🎯 Predicting labels for unlabeled images...")
    
    # Load the best trained model
    model_path = Path("detection-model/runs/auto_label_model/weights/best.pt")
    if not model_path.exists():
        print("❌ No trained model found. Train first!")
        return
    
    model = YOLO(str(model_path))
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Get all images
    all_images = list(IMAGE_DIR.glob("*.jpg"))
    
    # Get already labeled images
    labeled_names = {f.stem for f in LABEL_DIR.glob("*.json")}  # Keep full names
    
    # Filter to unlabeled only
    unlabeled_images = [img for img in all_images if img.stem not in labeled_names]
    
    print(f"Found {len(unlabeled_images)} unlabeled images")
    
    # Predict in batches
    predicted_count = 0
    for img_path in unlabeled_images:
        results = model(img_path, verbose=False)
        
        # Convert to your JSON format
        boxes = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Get normalized coordinates
                    x_center = float((box.xywhn[0][0]))
                    y_center = float((box.xywhn[0][1]))
                    width = float((box.xywhn[0][2]))
                    height = float((box.xywhn[0][3]))
                    
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Only save if confidence is high enough
                    if confidence > 0.3:
                        boxes.append({
                            "class": class_id,
                            "x": x_center,
                            "y": y_center,
                            "w": width,
                            "h": height,
                            "severity": 3,  # Default severity
                            "confidence": confidence
                        })
        
        # Save prediction
        if boxes:  # Only save if we found damage
            output_file = OUTPUT_DIR / f"{img_path.stem}.json"  # Use original name
            with open(output_file, 'w') as f:
                json.dump({"boxes": boxes}, f, indent=2)
            predicted_count += 1
    
    print(f"✅ Generated predictions for {predicted_count} images")
    print(f"📁 Saved to: {OUTPUT_DIR}")
    print("\n💡 Tip: Review predictions in annotation tool and move good ones to real_labels/")


def main():
    """Main workflow"""
    print("🏗️  Auto-Labeling Pipeline")
    print("=" * 50)
    
    choice = input("\nWhat would you like to do?\n1. Train model on your labels\n2. Predict on unlabeled images (requires trained model)\n3. Both (train then predict)\n\nChoice (1/2/3): ").strip()
    
    if choice == "1":
        train_model()
    elif choice == "2":
        predict_unlabeled()
    elif choice == "3":
        train_model()
        predict_unlabeled()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
