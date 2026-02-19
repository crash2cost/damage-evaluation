"""
Continuous Training Pipeline - Runs Overnight
Iteratively improves the model using semi-supervised learning:
1. Train on labeled data
2. Predict on unlabeled data
3. Add high-confidence predictions to training set
4. Retrain with expanded dataset
5. Repeat until convergence or max iterations
"""

import json
from pathlib import Path
import shutil
from ultralytics import YOLO
import torch
from datetime import datetime
import time
import sys

# Configuration
IMAGE_DIR = Path("detection-model/dataset-multiclass/train/images")
MANUAL_LABELS = Path("detection-model/real_labels")
AUTO_LABELS = Path("detection-model/auto_predicted_labels")
TRAINING_DIR = Path("detection-model/training_temp")
RUNS_DIR = Path("detection-model/runs")

CLASSES = [
    "bumper_dent", "bumper_scratch", "door_dent", "door_scratch",
    "glass_shatter", "head_lamp", "tail_lamp",
]

# Training parameters
MAX_ITERATIONS = 20
CONFIDENCE_THRESHOLD = 0.5  # Start stricter for quality
MIN_CONFIDENCE = 0.3  # Minimum confidence to accept auto-labels
EPOCHS_PER_ITERATION = 50
BATCH_SIZE = 16

AUTO_LABELS.mkdir(exist_ok=True)


def log(message):
    """Print with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()  # Force immediate output


def get_all_labels():
    """Get all current labels (manual + auto)"""
    all_labels = list(MANUAL_LABELS.glob("*.json"))
    all_labels.extend(AUTO_LABELS.glob("*.json"))
    return all_labels


def prepare_training_data():
    """Prepare training data from all available labels"""
    log("📦 Preparing training data...")
    
    # Clear previous training data
    if TRAINING_DIR.exists():
        shutil.rmtree(TRAINING_DIR)
    
    train_images = TRAINING_DIR / "images" / "train"
    train_labels = TRAINING_DIR / "labels" / "train"
    val_images = TRAINING_DIR / "images" / "val"
    val_labels = TRAINING_DIR / "labels" / "val"
    
    for d in [train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Get all labeled images
    labeled_files = []
    for json_file in get_all_labels():
        img_name = json_file.stem + ".jpg"
        img_path = IMAGE_DIR / img_name
        if img_path.exists():
            labeled_files.append((img_path, json_file))
    
    log(f"Found {len(labeled_files)} labeled images")
    
    # Split 85/15 train/val (more data for training)
    split_idx = int(len(labeled_files) * 0.85)
    train_files = labeled_files[:split_idx]
    val_files = labeled_files[split_idx:]
    
    # Process training files
    for img_path, json_path in train_files:
        shutil.copy(img_path, train_images / img_path.name)
        
        with open(json_path) as f:
            data = json.load(f)
        
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
    yaml_content = f"""path: {TRAINING_DIR.absolute()}
train: images/train
val: images/val

nc: {len(CLASSES)}
names: {CLASSES}
"""
    
    yaml_path = TRAINING_DIR / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    log(f"✅ Training: {len(train_files)} | Validation: {len(val_files)}")
    return yaml_path, len(labeled_files)


def train_iteration(iteration, epochs=EPOCHS_PER_ITERATION):
    """Train one iteration"""
    log(f"🚀 Iteration {iteration}: Training model...")
    
    yaml_path, num_samples = prepare_training_data()
    
    # Use YOLOv8s for multiclass detection (good balance of speed and accuracy)
    model = YOLO('yolov8s.pt')
    log(f"Using YOLOv8s ({num_samples} samples)")
    
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=640,
        batch=BATCH_SIZE,
        patience=20,
        save=True,
        device='mps' if torch.backends.mps.is_available() else 'cpu',
        project=str(RUNS_DIR),
        name=f'iteration_{iteration}',
        exist_ok=True,
        verbose=True,
        plots=True
    )
    
    log(f"✅ Iteration {iteration} training complete")
    return model


def predict_and_add_labels(model, iteration):
    """Predict on unlabeled images and add high-confidence predictions"""
    log(f"🎯 Iteration {iteration}: Predicting on unlabeled images...")
    
    # Get all images
    all_images = list(IMAGE_DIR.glob("*.jpg"))
    
    # Get already labeled images
    labeled_names = {f.stem for f in get_all_labels()}
    
    # Filter to unlabeled only
    unlabeled_images = [img for img in all_images if img.stem not in labeled_names]
    
    log(f"Found {len(unlabeled_images)} unlabeled images")
    
    if len(unlabeled_images) == 0:
        log("🎉 All images are labeled!")
        return 0
    
    # Adjust confidence threshold based on iteration
    # Start strict, then relax to get more data
    conf_threshold = max(MIN_CONFIDENCE, CONFIDENCE_THRESHOLD - (iteration * 0.05))
    log(f"Using confidence threshold: {conf_threshold:.2f}")
    
    # Predict
    added_count = 0
    log(f"Processing {len(unlabeled_images)} unlabeled images...")
    for idx, img_path in enumerate(unlabeled_images, 1):
        if idx % 50 == 0:  # Print every 50 images
            log(f"  Processed {idx}/{len(unlabeled_images)} images...")
        results = model(img_path, verbose=False, conf=conf_threshold)
        
        boxes = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x_center = float(box.xywhn[0][0])
                    y_center = float(box.xywhn[0][1])
                    width = float(box.xywhn[0][2])
                    height = float(box.xywhn[0][3])
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    boxes.append({
                        "class": class_id,
                        "x": x_center,
                        "y": y_center,
                        "w": width,
                        "h": height,
                        "severity": 3,
                        "confidence": confidence
                    })
        
        # Save if we found high-confidence detections
        if boxes:
            output_file = AUTO_LABELS / f"{img_path.stem}.json"
            with open(output_file, 'w') as f:
                json.dump({"boxes": boxes, "iteration": iteration}, f, indent=2)
            added_count += 1
    
    log(f"✅ Added {added_count} new auto-labeled images")
    return added_count


def cleanup_low_confidence_labels():
    """Remove auto labels that might be incorrect (low confidence)"""
    removed = 0
    for label_file in AUTO_LABELS.glob("*.json"):
        with open(label_file) as f:
            data = json.load(f)
        
        # Keep only high-confidence boxes
        good_boxes = [b for b in data.get('boxes', []) if b.get('confidence', 1.0) > 0.4]
        
        if len(good_boxes) == 0:
            label_file.unlink()
            removed += 1
        elif len(good_boxes) < len(data.get('boxes', [])):
            data['boxes'] = good_boxes
            with open(label_file, 'w') as f:
                json.dump(data, f, indent=2)
    
    if removed > 0:
        log(f"🧹 Cleaned up {removed} low-confidence labels")


def main():
    """Main continuous training loop"""
    log("=" * 60)
    log("🌙 OVERNIGHT CONTINUOUS TRAINING STARTED")
    log("=" * 60)
    log(f"Max iterations: {MAX_ITERATIONS}")
    log(f"Epochs per iteration: {EPOCHS_PER_ITERATION}")
    log(f"Device: {'MPS (Apple Silicon)' if torch.backends.mps.is_available() else 'CPU'}")
    log("")
    
    start_time = time.time()
    best_model_path = None
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        log(f"\n{'='*60}")
        log(f"ITERATION {iteration}/{MAX_ITERATIONS}")
        log(f"{'='*60}")
        
        # Train model
        model = train_iteration(iteration, epochs=EPOCHS_PER_ITERATION)
        
        # Save best model path
        best_model_path = RUNS_DIR / f"iteration_{iteration}" / "weights" / "best.pt"
        
        # Predict on unlabeled data
        added = predict_and_add_labels(model, iteration)
        
        # Clean up bad predictions
        cleanup_low_confidence_labels()
        
        # Check if we should continue
        if added == 0:
            log("✅ No more unlabeled images - training complete!")
            break
        
        # Brief pause between iterations
        log("⏸️  Pausing 10 seconds before next iteration...")
        time.sleep(10)
    
    # Final statistics
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    
    log("\n" + "="*60)
    log("🎉 CONTINUOUS TRAINING COMPLETE!")
    log("="*60)
    log(f"Total time: {hours}h {minutes}m")
    log(f"Total labeled images: {len(list(get_all_labels()))}")
    log(f"Manual labels: {len(list(MANUAL_LABELS.glob('*.json')))}")
    log(f"Auto labels: {len(list(AUTO_LABELS.glob('*.json')))}")
    
    if best_model_path and best_model_path.exists():
        log(f"Best model: {best_model_path}")
        
        # Copy best model to easy location
        final_model = Path("detection-model/best_model.pt")
        shutil.copy(best_model_path, final_model)
        log(f"✅ Final model saved to: {final_model}")
    
    log("\n💤 Good night! Your model is trained and ready.")


if __name__ == "__main__":
    main()
