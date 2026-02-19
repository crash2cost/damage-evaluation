import sys
import os
from pathlib import Path
from PIL import Image
import shutil
from tqdm import tqdm
sys.path.append('/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/ml-service/detection-model/src')
from custom_inference import CustomYOLOInference
manual_labels_dir = Path("/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/ml-service/manual_labels")
cardd_images_dir = Path("/Users/idolevi/.cache/huggingface/hub/datasets--harpreetsahota--CarDD/snapshots/56900bde8dddfe00eb7c03114a1d46e9105e3cdb/data")
model_path = "/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/ml-service/detection-model/runs/custom-yolo/best.pt"
output_labels_dir = Path("/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/ml-service/auto_labels")
final_dataset_dir = Path("/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/ml-service/detection-model/dataset-final")
output_labels_dir.mkdir(exist_ok=True)
print(" Auto-labeling remaining CarDD images...")
print(f"   Model: {model_path}")
print(f"   Images: {cardd_images_dir}")
print(f"   Output: {output_labels_dir}\n")
model = CustomYOLOInference(model_path)
all_images = sorted(cardd_images_dir.glob("*.jpg"))
print(f" Found {len(all_images)} total CarDD images")
manually_labeled = {p.stem for p in manual_labels_dir.glob("*.txt")}
print(f"   Already labeled manually: {len(manually_labeled)}")
unlabeled_images = [img for img in all_images if img.stem not in manually_labeled]
print(f"   To auto-label: {len(unlabeled_images)}\n")
confidence = 0.15
detections_found = 0
no_detections = 0
print(f" Auto-labeling with confidence={confidence}...")
for img_path in tqdm(unlabeled_images, desc="Auto-labeling"):
    try:
        result = model.predict(str(img_path), conf=confidence)
        if result and hasattr(result, 'boxes') and result.boxes.xyxy is not None and len(result.boxes.xyxy) > 0:
            img = Image.open(img_path)
            img_w, img_h = img.size
            label_path = output_labels_dir / f"{img_path.stem}.txt"
            with open(label_path, 'w') as f:
                for i in range(len(result.boxes.xyxy)):
                    x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
                    x_center = (x1 + x2) / 2 / img_w
                    y_center = (y1 + y2) / 2 / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            detections_found += 1
        else:
            no_detections += 1
    except Exception as e:
        no_detections += 1
print(f"\n Auto-labeling complete!")
print(f"   Images with detections: {detections_found} ({detections_found/len(unlabeled_images)*100:.1f}%)")
print(f"   Images without detections: {no_detections} ({no_detections/len(unlabeled_images)*100:.1f}%)")
print(f"\n Creating final dataset...")
print(f"   Manual labels: {len(manually_labeled)}")
print(f"   Auto labels: {detections_found}")
print(f"   Total labeled: {len(manually_labeled) + detections_found}")
(final_dataset_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
(final_dataset_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
(final_dataset_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
(final_dataset_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)
import random
random.seed(42)
all_labeled = []
for label_path in manual_labels_dir.glob("*.txt"):
    img_path = cardd_images_dir / f"{label_path.stem}.jpg"
    if img_path.exists():
        all_labeled.append(('manual', img_path, label_path))
for label_path in output_labels_dir.glob("*.txt"):
    img_path = cardd_images_dir / f"{label_path.stem}.jpg"
    if img_path.exists():
        all_labeled.append(('auto', img_path, label_path))
original_dataset = Path("/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/ml-service/detection-model/dataset")
for split in ['train', 'val']:
    img_dir = original_dataset / split / "images"
    lbl_dir = original_dataset / split / "labels"
    for img_path in img_dir.glob("*.jpg"):
        label_path = lbl_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            all_labeled.append(('original', img_path, label_path))
print(f"   Total images to merge: {len(all_labeled)}")
random.shuffle(all_labeled)
split_idx = int(len(all_labeled) * 0.8)
train_set = all_labeled[:split_idx]
val_set = all_labeled[split_idx:]
for source, img_path, label_path in tqdm(train_set, desc="Copying train"):
    prefix = source
    shutil.copy2(img_path, final_dataset_dir / "train" / "images" / f"{prefix}_{img_path.name}")
    shutil.copy2(label_path, final_dataset_dir / "train" / "labels" / f"{prefix}_{label_path.name}")
for source, img_path, label_path in tqdm(val_set, desc="Copying val"):
    prefix = source
    shutil.copy2(img_path, final_dataset_dir / "val" / "images" / f"{prefix}_{img_path.name}")
    shutil.copy2(label_path, final_dataset_dir / "val" / "labels" / f"{prefix}_{label_path.name}")
yaml_content = f"""# Final merged dataset
path: {final_dataset_dir}
train: train/images
val: val/images
names:
  0: damage
nc: 1