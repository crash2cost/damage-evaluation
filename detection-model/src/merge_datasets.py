import shutil
from pathlib import Path
import random
manual_labels_dir = Path("/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/machines/manual_labels")
cardd_images_dir = Path("/Users/idolevi/.cache/huggingface/hub/datasets--harpreetsahota--CarDD/snapshots/56900bde8dddfe00eb7c03114a1d46e9105e3cdb/data")
original_dataset = Path("/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/machines/detection-model/dataset")
output_dataset = Path("/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/machines/detection-model/dataset-manual")
(output_dataset / "train" / "images").mkdir(parents=True, exist_ok=True)
(output_dataset / "train" / "labels").mkdir(parents=True, exist_ok=True)
(output_dataset / "val" / "images").mkdir(parents=True, exist_ok=True)
(output_dataset / "val" / "labels").mkdir(parents=True, exist_ok=True)
print(" Merging datasets...")
print(f"   Manual labels: {manual_labels_dir}")
print(f"   Original dataset: {original_dataset}")
print(f"   Output: {output_dataset}\n")
print(" Copying original labeled images...")
train_count = 0
val_count = 0
for split in ['train', 'val']:
    img_dir = original_dataset / split / "images"
    lbl_dir = original_dataset / split / "labels"
    for img_path in img_dir.glob("*.jpg"):
        label_path = lbl_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(img_path, output_dataset / split / "images" / f"original_{img_path.name}")
            shutil.copy2(label_path, output_dataset / split / "labels" / f"original_{img_path.stem}.txt")
            if split == 'train':
                train_count += 1
            else:
                val_count += 1
print(f"    Original train: {train_count}")
print(f"    Original val: {val_count}")
print("\n Adding manually labeled images...")
labeled_files = list(manual_labels_dir.glob("*.txt"))
random.shuffle(labeled_files)
split_idx = int(len(labeled_files) * 0.8)
train_labels = labeled_files[:split_idx]
val_labels = labeled_files[split_idx:]
manual_train = 0
manual_val = 0
for label_path in train_labels:
    img_name = label_path.stem + ".jpg"
    img_path = cardd_images_dir / img_name
    if img_path.exists():
        shutil.copy2(img_path, output_dataset / "train" / "images" / f"manual_{img_name}")
        shutil.copy2(label_path, output_dataset / "train" / "labels" / f"manual_{label_path.stem}.txt")
        manual_train += 1
for label_path in val_labels:
    img_name = label_path.stem + ".jpg"
    img_path = cardd_images_dir / img_name
    if img_path.exists():
        shutil.copy2(img_path, output_dataset / "val" / "images" / f"manual_{img_name}")
        shutil.copy2(label_path, output_dataset / "val" / "labels" / f"manual_{label_path.stem}.txt")
        manual_val += 1
print(f"    Manual train: {manual_train}")
print(f"    Manual val: {manual_val}")
yaml_content = f"""# Merged dataset configuration
path: {output_dataset}
train: train/images
val: val/images
names:
  0: damage
nc: 1