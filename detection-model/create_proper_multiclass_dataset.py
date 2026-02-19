#!/usr/bin/env python3
"""
Create Proper Multiclass YOLO Dataset
======================================
Combines real bounding boxes from the single-class detector with
multiclass labels from the severity classifier to create a proper
multiclass detection dataset.

Strategy:
1. For dataset-multiclass images: Use single-class YOLO to get real bboxes,
   keep original multiclass label.
2. For dataset-final images: Use single-class YOLO bboxes + severity model
   to classify crops into damage types (only high-confidence predictions).
3. Merge into dataset-multiclass-v2/ with stratified train/val split.

Usage:
    python create_proper_multiclass_dataset.py
    python create_proper_multiclass_dataset.py --conf-threshold 0.6
    python create_proper_multiclass_dataset.py --skip-final  # Only process multiclass images
"""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import models, transforms
from ultralytics import YOLO

# =============================================================================
# Constants
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent
DETECTION_WEIGHTS = ROOT / "detection-model" / "runs" / "train" / "weights" / "best.pt"
SEVERITY_WEIGHTS = ROOT / "severity-model" / "models" / "best_model.pt"

MULTICLASS_DIR = ROOT / "detection-model" / "dataset-multiclass"
FINAL_DIR = ROOT / "detection-model" / "dataset-final"
OUTPUT_DIR = ROOT / "detection-model" / "dataset-multiclass-v2"

# 7 classes (no "unknown")
CLASSES = [
    "bumper_dent",
    "bumper_scratch",
    "door_dent",
    "door_scratch",
    "glass_shatter",
    "head_lamp",
    "tail_lamp",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

# Old multiclass dataset mapping (had "unknown" at index 0)
OLD_CLASS_NAMES = {
    0: "unknown",
    1: "door_dent",
    2: "bumper_scratch",
    3: "door_scratch",
    4: "glass_shatter",
    5: "tail_lamp",
    6: "head_lamp",
    7: "bumper_dent",
}

DETECTION_CONF_THRESHOLD = 0.25
SEVERITY_CONF_THRESHOLD = 0.7
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SEVERITY_INPUT_SIZE = 224

VAL_SPLIT = 0.2
RANDOM_SEED = 42

# Parallel processing
YOLO_BATCH_SIZE = 32  # Images per YOLO batch prediction
SEVERITY_BATCH_SIZE = 64  # Crops per severity model batch
IO_WORKERS = 8  # Threads for file I/O (hashing, reading labels, opening images)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_severity_model(device: torch.device) -> Tuple[nn.Module, List[str], transforms.Compose]:
    """Load the severity/damage-type classification model."""
    checkpoint = torch.load(str(SEVERITY_WEIGHTS), map_location=device)
    classes = checkpoint["classes"]
    config = checkpoint.get("config", checkpoint.get("hyperparameters", {}))
    backbone = config.get("backbone", "resnet18")
    dropout_rate = config.get("dropout_rate", 0.3)
    num_classes = len(classes)

    if backbone.startswith("efficientnet"):
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )
    elif backbone == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(model.fc.in_features, num_classes),
        )
    else:
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(model.fc.in_features, num_classes),
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((SEVERITY_INPUT_SIZE, SEVERITY_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return model, classes, transform


def classify_crop(
    model: nn.Module,
    image: Image.Image,
    bbox: Tuple[float, float, float, float],
    transform: transforms.Compose,
    classes: List[str],
    device: torch.device,
) -> Tuple[str, float]:
    """Classify a single cropped region using the severity model."""
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(image.width, int(x2)), min(image.height, int(y2))

    if x2 <= x1 or y2 <= y1:
        return "unknown", 0.0

    cropped = image.crop((x1, y1, x2, y2))
    input_tensor = transform(cropped).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = probs.max(1)

    return classes[pred.item()], conf.item()


def classify_crops_batch(
    model: nn.Module,
    crops: List[Image.Image],
    transform: transforms.Compose,
    classes: List[str],
    device: torch.device,
    batch_size: int = SEVERITY_BATCH_SIZE,
) -> List[Tuple[str, float]]:
    """Classify multiple crops in batches for GPU efficiency."""
    if not crops:
        return []

    results = []
    for i in range(0, len(crops), batch_size):
        batch_crops = crops[i:i + batch_size]
        tensors = torch.stack([transform(c) for c in batch_crops]).to(device)

        with torch.no_grad():
            outputs = model(tensors)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(1)

        for conf, pred in zip(confs, preds):
            results.append((classes[pred.item()], conf.item()))

    return results


def image_hash(path: Path) -> str:
    """Quick hash to detect duplicate images."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _read_multiclass_label(label_path: Path) -> Optional[Tuple[int, str, int]]:
    """Read a multiclass label file. Returns (old_class_id, class_name, new_class_id) or None."""
    if not label_path.exists():
        return None
    with open(label_path) as f:
        line = f.readline().strip()
    if not line:
        return None
    old_class_id = int(line.split()[0])
    old_class_name = OLD_CLASS_NAMES.get(old_class_id, "unknown")
    if old_class_name == "unknown" or old_class_name not in CLASS_TO_ID:
        return None
    return old_class_id, old_class_name, CLASS_TO_ID[old_class_name]


def process_multiclass_images(
    detector: YOLO,
    det_conf: float,
) -> List[dict]:
    """
    Process dataset-multiclass images with batched YOLO detection.
    Uses threading for I/O and batched prediction for speed.
    """
    entries = []
    skipped = 0

    for split in ["train", "val"]:
        img_dir = MULTICLASS_DIR / split / "images"
        lbl_dir = MULTICLASS_DIR / split / "labels"

        if not img_dir.exists():
            continue

        all_images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        print(f"  Processing {len(all_images)} multiclass {split} images...")

        # Step 1: Read labels in parallel to filter valid images
        label_paths = [lbl_dir / (p.stem + ".txt") for p in all_images]
        with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
            label_results = list(pool.map(_read_multiclass_label, label_paths))

        valid = [(img, lr) for img, lr in zip(all_images, label_results) if lr is not None]
        skipped += len(all_images) - len(valid)

        if not valid:
            continue

        # Step 2: Batch YOLO prediction on all valid images
        img_paths_str = [str(img) for img, _ in valid]
        for batch_start in range(0, len(img_paths_str), YOLO_BATCH_SIZE):
            batch_paths = img_paths_str[batch_start:batch_start + YOLO_BATCH_SIZE]
            batch_valid = valid[batch_start:batch_start + YOLO_BATCH_SIZE]

            try:
                batch_results = detector.predict(batch_paths, conf=det_conf, verbose=False)
            except Exception as e:
                print(f"    Batch prediction error: {e}, falling back to single")
                batch_results = []
                for p in batch_paths:
                    try:
                        batch_results.append(detector.predict(p, conf=det_conf, verbose=False)[0])
                    except Exception:
                        batch_results.append(None)

            for (img_path, (_, old_class_name, new_class_id)), result in zip(batch_valid, batch_results):
                if result is None:
                    skipped += 1
                    continue

                # Extract bboxes
                bboxes = []
                results_list = [result] if not isinstance(result, list) else result
                for r in results_list:
                    if r.boxes is None:
                        continue
                    img_w, img_h = r.orig_shape[1], r.orig_shape[0]
                    for box in r.boxes.xyxy:
                        x1, y1, x2, y2 = box.cpu().numpy()
                        cx = ((x1 + x2) / 2) / img_w
                        cy = ((y1 + y2) / 2) / img_h
                        bw = (x2 - x1) / img_w
                        bh = (y2 - y1) / img_h
                        bboxes.append((cx, cy, bw, bh))

                if not bboxes:
                    bboxes.append((0.5, 0.5, 0.7, 0.7))

                entries.append({
                    "image_path": img_path,
                    "class_name": old_class_name,
                    "class_id": new_class_id,
                    "bboxes": bboxes,
                    "source": "multiclass",
                })

            if batch_start % (YOLO_BATCH_SIZE * 4) == 0 and batch_start > 0:
                print(f"    {batch_start}/{len(img_paths_str)} images processed...")

    print(f"  Multiclass: {len(entries)} images kept, {skipped} skipped")
    return entries


def _read_final_label(label_path: Path) -> List[Tuple[float, float, float, float]]:
    """Read bboxes from a dataset-final label file."""
    if not label_path.exists():
        return []
    bboxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                bboxes.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return bboxes


def process_final_images(
    detector: YOLO,
    severity_model: nn.Module,
    severity_classes: List[str],
    severity_transform: transforms.Compose,
    device: torch.device,
    det_conf: float,
    sev_conf: float,
    seen_hashes: set,
) -> List[dict]:
    """
    Process dataset-final images with threaded I/O and batched severity classification.
    """
    entries = []
    skipped = 0
    low_conf = 0

    for split in ["train", "val"]:
        img_dir = FINAL_DIR / split / "images"
        lbl_dir = FINAL_DIR / split / "labels"

        if not img_dir.exists():
            continue

        all_images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        print(f"  Processing {len(all_images)} dataset-final {split} images...")

        # Step 1: Hash all images in parallel to find duplicates
        print(f"    Hashing {len(all_images)} images for dedup...")
        with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
            hashes = list(pool.map(image_hash, all_images))

        # Step 2: Read all labels in parallel
        label_paths = [lbl_dir / (p.stem + ".txt") for p in all_images]
        with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
            all_bboxes = list(pool.map(_read_final_label, label_paths))

        # Step 3: Filter to non-duplicate images with valid labels
        valid_items = []
        for img_path, h, bboxes in zip(all_images, hashes, all_bboxes):
            if h in seen_hashes:
                skipped += 1
                continue
            seen_hashes.add(h)
            if not bboxes:
                skipped += 1
                continue
            valid_items.append((img_path, bboxes))

        print(f"    {len(valid_items)} unique images with labels (skipped {skipped})")

        # Step 4: Open images and collect all crops for batched classification
        # Process in chunks to limit memory usage
        CHUNK_SIZE = 500
        for chunk_start in range(0, len(valid_items), CHUNK_SIZE):
            chunk = valid_items[chunk_start:chunk_start + CHUNK_SIZE]

            # Open images in parallel
            def _open_image(item):
                img_path, bboxes = item
                try:
                    return img_path, Image.open(img_path).convert("RGB"), bboxes
                except Exception:
                    return img_path, None, bboxes

            with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
                opened = list(pool.map(_open_image, chunk))

            # Collect all crops across all images in this chunk
            all_crops = []  # PIL crops
            crop_meta = []  # (img_idx, bbox) to trace back
            valid_opened = []

            for img_idx, (img_path, image, bboxes) in enumerate(opened):
                if image is None:
                    skipped += 1
                    continue
                valid_opened.append((img_path, image, bboxes))

                for cx, cy, bw, bh in bboxes:
                    w, h = image.width, image.height
                    x1 = max(0, int((cx - bw / 2) * w))
                    y1 = max(0, int((cy - bh / 2) * h))
                    x2 = min(w, int((cx + bw / 2) * w))
                    y2 = min(h, int((cy + bh / 2) * h))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = image.crop((x1, y1, x2, y2))
                    all_crops.append(crop)
                    crop_meta.append((len(valid_opened) - 1, (cx, cy, bw, bh)))

            # Step 5: Batched severity classification on ALL crops at once
            if all_crops:
                batch_results = classify_crops_batch(
                    severity_model, all_crops, severity_transform,
                    severity_classes, device, SEVERITY_BATCH_SIZE,
                )
            else:
                batch_results = []

            # Step 6: Map classification results back to images
            # Group results by image index
            img_classified = {}  # img_idx -> [(class_id, class_name, bbox), ...]
            for (img_idx, bbox), (damage_type, conf) in zip(crop_meta, batch_results):
                if conf < sev_conf or damage_type not in CLASS_TO_ID:
                    low_conf += 1
                    continue
                if img_idx not in img_classified:
                    img_classified[img_idx] = []
                img_classified[img_idx].append({
                    "class_id": CLASS_TO_ID[damage_type],
                    "class_name": damage_type,
                    "bbox": bbox,
                })

            for img_idx, img_entries in img_classified.items():
                img_path = valid_opened[img_idx][0]
                entries.append({
                    "image_path": img_path,
                    "class_name": img_entries[0]["class_name"],
                    "class_id": img_entries[0]["class_id"],
                    "bboxes": [e["bbox"] for e in img_entries],
                    "bbox_classes": [e["class_id"] for e in img_entries],
                    "source": "final",
                })

            processed = min(chunk_start + CHUNK_SIZE, len(valid_items))
            print(f"    {processed}/{len(valid_items)} images classified...")

    print(f"  Dataset-final: {len(entries)} images kept, {skipped} skipped (dupes/missing), {low_conf} low-confidence crops")
    return entries


def write_dataset(entries: List[dict], output_dir: Path) -> None:
    """Write the merged dataset in YOLO format with stratified split."""
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ["train", "val"]:
        (output_dir / split / "images").mkdir(parents=True)
        (output_dir / split / "labels").mkdir(parents=True)

    # Stratified split by class
    labels = [e["class_name"] for e in entries]
    train_entries, val_entries = train_test_split(
        entries,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    print(f"\n  Train: {len(train_entries)} images")
    print(f"  Val:   {len(val_entries)} images")

    for split_name, split_entries in [("train", train_entries), ("val", val_entries)]:
        for idx, entry in enumerate(split_entries):
            ext = entry["image_path"].suffix
            img_name = f"img_{idx:06d}{ext}"
            img_dst = output_dir / split_name / "images" / img_name
            lbl_dst = output_dir / split_name / "labels" / f"img_{idx:06d}.txt"

            shutil.copy2(entry["image_path"], img_dst)

            with open(lbl_dst, "w") as f:
                if "bbox_classes" in entry:
                    # Dataset-final: each bbox may have a different class
                    for class_id, (cx, cy, bw, bh) in zip(entry["bbox_classes"], entry["bboxes"]):
                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                else:
                    # Multiclass: all bboxes share the same class
                    for cx, cy, bw, bh in entry["bboxes"]:
                        f.write(f"{entry['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    # Write data.yaml
    yaml_content = f"""# Crash2Cost Multiclass Detection Dataset v2
# Auto-generated with real bounding boxes
path: {output_dir.absolute()}
train: train/images
val: val/images

nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(output_dir / "data.yaml", "w") as f:
        f.write(yaml_content)

    # Write class distribution report
    train_counts = Counter(e["class_name"] for e in train_entries)
    val_counts = Counter(e["class_name"] for e in val_entries)

    print("\n  Class Distribution:")
    print(f"  {'Class':<20s} {'Train':>6s} {'Val':>6s} {'Total':>6s}")
    print(f"  {'-'*40}")
    for cls in CLASSES:
        t = train_counts.get(cls, 0)
        v = val_counts.get(cls, 0)
        print(f"  {cls:<20s} {t:>6d} {v:>6d} {t+v:>6d}")

    stats = {
        "total_images": len(entries),
        "train_images": len(train_entries),
        "val_images": len(val_entries),
        "classes": CLASSES,
        "class_distribution_train": dict(train_counts),
        "class_distribution_val": dict(val_counts),
    }
    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Create proper multiclass YOLO dataset")
    parser.add_argument("--det-conf", type=float, default=DETECTION_CONF_THRESHOLD,
                        help="Detection confidence threshold")
    parser.add_argument("--sev-conf", type=float, default=SEVERITY_CONF_THRESHOLD,
                        help="Severity classification confidence threshold")
    parser.add_argument("--skip-final", action="store_true",
                        help="Skip processing dataset-final (only use multiclass)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    device = get_device()
    print(f"Device: {device}")

    # Load models
    print("\nLoading detection model...")
    detector = YOLO(str(DETECTION_WEIGHTS))

    print("Loading severity model...")
    severity_model, severity_classes, severity_transform = load_severity_model(device)
    print(f"  Severity classes: {severity_classes}")

    # Process multiclass images
    print(f"\n{'='*60}")
    print("Step 1: Processing multiclass dataset (real bboxes + original labels)")
    print(f"{'='*60}")
    multiclass_entries = process_multiclass_images(detector, args.det_conf)

    # Track seen image hashes to avoid duplicates (parallel hashing)
    print("  Hashing multiclass images for dedup...")
    mc_paths = [entry["image_path"] for entry in multiclass_entries]
    with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
        mc_hashes = list(pool.map(image_hash, mc_paths))
    seen_hashes = set(mc_hashes)

    # Process dataset-final images
    final_entries = []
    if not args.skip_final:
        print(f"\n{'='*60}")
        print("Step 2: Processing dataset-final (real bboxes + severity classifier)")
        print(f"{'='*60}")
        final_entries = process_final_images(
            detector, severity_model, severity_classes, severity_transform,
            device, args.det_conf, args.sev_conf, seen_hashes,
        )

    # Merge and write
    all_entries = multiclass_entries + final_entries
    print(f"\n{'='*60}")
    print(f"Step 3: Writing merged dataset ({len(all_entries)} total images)")
    print(f"{'='*60}")

    if len(all_entries) < 10:
        print("ERROR: Too few images. Check model weights and dataset paths.")
        return

    write_dataset(all_entries, output_dir)

    print(f"\nDataset created at: {output_dir}")
    print(f"Config file: {output_dir / 'data.yaml'}")
    print("\nNext step: Train with:")
    print(f"  python train.py --data {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
