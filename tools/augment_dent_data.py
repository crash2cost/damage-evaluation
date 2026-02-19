#!/usr/bin/env python3
"""
Augment Severity Model Training Data
=====================================
Copies images from multiclass detection dataset to severity model dataset
to balance the training data, especially for underrepresented dent classes.

Usage:
    python tools/augment_dent_data.py --dry-run     # Preview what will be copied
    python tools/augment_dent_data.py               # Actually copy files
"""

import argparse
import shutil
from pathlib import Path
from collections import defaultdict
import random

# Paths
ROOT = Path(__file__).resolve().parent.parent
MULTICLASS_DIR = ROOT / "detection-model" / "dataset-multiclass"
SEVERITY_DIR = ROOT / "severity-model" / "dataset"

# Class mapping (multiclass ID -> class name)
CLASS_MAPPING = {
    0: "bumper_dent",
    1: "bumper_scratch",
    2: "door_dent",
    3: "door_scratch",
    4: "glass_shatter",
    5: "head_lamp",
    6: "tail_lamp",
}


def count_existing_images(dataset_dir: Path) -> dict:
    """Count images in each class folder."""
    counts = {}
    for split in ["train", "val", "test"]:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        for class_folder in split_dir.iterdir():
            if class_folder.is_dir():
                class_name = class_folder.name
                if class_name not in counts:
                    counts[class_name] = {"train": 0, "val": 0, "test": 0}
                counts[class_name][split] = len(list(class_folder.glob("*.[jJ][pP][gG]")) +
                                                 list(class_folder.glob("*.[jJ][pP][eE][gG]")) +
                                                 list(class_folder.glob("*.[pP][nN][gG]")))
    return counts


def get_multiclass_images_by_class(multiclass_dir: Path) -> dict:
    """Get all images grouped by class from multiclass dataset."""
    images_by_class = defaultdict(list)

    for split in ["train", "val"]:
        labels_dir = multiclass_dir / split / "labels"
        images_dir = multiclass_dir / split / "images"

        if not labels_dir.exists():
            continue

        for label_file in labels_dir.glob("*.txt"):
            with open(label_file) as f:
                content = f.read().strip()
                if not content:
                    continue
                class_id = int(content.split()[0])

            # Find corresponding image
            img_name = label_file.stem
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                img_path = images_dir / f"{img_name}{ext}"
                if img_path.exists():
                    class_name = CLASS_MAPPING.get(class_id)
                    if class_name:
                        images_by_class[class_name].append(img_path)
                    break

    return images_by_class


def augment_dataset(dry_run: bool = True, target_min: int = 200):
    """
    Augment severity dataset with images from multiclass dataset.

    Args:
        dry_run: If True, only preview what would be copied
        target_min: Minimum number of training images per class to target
    """
    print("=" * 70)
    print("SEVERITY MODEL DATA AUGMENTATION")
    print("=" * 70)

    # Count existing images
    print("\n Current severity dataset distribution:")
    existing = count_existing_images(SEVERITY_DIR)
    for class_name in CLASS_MAPPING.values():
        counts = existing.get(class_name, {"train": 0, "val": 0, "test": 0})
        total = counts["train"] + counts["val"] + counts["test"]
        print(f"   {class_name:20} train={counts['train']:3}  val={counts['val']:3}  test={counts['test']:3}  total={total}")

    # Get multiclass images
    print("\n Available images in multiclass dataset:")
    multiclass_images = get_multiclass_images_by_class(MULTICLASS_DIR)
    for class_name, images in sorted(multiclass_images.items()):
        print(f"   {class_name:20} {len(images):4} images")

    # Calculate what to copy
    print(f"\n Target: at least {target_min} training images per class")
    print("\n Augmentation plan:")

    to_copy = {}
    for class_name in CLASS_MAPPING.values():
        current_train = existing.get(class_name, {}).get("train", 0)
        available = multiclass_images.get(class_name, [])

        # Get existing image names to avoid duplicates
        existing_names = set()
        train_dir = SEVERITY_DIR / "train" / class_name
        if train_dir.exists():
            for img in train_dir.iterdir():
                existing_names.add(img.stem)

        # Filter out duplicates
        new_images = [img for img in available if img.stem not in existing_names]

        needed = max(0, target_min - current_train)
        to_add = min(needed, len(new_images))

        if to_add > 0:
            # Randomly select images to add
            selected = random.sample(new_images, to_add) if len(new_images) > to_add else new_images[:to_add]
            to_copy[class_name] = selected
            print(f"   {class_name:20} +{to_add:3} images ({current_train}  {current_train + to_add})")
        else:
            print(f"   {class_name:20} +{0:3} images (already has {current_train})")

    # Execute copy
    if dry_run:
        print("\n  DRY RUN - no files copied")
        print("   Run without --dry-run to actually copy files")
    else:
        print("\n Copying files...")
        total_copied = 0
        for class_name, images in to_copy.items():
            dest_dir = SEVERITY_DIR / "train" / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img_path in images:
                dest_path = dest_dir / f"aug_{img_path.name}"
                shutil.copy2(img_path, dest_path)
                total_copied += 1

            print(f"    {class_name}: copied {len(images)} images")

        print(f"\n Total: {total_copied} images copied")

    # Show final distribution
    if not dry_run:
        print("\n Updated severity dataset distribution:")
        updated = count_existing_images(SEVERITY_DIR)
        for class_name in CLASS_MAPPING.values():
            counts = updated.get(class_name, {"train": 0, "val": 0, "test": 0})
            total = counts["train"] + counts["val"] + counts["test"]
            print(f"   {class_name:20} train={counts['train']:3}  val={counts['val']:3}  test={counts['test']:3}  total={total}")

    print("\n" + "=" * 70)
    return to_copy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment severity dataset with multiclass images")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't copy files")
    parser.add_argument("--target", type=int, default=200, help="Target minimum training images per class")
    args = parser.parse_args()

    augment_dataset(dry_run=args.dry_run, target_min=args.target)
