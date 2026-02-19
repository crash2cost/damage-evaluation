"""
Semi-Automated Dataset Labeling (Multiclass)
Uses trained multiclass YOLO model to generate labels for new images.
High-confidence detections are auto-accepted; borderline cases are flagged for review.
"""

import shutil
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "detection-model" / "runs" / "train" / "weights" / "best.pt"
DEFAULT_SOURCE = ROOT / "severity-model" / "dataset"
DEFAULT_OUTPUT = ROOT / "detection-model" / "dataset-auto-labeled"

CLASSES = [
    "bumper_dent", "bumper_scratch", "door_dent", "door_scratch",
    "glass_shatter", "head_lamp", "tail_lamp",
]

# Confidence thresholds
HIGH_CONF = 0.6    # Auto-accept
LOW_CONF = 0.3     # Flag for review (between LOW and HIGH)
# Below LOW_CONF: skip entirely


class AutoLabeler:
    def __init__(self, model_path, high_conf=HIGH_CONF, low_conf=LOW_CONF):
        self.model = YOLO(str(model_path))
        self.high_conf = high_conf
        self.low_conf = low_conf

    def label_image(self, image_path):
        """
        Generate YOLO format labels for an image.
        Returns: (labels, needs_review)
            labels: list of YOLO format strings
            needs_review: True if any detection is in borderline confidence range
        """
        results = self.model.predict(str(image_path), conf=self.low_conf, verbose=False)

        labels = []
        needs_review = False

        for result in results:
            if result.boxes is None:
                continue
            img_w = result.orig_shape[1]
            img_h = result.orig_shape[0]

            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                confidence = float(conf.item())
                class_id = int(cls.item())

                if confidence < self.low_conf:
                    continue

                if confidence < self.high_conf:
                    needs_review = True

                x1, y1, x2, y2 = box.cpu().numpy()
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        return labels, needs_review


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Auto-label images for YOLO training")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE),
                        help="Source directory with images organized by class")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--high-conf", type=float, default=HIGH_CONF)
    parser.add_argument("--low-conf", type=float, default=LOW_CONF)
    args = parser.parse_args()

    source = Path(args.source)
    output_dir = Path(args.output)

    for split in ["train", "val"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "review").mkdir(parents=True, exist_ok=True)

    labeler = AutoLabeler(args.model, args.high_conf, args.low_conf)

    stats = {"total": 0, "labeled": 0, "no_detection": 0, "needs_review": 0}

    for split in ["train", "val"]:
        split_dir = source / split
        if not split_dir.exists():
            continue

        for damage_type in CLASSES:
            folder = split_dir / damage_type
            if not folder.exists():
                continue

            images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))

            for img_path in tqdm(images, desc=f"  {split}/{damage_type}"):
                stats["total"] += 1

                labels, needs_review = labeler.label_image(img_path)
                output_name = f"{damage_type}_{img_path.name}"
                stem = Path(output_name).stem

                if not labels:
                    stats["no_detection"] += 1
                    continue

                if needs_review:
                    stats["needs_review"] += 1
                    shutil.copy(img_path, output_dir / "review" / output_name)
                    with open(output_dir / "review" / f"{stem}.txt", "w") as f:
                        f.write("\n".join(labels))
                else:
                    stats["labeled"] += 1
                    shutil.copy(img_path, output_dir / split / "images" / output_name)
                    with open(output_dir / split / "labels" / f"{stem}.txt", "w") as f:
                        f.write("\n".join(labels))

    # Write data.yaml
    yaml_content = f"""# Auto-labeled multiclass detection dataset
path: {output_dir.absolute()}
train: train/images
val: val/images

nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(output_dir / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\n{'='*60}")
    print("AUTO-LABELING COMPLETE")
    print(f"{'='*60}")
    print(f"Total images: {stats['total']}")
    print(f"  Auto-labeled (high conf): {stats['labeled']}")
    print(f"  Needs review (borderline): {stats['needs_review']}")
    print(f"  No detection: {stats['no_detection']}")
    print(f"\nOutput: {output_dir}")
    print(f"Review queue: {output_dir / 'review'} ({stats['needs_review']} images)")
    print("\nIMPORTANT: Review borderline images before adding to training set!")


if __name__ == "__main__":
    main()
