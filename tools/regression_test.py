#!/usr/bin/env python3
"""
Regression Testing - Compare old vs new model
===============================================
Ensures the new multiclass model doesn't regress on detection recall
while improving classification accuracy.

Usage:
    python regression_test.py
    python regression_test.py --new-weights path/to/new.pt --old-weights path/to/old.pt
    python regression_test.py --test-dir path/to/test/images
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
OLD_WEIGHTS = ROOT / "detection-model" / "runs" / "train" / "weights" / "best.pt"
NEW_WEIGHTS = OLD_WEIGHTS  # Same path after retraining
TEST_DIR = ROOT / "detection-model" / "dataset-multiclass-v2" / "val" / "images"
RESULTS_DIR = ROOT / "results" / "regression"

CONF_THRESHOLD = 0.25
RECALL_DROP_THRESHOLD = 0.05  # Fail if recall drops more than 5%


def run_inference(model: YOLO, image_dir: Path, conf: float) -> dict:
    """Run inference on all images and return stats."""
    images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))

    total_images = len(images)
    images_with_detections = 0
    total_detections = 0
    class_counts = Counter()
    confidences = []

    for img_path in images:
        try:
            results = model.predict(str(img_path), conf=conf, verbose=False)
            n_dets = 0
            for result in results:
                if result.boxes is not None:
                    n_dets += len(result.boxes)
                    for cls, c in zip(result.boxes.cls, result.boxes.conf):
                        class_name = result.names.get(int(cls.item()), "unknown")
                        class_counts[class_name] += 1
                        confidences.append(float(c.item()))
            if n_dets > 0:
                images_with_detections += 1
            total_detections += n_dets
        except Exception as e:
            print(f"  Error on {img_path.name}: {e}")

    detection_rate = images_with_detections / total_images if total_images > 0 else 0
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    return {
        "total_images": total_images,
        "images_with_detections": images_with_detections,
        "detection_rate": round(detection_rate, 4),
        "total_detections": total_detections,
        "avg_confidence": round(avg_conf, 4),
        "class_distribution": dict(class_counts),
    }


def regression_test(
    old_weights: Path,
    new_weights: Path,
    test_dir: Path,
) -> bool:
    """Compare old and new models. Returns True if new model passes."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not test_dir.exists():
        print(f"ERROR: Test directory not found: {test_dir}")
        return False

    print(f"Test images: {test_dir}")
    print(f"Old weights: {old_weights}")
    print(f"New weights: {new_weights}")

    # Run old model
    print("\nRunning OLD model...")
    old_model = YOLO(str(old_weights))
    old_stats = run_inference(old_model, test_dir, CONF_THRESHOLD)

    # Run new model
    print("Running NEW model...")
    new_model = YOLO(str(new_weights))
    new_stats = run_inference(new_model, test_dir, CONF_THRESHOLD)

    # Compare
    print(f"\n{'='*60}")
    print("REGRESSION TEST RESULTS")
    print(f"{'='*60}")

    print(f"\n  {'Metric':<25s} {'Old':>10s} {'New':>10s}")
    print(f"  {'-'*45}")
    for key in ["total_images", "images_with_detections", "detection_rate",
                 "total_detections", "avg_confidence"]:
        old_val = old_stats[key]
        new_val = new_stats[key]
        print(f"  {key:<25s} {str(old_val):>10s} {str(new_val):>10s}")

    print(f"\n  Old class distribution: {old_stats['class_distribution']}")
    print(f"  New class distribution: {new_stats['class_distribution']}")

    # Check regression
    recall_drop = old_stats["detection_rate"] - new_stats["detection_rate"]
    passed = recall_drop <= RECALL_DROP_THRESHOLD

    multiclass = len(new_stats["class_distribution"]) > 1

    print(f"\n  Detection rate change: {recall_drop:+.4f}")
    print(f"  Multiclass output:    {'YES' if multiclass else 'NO (still single-class)'}")
    print(f"  Status:               {'PASS' if passed else 'FAIL - recall dropped too much'}")

    # Save results
    report = {
        "old_stats": old_stats,
        "new_stats": new_stats,
        "recall_drop": round(recall_drop, 4),
        "multiclass": multiclass,
        "passed": passed,
    }
    with open(RESULTS_DIR / "regression_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return passed


def main():
    parser = argparse.ArgumentParser(description="Regression Test")
    parser.add_argument("--old-weights", type=str, default=str(OLD_WEIGHTS))
    parser.add_argument("--new-weights", type=str, default=str(NEW_WEIGHTS))
    parser.add_argument("--test-dir", type=str, default=str(TEST_DIR))
    args = parser.parse_args()

    passed = regression_test(
        Path(args.old_weights),
        Path(args.new_weights),
        Path(args.test_dir),
    )

    exit(0 if passed else 1)


if __name__ == "__main__":
    main()
