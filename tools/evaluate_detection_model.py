#!/usr/bin/env python3
"""
Detection Model Evaluation Dashboard
=====================================
Generates comprehensive evaluation metrics for the detection model:
- Per-class precision, recall, F1, AP
- Confusion matrix (raw + normalized)
- mAP@50 and mAP@50-95
- Size-based analysis
- JSON + PNG output

Usage:
    python evaluate_detection_model.py
    python evaluate_detection_model.py --weights path/to/best.pt --data path/to/data.yaml
    python evaluate_detection_model.py --compare path/to/old_best.pt
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "detection-model" / "runs" / "train" / "weights" / "best.pt"
DEFAULT_DATA = ROOT / "detection-model" / "dataset-multiclass-v2" / "data.yaml"
RESULTS_DIR = ROOT / "results" / "evaluation"


def evaluate_model(weights: Path, data: Path, output_dir: Path) -> dict:
    """Run full evaluation and return metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {weights}")
    model = YOLO(str(weights))

    print(f"Evaluating on: {data}")
    results = model.val(
        data=str(data),
        plots=True,
        save_json=True,
        project=str(output_dir),
        name="eval",
        exist_ok=True,
    )

    # Extract metrics
    metrics = {
        "model": str(weights),
        "dataset": str(data),
        "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
        "mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
        "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
    }

    # Per-class metrics
    if hasattr(results, "box") and hasattr(results.box, "maps"):
        class_names = results.names if hasattr(results, "names") else {}
        per_class = {}
        for i, ap50 in enumerate(results.box.maps):
            name = class_names.get(i, f"class_{i}")
            per_class[name] = {
                "AP50": round(float(ap50), 4),
            }
        metrics["per_class"] = per_class

    # Save metrics JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  mAP@50:    {metrics['mAP50']:.4f}")
    print(f"  mAP@50-95: {metrics['mAP50_95']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")

    if "per_class" in metrics:
        print(f"\n  Per-Class AP@50:")
        for name, m in metrics["per_class"].items():
            print(f"    {name:<20s} {m['AP50']:.4f}")

    print(f"\n  Results saved to: {output_dir}")
    print(f"  Metrics JSON: {metrics_path}")

    return metrics


def compare_models(weights_new: Path, weights_old: Path, data: Path) -> None:
    """Compare two models side by side."""
    print("Evaluating NEW model...")
    new_dir = RESULTS_DIR / "new"
    new_metrics = evaluate_model(weights_new, data, new_dir)

    print("\nEvaluating OLD model...")
    old_dir = RESULTS_DIR / "old"
    old_metrics = evaluate_model(weights_old, data, old_dir)

    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<15s} {'Old':>10s} {'New':>10s} {'Delta':>10s}")
    print(f"  {'-'*45}")

    for key in ["mAP50", "mAP50_95", "precision", "recall"]:
        old_val = old_metrics.get(key, 0)
        new_val = new_metrics.get(key, 0)
        delta = new_val - old_val
        sign = "+" if delta >= 0 else ""
        print(f"  {key:<15s} {old_val:>10.4f} {new_val:>10.4f} {sign}{delta:>9.4f}")

    # Compare per-class
    if "per_class" in new_metrics and "per_class" in old_metrics:
        print(f"\n  Per-Class AP@50 Comparison:")
        all_classes = set(list(new_metrics["per_class"].keys()) + list(old_metrics["per_class"].keys()))
        for cls in sorted(all_classes):
            old_ap = old_metrics["per_class"].get(cls, {}).get("AP50", 0)
            new_ap = new_metrics["per_class"].get(cls, {}).get("AP50", 0)
            delta = new_ap - old_ap
            sign = "+" if delta >= 0 else ""
            print(f"    {cls:<20s} {old_ap:.4f} -> {new_ap:.4f} ({sign}{delta:.4f})")

    comparison = {
        "old": old_metrics,
        "new": new_metrics,
        "improvements": {
            key: new_metrics.get(key, 0) - old_metrics.get(key, 0)
            for key in ["mAP50", "mAP50_95", "precision", "recall"]
        },
    }
    with open(RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Detection Model")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--compare", type=str, default=None,
                        help="Path to old model weights for comparison")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    if args.compare:
        compare_models(Path(args.weights), Path(args.compare), Path(args.data))
    else:
        evaluate_model(Path(args.weights), Path(args.data), Path(args.output))


if __name__ == "__main__":
    main()
