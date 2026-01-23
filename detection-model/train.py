#!/usr/bin/env python3
"""
Detection Model Training - Car Damage Detection with YOLOv8
============================================================
Uses Ultralytics YOLO library for object detection training.
No custom implementations - leverages the well-tested library.

Usage:
    python train.py                          # Train with defaults
    python train.py --epochs 100 --batch 16  # Custom settings
    python train.py --resume                 # Resume from checkpoint
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATASET_CONFIG = ROOT / "detection-model" / "car-damage-detector-1" / "data.yaml"
RUNS_DIR = ROOT / "detection-model" / "runs"


def get_device():
    """Auto-detect best available device."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train(
    epochs: int = 50,
    batch_size: int = 32,
    img_size: int = 640,
    model_size: str = "s",
    resume: bool = False,
    patience: int = 15,
):
    """
    Train YOLOv8 model for car damage detection.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        model_size: YOLO model size (n/s/m/l/x)
        resume: Resume from last checkpoint
        patience: Early stopping patience
    """
    device = get_device()
    print(f"\n{'='*60}")
    print("🚗 Car Damage Detection - YOLOv8 Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: YOLOv8{model_size}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {img_size}")
    print(f"{'='*60}\n")
    
    # Initialize model
    if resume:
        checkpoint = RUNS_DIR / "train" / "weights" / "last.pt"
        if checkpoint.exists():
            print(f"📂 Resuming from: {checkpoint}")
            model = YOLO(str(checkpoint))
        else:
            print("⚠️ No checkpoint found, starting fresh")
            model = YOLO(f"yolov8{model_size}.pt")
    else:
        model = YOLO(f"yolov8{model_size}.pt")
    
    # Train
    results = model.train(
        data=str(DATASET_CONFIG),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        patience=patience,
        cache=True,  # Cache images in RAM for faster training
        project=str(RUNS_DIR),
        name="train",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        val=True,
        plots=True,
        save=True,
    )
    
    print(f"\n✅ Training complete!")
    print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"Weights saved to: {RUNS_DIR}/train/weights/")
    
    return results


def validate(weights: str = None):
    """Validate trained model on test set."""
    if weights is None:
        weights = RUNS_DIR / "train" / "weights" / "best.pt"
    
    model = YOLO(str(weights))
    results = model.val(data=str(DATASET_CONFIG))
    return results


def predict(source: str, weights: str = None, conf: float = 0.25, save: bool = True):
    """Run inference on images."""
    if weights is None:
        weights = RUNS_DIR / "train" / "weights" / "best.pt"
    
    model = YOLO(str(weights))
    results = model.predict(
        source=source,
        conf=conf,
        save=save,
        project=str(RUNS_DIR),
        name="predict",
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Detection Training")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--model", type=str, default="s", choices=["n", "s", "m", "l", "x"],
                       help="YOLO model size")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        model_size=args.model,
        resume=args.resume,
        patience=args.patience,
    )
