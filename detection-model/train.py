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
import random
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from ultralytics import YOLO

# =============================================================================
# Constants
# =============================================================================

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATASET_CONFIG = ROOT / "detection-model" / "dataset-final" / "data.yaml"
RUNS_DIR = ROOT / "detection-model" / "runs"
PRETRAINED_DIR = ROOT / "detection-model" / "pretrained"

# Training defaults
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_IMAGE_SIZE = 640
DEFAULT_MODEL_SIZE = "n"
DEFAULT_PATIENCE = 15
DEFAULT_CONFIDENCE = 0.25

# Training configuration
NUM_WORKERS = 4
MAX_DETECTIONS = 100
MOSAIC_CLOSE_EPOCHS = 10

# Display
SEPARATOR_WIDTH = 60

# Reproducibility
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """
    Auto-detect best available device for training.

    Returns:
        Device string: 'mps' for Apple Silicon, 'cuda' for NVIDIA, or 'cpu'.
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    img_size: int = DEFAULT_IMAGE_SIZE,
    model_size: str = DEFAULT_MODEL_SIZE,
    resume: bool = False,
    patience: int = DEFAULT_PATIENCE,
) -> object:
    """
    Train YOLOv8 model for car damage detection.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        img_size: Input image size (square).
        model_size: YOLO model size variant (n/s/m/l/x).
        resume: Whether to resume from last checkpoint.
        patience: Early stopping patience (epochs without improvement).

    Returns:
        Training results object from Ultralytics YOLO.
    """
    set_seed(RANDOM_SEED)
    device = get_device()

    print(f"\n{'=' * SEPARATOR_WIDTH}")
    print("Car Damage Detection - YOLOv8 Training")
    print(f"{'=' * SEPARATOR_WIDTH}")
    print(f"Device: {device}")
    print(f"Model: YOLOv8{model_size}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {img_size}")
    print(f"{'=' * SEPARATOR_WIDTH}\n")

    # Initialize model - use pretrained directory to avoid downloads to cwd
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    pretrained_model = PRETRAINED_DIR / f"yolov8{model_size}.pt"

    if resume:
        checkpoint = RUNS_DIR / "train" / "weights" / "last.pt"
        if checkpoint.exists():
            print(f"Resuming from: {checkpoint}")
            model = YOLO(str(checkpoint))
        else:
            print("No checkpoint found, starting fresh")
            model = YOLO(str(pretrained_model) if pretrained_model.exists() else f"yolov8{model_size}.pt")
    else:
        model = YOLO(str(pretrained_model) if pretrained_model.exists() else f"yolov8{model_size}.pt")

    # Train with optimized settings for speed and stability
    results = model.train(
        data=str(DATASET_CONFIG),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        patience=patience,
        cache=False,  # Disable caching to prevent MPS memory issues
        project=str(RUNS_DIR),
        name="train",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        verbose=True,
        val=True,
        plots=True,
        save=True,
        workers=NUM_WORKERS,
        amp=True,  # Mixed precision training
        max_det=MAX_DETECTIONS,
        close_mosaic=MOSAIC_CLOSE_EPOCHS,
        rect=True,  # Rectangular training for faster inference
    )

    print("\nTraining complete!")
    print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"Weights saved to: {RUNS_DIR}/train/weights/")

    return results


def validate(weights: Optional[Union[str, Path]] = None) -> object:
    """
    Validate trained model on test set.

    Args:
        weights: Path to model weights. Defaults to best.pt from training.

    Returns:
        Validation results object from Ultralytics YOLO.
    """
    if weights is None:
        weights = RUNS_DIR / "train" / "weights" / "best.pt"

    model = YOLO(str(weights))
    results = model.val(data=str(DATASET_CONFIG))
    return results


def predict(
    source: str,
    weights: Optional[Union[str, Path]] = None,
    conf: float = DEFAULT_CONFIDENCE,
    save: bool = True,
) -> List:
    """
    Run inference on images.

    Args:
        source: Path to image(s) or directory for inference.
        weights: Path to model weights. Defaults to best.pt from training.
        conf: Confidence threshold for detections.
        save: Whether to save annotated images.

    Returns:
        List of detection results from Ultralytics YOLO.
    """
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
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Image size")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_SIZE, choices=["n", "s", "m", "l", "x"],
                        help="YOLO model size (n=nano/fastest, s=small, m=medium, l=large, x=xlarge)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        model_size=args.model,
        resume=args.resume,
        patience=args.patience,
    )
