#!/usr/bin/env python3
"""
Progressive Training for YOLOv8 Multiclass Detection
=====================================================
Two-stage training for better convergence:
  Stage 1: Lower resolution (480px), higher batch → learn features fast
  Stage 2: Full resolution (640px), lower batch → fine-tune details

Usage:
    python train_progressive.py
    python train_progressive.py --model s --stage1-epochs 100 --stage2-epochs 50
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATASET_CONFIG = ROOT / "detection-model" / "dataset-multiclass-v2" / "data.yaml"
RUNS_DIR = ROOT / "detection-model" / "runs"
PRETRAINED_DIR = ROOT / "detection-model" / "pretrained"

RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_progressive(
    model_size: str = "s",
    stage1_epochs: int = 100,
    stage2_epochs: int = 50,
    stage1_imgsz: int = 480,
    stage2_imgsz: int = 640,
) -> None:
    set_seed(RANDOM_SEED)
    device = get_device()

    if not DATASET_CONFIG.exists():
        print(f"ERROR: Dataset not found: {DATASET_CONFIG}")
        print("Run create_proper_multiclass_dataset.py first.")
        return

    # =========================================================================
    # Stage 1: Low-res, high batch — learn broad features
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"STAGE 1: {stage1_epochs} epochs @ {stage1_imgsz}px")
    print(f"{'='*60}\n")

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    pretrained = PRETRAINED_DIR / f"yolov8{model_size}.pt"
    model = YOLO(str(pretrained) if pretrained.exists() else f"yolov8{model_size}.pt")

    batch1 = 24 if device == "mps" else 32

    model.train(
        data=str(DATASET_CONFIG),
        epochs=stage1_epochs,
        imgsz=stage1_imgsz,
        batch=batch1,
        device=device,
        patience=20,
        cache=False,
        project=str(RUNS_DIR),
        name="progressive_stage1",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        cos_lr=True,
        warmup_epochs=5,
        lr0=0.01,
        lrf=0.01,
        cls=1.5,
        box=7.5,
        label_smoothing=0.1,
        dropout=0.1,
        mosaic=1.0,
        close_mosaic=15,
        mixup=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.2,
        scale=0.5,
        shear=5.0,
        flipud=0.1,
        fliplr=0.5,
        amp=True,
        verbose=True,
        val=True,
        plots=True,
        save=True,
    )

    stage1_best = RUNS_DIR / "progressive_stage1" / "weights" / "best.pt"
    if not stage1_best.exists():
        print("ERROR: Stage 1 did not produce weights.")
        return

    # =========================================================================
    # Stage 2: Full-res, lower batch — fine-tune at high resolution
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"STAGE 2: {stage2_epochs} epochs @ {stage2_imgsz}px (from stage 1 best)")
    print(f"{'='*60}\n")

    model = YOLO(str(stage1_best))

    batch2 = 16 if device == "mps" else 24

    model.train(
        data=str(DATASET_CONFIG),
        epochs=stage2_epochs,
        imgsz=stage2_imgsz,
        batch=batch2,
        device=device,
        patience=15,
        cache=False,
        project=str(RUNS_DIR),
        name="progressive_stage2",
        exist_ok=True,
        optimizer="AdamW",
        cos_lr=True,
        warmup_epochs=3,
        lr0=0.001,  # Lower LR for fine-tuning
        lrf=0.01,
        cls=1.5,
        box=7.5,
        label_smoothing=0.05,  # Less smoothing for fine-tuning
        dropout=0.1,
        mosaic=0.5,  # Less mosaic for fine-tuning
        close_mosaic=10,
        mixup=0.05,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        shear=3.0,
        flipud=0.05,
        fliplr=0.5,
        amp=True,
        verbose=True,
        val=True,
        plots=True,
        save=True,
    )

    stage2_best = RUNS_DIR / "progressive_stage2" / "weights" / "best.pt"
    if stage2_best.exists():
        # Copy final model to the main train weights location
        final_dst = RUNS_DIR / "train" / "weights"
        final_dst.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(stage2_best, final_dst / "best.pt")
        print(f"\nFinal model saved to: {final_dst / 'best.pt'}")

    print("\nProgressive training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Progressive YOLOv8 Training")
    parser.add_argument("--model", default="s", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--stage1-epochs", type=int, default=100)
    parser.add_argument("--stage2-epochs", type=int, default=50)
    parser.add_argument("--stage1-imgsz", type=int, default=480)
    parser.add_argument("--stage2-imgsz", type=int, default=640)
    args = parser.parse_args()

    train_progressive(
        model_size=args.model,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage1_imgsz=args.stage1_imgsz,
        stage2_imgsz=args.stage2_imgsz,
    )
