#!/usr/bin/env python3
"""
Test script for Crash2Cost models.
Verifies that all models can load and run inference.
"""

import sys
from pathlib import Path

# Add parent to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_detection_model():
    """Test YOLOv8 detection model."""
    print(" Testing Detection Model...")
    
    try:
        from ultralytics import YOLO
        weights = ROOT / "detection-model" / "runs" / "train" / "weights" / "best.pt"
        
        if weights.exists():
            model = YOLO(str(weights))
            print(f"    Loaded: {weights}")
        else:
            model = YOLO("yolov8n.pt")  # Fallback to pretrained
            print(f"    Using pretrained YOLOv8n (no custom weights found)")
        
        # Quick inference test
        import numpy as np
        from PIL import Image
        test_img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        results = model.predict(test_img, verbose=False)
        print("    Inference works!")
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


def test_severity_model():
    """Test severity classification model."""
    print("\n Testing Severity Model...")
    
    try:
        import torch
        from torchvision import models
        import torch.nn as nn
        
        # Try multiple possible weight locations
        possible_paths = [
            ROOT / "severity-model" / "models" / "best_model.pt",
            ROOT / "severity-model" / "models" / "damage_classifier_best.pt",
        ]
        
        weights = None
        for path in possible_paths:
            if path.exists():
                weights = path
                break
        
        if weights is None:
            print(f"    No weights found")
            print("    Train first with: python severity-model/train.py")
            return False
        
        checkpoint = torch.load(weights, map_location="cpu")
        classes = checkpoint["classes"]
        print(f"    Loaded weights with {len(classes)} classes: {classes}")
        
        # Build model - handle different checkpoint formats
        model = models.resnet18(weights=None)
        state_dict = checkpoint["model_state_dict"]
        
        if any("fc.0" in k or "fc.1" in k for k in state_dict.keys()):
            model.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(model.fc.in_features, len(classes)),
            )
        else:
            model.fc = nn.Linear(model.fc.in_features, len(classes))
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        # Quick inference test
        test_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(test_input)
        print(f"    Inference works! Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


def test_cost_model():
    """Test cost estimation model."""
    print("\n Testing Cost Model...")
    
    try:
        import joblib
        
        model_path = ROOT / "cost-model" / "models" / "cost_estimator.pkl"
        part_enc_path = ROOT / "cost-model" / "models" / "part_encoder.pkl"
        seg_enc_path = ROOT / "cost-model" / "models" / "segment_encoder.pkl"
        
        if not model_path.exists():
            print(f"    No model found at {model_path}")
            print("    Train first with: python cost-model/train.py")
            return False
        
        model = joblib.load(model_path)
        part_encoder = joblib.load(part_enc_path)
        segment_encoder = joblib.load(seg_enc_path)
        
        print(f"    Loaded model: {type(model).__name__}")
        print(f"    Parts: {list(part_encoder.classes_)}")
        print(f"    Segments: {list(segment_encoder.classes_)}")
        
        # Quick prediction test
        part_enc = part_encoder.transform(["Front Bumper"])[0]
        seg_enc = segment_encoder.transform(["Family"])[0]
        cost = model.predict([[part_enc, 3, seg_enc]])[0]
        print(f"    Test prediction: Front Bumper, Severity 3, Family = ₪{cost:,.0f}")
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


def test_pipeline():
    """Test full pipeline."""
    print("\n Testing Full Pipeline...")
    
    try:
        from pipeline import Crash2CostPipeline
        
        pipeline = Crash2CostPipeline()
        print("    Pipeline initialized successfully!")
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print(" Crash2Cost Model Tests")
    print("=" * 60)
    
    results = {
        "Detection": test_detection_model(),
        "Severity": test_severity_model(),
        "Cost": test_cost_model(),
        "Pipeline": test_pipeline(),
    }
    
    print("\n" + "=" * 60)
    print(" Summary")
    print("=" * 60)
    
    for name, passed in results.items():
        status = " PASS" if passed else " FAIL"
        print(f"   {name}: {status}")
    
    all_passed = all(results.values())
    print(f"\n{' All tests passed!' if all_passed else ' Some tests failed'}")
