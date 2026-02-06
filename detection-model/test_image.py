#!/usr/bin/env python3
"""Test the trained damage detection model on an image."""

from ultralytics import YOLO
from pathlib import Path
import sys

# Paths
MODEL_PATH = Path(__file__).parent / "runs/train/weights/best.pt"
OUTPUT_DIR = Path(__file__).parent / "runs/test_inference"

def test_image(image_path: str):
    """Run inference on an image."""
    print(f"\n{'='*60}")
    print(" Car Damage Detection - Inference")
    print(f"{'='*60}")
    print(f"Model: {MODEL_PATH}")
    print(f"Image: {image_path}")
    print(f"{'='*60}\n")
    
    # Load model
    model = YOLO(str(MODEL_PATH))
    
    # Run inference
    results = model.predict(
        source=image_path,
        save=True,
        conf=0.25,
        project=str(OUTPUT_DIR.parent),
        name="test_inference",
        exist_ok=True
    )
    
    # Print results
    for r in results:
        print(f"\n Detection Results:")
        print(f"   Image size: {r.orig_shape[1]}x{r.orig_shape[0]}")
        print(f"   Detections: {len(r.boxes)}")
        
        if len(r.boxes) == 0:
            print("     No damage detected")
        else:
            for i, box in enumerate(r.boxes):
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = r.names[cls]
                coords = box.xyxy[0].tolist()
                print(f"   [{i+1}] {cls_name}: {conf:.1%} @ [{int(coords[0])}, {int(coords[1])}, {int(coords[2])}, {int(coords[3])}]")
        
        print(f"\n Result saved to: {OUTPUT_DIR}/")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_image.py <image_path>")
        sys.exit(1)
    
    test_image(sys.argv[1])
