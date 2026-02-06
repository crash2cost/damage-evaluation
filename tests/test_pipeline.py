#!/usr/bin/env python3
"""Test the full pipeline on an image."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import Crash2CostPipeline

def test_full_pipeline(image_path: str, car_segment: str = "Family"):
    """Run full damage assessment pipeline on an image."""
    print("Loading pipeline...")
    pipeline = Crash2CostPipeline()
    
    print(f"\nTesting: {image_path}")
    
    # Run full assessment
    result = pipeline.assess_damage(image_path, car_segment=car_segment)
    
    print(f"\n{'='*60}")
    print(" DAMAGE ASSESSMENT RESULTS")
    print(f"{'='*60}")
    print(f"Total damages found: {len(result.damages)}")
    print(f"Total estimated cost: ${result.total_cost:,.2f}")
    print(f"Inference time: {result.inference_time_ms:.1f}ms")
    
    for i, damage in enumerate(result.damages, 1):
        print(f"\n[Damage {i}]")
        print(f"  Detection confidence: {damage.detection.confidence:.1%}")
        print(f"  Damage type: {damage.damage_type}")
        print(f"  Type confidence: {damage.damage_confidence:.1%}")
        print(f"  Severity: {damage.severity}/5")
        print(f"  Repair action: {damage.repair_or_replace}")
        print(f"  Estimated cost: ${damage.estimated_cost:,.2f}")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_pipeline.py <image_path> [car_segment]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    car_segment = sys.argv[2] if len(sys.argv) > 2 else "Family"
    
    test_full_pipeline(image_path, car_segment)
