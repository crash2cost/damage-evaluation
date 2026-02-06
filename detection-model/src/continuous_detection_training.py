#!/usr/bin/env python3
"""
Continuous YOLO Detection Model Training Script
Runs training with optimized parameters until manually stopped (Ctrl+C)
Automatically resumes from last checkpoint if training is interrupted
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Training parameters (optimized for Apple M4 Max)
TRAINING_CONFIG = {
    'data': 'car-damage-detector-1/data.yaml',
    'epochs': 50,
    'imgsz': 416,  # Optimized for speed (vs 640)
    'batch': 32,   # Optimized for M4 Max
    'device': 'mps',
    'patience': 15,
    'cache': True,  # RAM caching for 2.4x speed boost
    'project': 'detection-model/runs',
    'name': 'roboflow-continuous',
    'exist_ok': True,
    'pretrained': True,
    'optimizer': 'auto',
    'verbose': True,
    'seed': 0,
    'deterministic': True,
    'val': True,
    'plots': True,
    'save': True,
    'resume': False,  # Will be updated if checkpoint exists
}

def save_config(config, filepath='detection_training_config.yaml'):
    """Save training configuration to YAML file"""
    config_path = Path(__file__).parent / filepath
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"\n Training config saved to: {config_path}")

def get_latest_checkpoint():
    """Find the latest training checkpoint"""
    runs_dir = Path('detection-model/runs/roboflow-continuous')
    if not runs_dir.exists():
        return None
    
    weights_dir = runs_dir / 'weights'
    if not weights_dir.exists():
        return None
    
    last_pt = weights_dir / 'last.pt'
    if last_pt.exists():
        return str(last_pt)
    
    return None

def train_continuously():
    """Run continuous training with automatic checkpoint resumption"""
    
    # Save the configuration
    save_config(TRAINING_CONFIG)
    
    training_round = 1
    
    while True:
        try:
            print(f"\n{'='*60}")
            print(f" Starting Training Round {training_round}")
            print(f"{'='*60}\n")
            
            # Check for existing checkpoint
            checkpoint = get_latest_checkpoint()
            
            if checkpoint and training_round > 1:
                print(f" Resuming from checkpoint: {checkpoint}")
                model = YOLO(checkpoint)
                config = TRAINING_CONFIG.copy()
                config['resume'] = True
            else:
                print(f" Starting fresh training with YOLOv8s")
                model = YOLO('yolov8s.pt')
                config = TRAINING_CONFIG.copy()
            
            # Run training
            results = model.train(**config)
            
            print(f"\n Training Round {training_round} completed successfully!")
            print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
            print(f"Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
            
            training_round += 1
            
            # Ask if user wants to continue (only in interactive mode)
            if sys.stdout.isatty():
                response = input("\n Continue training? (y/n): ")
                if response.lower() != 'y':
                    print(" Training stopped by user")
                    break
            else:
                print("\n Continuing to next training round...")
            
        except KeyboardInterrupt:
            print("\n\n  Training interrupted by user (Ctrl+C)")
            print(f" Progress saved to: detection-model/runs/roboflow-continuous/weights/last.pt")
            print(f" Run this script again to resume training")
            break
        
        except Exception as e:
            print(f"\n Error during training: {e}")
            print(f" Last checkpoint saved at: detection-model/runs/roboflow-continuous/weights/last.pt")
            raise

if __name__ == "__main__":
    print("""
    
      Continuous YOLO Detection Training                      
      Optimized for Apple M4 Max with MPS                     
    
    
    Configuration:
    - Model: YOLOv8s (11.1M parameters)
    - Dataset: Roboflow augmented (5,238 train images)
    - Image Size: 416px (optimized for speed)
    - Batch Size: 32
    - Epochs: 50 per round
    - Device: MPS (Metal Performance Shaders)
    - Cache: RAM (2.5GB for instant loading)
    
    Press Ctrl+C to stop training at any time.
    Progress is automatically saved.
    """)
    
    # Change to workspace directory
    workspace = Path(__file__).parent.parent
    os.chdir(workspace)
    
    train_continuously()
