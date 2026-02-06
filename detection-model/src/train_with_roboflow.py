#!/usr/bin/env python3
"""
Roboflow Integration for Crash2Cost
====================================
Upload dataset, train, and download models using Roboflow.

Usage:
    1. Get API key from: https://app.roboflow.com/settings/api
    2. Run: python train_with_roboflow.py --api-key YOUR_KEY
"""

import os
import argparse
from pathlib import Path
from roboflow import Roboflow

ROOT = Path(__file__).resolve().parents[1]


def upload_dataset_to_roboflow(api_key: str, project_name: str = "crash2cost-damage"):
    """Upload local dataset to Roboflow for augmentation and training."""
    
    print(" Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)
    
    # Get or create workspace
    workspace = rf.workspace()
    print(f" Workspace: {workspace.name}")
    
    # Create project
    print(f" Creating project: {project_name}")
    project = workspace.create_project(
        project_name=project_name,
        project_type="object-detection",
        project_license="MIT"
    )
    
    # Upload images and annotations
    dataset_path = ROOT / "detection-model" / "dataset-final"
    
    print("\n Uploading training data...")
    train_images = list((dataset_path / "train" / "images").glob("*.jpg"))
    train_labels = dataset_path / "train" / "labels"
    
    for i, img_path in enumerate(train_images):
        label_path = train_labels / f"{img_path.stem}.txt"
        if label_path.exists():
            project.upload(
                image_path=str(img_path),
                annotation_path=str(label_path),
                annotation_format="yolov8"
            )
        if (i + 1) % 100 == 0:
            print(f"   Uploaded {i + 1}/{len(train_images)} images...")
    
    print(f" Uploaded {len(train_images)} training images")
    
    return project


def download_roboflow_dataset(api_key: str, workspace: str, project: str, version: int = 1):
    """Download a dataset from Roboflow."""
    
    print(" Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)
    
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download("yolov8")
    
    print(f" Dataset downloaded to: {dataset.location}")
    return dataset


def train_with_roboflow(api_key: str, workspace: str, project: str, version: int = 1):
    """Train YOLO model using Roboflow dataset."""
    
    from ultralytics import YOLO
    
    print(" Downloading dataset from Roboflow...")
    rf = Roboflow(api_key=api_key)
    
    project_obj = rf.workspace(workspace).project(project)
    dataset = project_obj.version(version).download("yolov8")
    
    print(f"\n Dataset location: {dataset.location}")
    
    # Train with Ultralytics
    print("\n Starting training...")
    model = YOLO("yolov8s.pt")
    
    results = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device="mps",
        patience=20,
        save=True,
        project=str(ROOT / "detection-model" / "runs"),
        name="roboflow-trained",
        exist_ok=True
    )
    
    print(" Training complete!")
    return results


def use_public_car_damage_dataset(api_key: str):
    """
    Use a public car damage detection dataset from Roboflow Universe.
    Several good options:
    - roboflow/car-damage-detection
    - roboflow-universe/car-damage-b0ixc
    """
    
    from ultralytics import YOLO
    
    print(" Downloading public car damage dataset from Roboflow Universe...")
    rf = Roboflow(api_key=api_key)
    
    # Popular car damage datasets on Roboflow Universe:
    datasets = [
        ("car-damage-b0ixc", "car-damage-b0ixc"),  # 3000+ images
        ("vehicle-damage-detection-eocfq", "vehicle-damage-detection"),  # Multi-class
        ("car-damage-detection-weoez", "car-damage-detection"),  # Good annotations
    ]
    
    print("\n Available public datasets:")
    for i, (ws, proj) in enumerate(datasets, 1):
        print(f"   {i}. {proj}")
    
    # Try the first one
    try:
        project = rf.workspace("roboflow-universe").project(datasets[0][1])
        versions = project.versions
        latest = max([v.version for v in versions])
        dataset = project.version(latest).download("yolov8")
        
        print(f"\n Downloaded: {dataset.location}")
        print(f"   Classes: {dataset.classes}")
        print(f"   Train images: {len(list(Path(dataset.location).glob('train/images/*')))}")
        
        return dataset
        
    except Exception as e:
        print(f" Could not download: {e}")
        print("\n To use Roboflow Universe, go to:")
        print("   https://universe.roboflow.com/search?q=car+damage")
        print("   Find a dataset, click 'Download Dataset'  YOLOv8 format")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roboflow Integration for Crash2Cost")
    parser.add_argument("--api-key", required=True, help="Roboflow API key")
    parser.add_argument("--action", choices=["upload", "download", "train", "public"],
                       default="public", help="Action to perform")
    parser.add_argument("--workspace", default="", help="Roboflow workspace name")
    parser.add_argument("--project", default="crash2cost", help="Roboflow project name")
    parser.add_argument("--version", type=int, default=1, help="Dataset version")
    
    args = parser.parse_args()
    
    print(" Crash2Cost - Roboflow Integration")
    print("=" * 50)
    
    if args.action == "upload":
        upload_dataset_to_roboflow(args.api_key, args.project)
    elif args.action == "download":
        download_roboflow_dataset(args.api_key, args.workspace, args.project, args.version)
    elif args.action == "train":
        train_with_roboflow(args.api_key, args.workspace, args.project, args.version)
    elif args.action == "public":
        use_public_car_damage_dataset(args.api_key)
