#!/usr/bin/env python3
"""
Hyperparameter Tuning Script using Optuna
==========================================
Implements Grid Search / Random Search / Bayesian optimization for:
1. Detection Model (YOLO) - learning rate, batch size, augmentation
2. Severity Classifier (ResNet) - learning rate, optimizer, dropout
3. Cost Regressor (RandomForest/GradientBoosting) - n_estimators, max_depth, etc.

Required for thesis: Hyperparameter optimization with Early-Stopping and Model-Checkpoint
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================
# 1. SEVERITY CLASSIFIER HYPERPARAMETER TUNING (ResNet18)
# ============================================================

def tune_severity_classifier(n_trials: int = 50):
    """
    Tune hyperparameters for severity classification model.
    Uses Optuna with Bayesian optimization (TPE sampler).
    """
    print("=" * 60)
    print("🔧 Tuning Severity Classifier (ResNet18)")
    print("=" * 60)
    
    DATA_DIR = ROOT / "severity-model" / "dataset"
    MODEL_DIR = ROOT / "severity-model" / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    
    def objective(trial):
        # Hyperparameters to tune
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "AdamW"])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        
        # Data augmentation parameters
        rotation = trial.suggest_int("rotation_degrees", 5, 30)
        color_jitter = trial.suggest_float("color_jitter", 0.1, 0.4)
        
        # Transforms with tuned augmentation
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(rotation),
            transforms.ColorJitter(brightness=color_jitter, contrast=color_jitter),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        train_dataset = datasets.ImageFolder(DATA_DIR / 'train', transform=train_transform)
        val_dataset = datasets.ImageFolder(DATA_DIR / 'val', transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        num_classes = len(train_dataset.classes)
        
        # Model with dropout (no pretrained weights to avoid SSL issues)
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.fc.in_features, num_classes)
        )
        model = model.to(DEVICE)
        
        # Optimizer selection
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        criterion = nn.CrossEntropyLoss()
        
        # Training with early stopping
        best_val_acc = 0.0
        patience = 5
        patience_counter = 0
        
        for epoch in range(30):  # Max epochs
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            val_acc = 100.0 * correct / total
            
            # Report intermediate value for pruning
            trial.report(val_acc, epoch)
            
            # Prune trial if needed
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        return best_val_acc
    
    # Create study with Bayesian optimization
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("\n" + "=" * 60)
    print("📊 Best Trial Results:")
    print("=" * 60)
    print(f"  Best Validation Accuracy: {study.best_trial.value:.2f}%")
    print(f"  Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best hyperparameters
    with open(MODEL_DIR / "best_hyperparameters.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
    
    return study.best_trial.params


# ============================================================
# 2. COST REGRESSOR HYPERPARAMETER TUNING (RF/GB)
# ============================================================

def tune_cost_regressor(n_trials: int = 100):
    """
    Tune hyperparameters for cost estimation model.
    Compares RandomForest vs GradientBoosting.
    """
    print("\n" + "=" * 60)
    print("🔧 Tuning Cost Regressor (RF/GB)")
    print("=" * 60)
    
    DATA_PATH = ROOT / "cost-model" / "dataset" / "detailed_repair_costs.csv"
    MODEL_DIR = ROOT / "cost-model" / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    
    part_encoder = LabelEncoder()
    segment_encoder = LabelEncoder()
    
    df['Part_Encoded'] = part_encoder.fit_transform(df['Part_Name'])
    df['Segment_Encoded'] = segment_encoder.fit_transform(df['Car_Segment'])
    
    X = df[['Part_Encoded', 'Severity', 'Segment_Encoded']].values
    y = df['Estimated_Cost'].values
    
    def objective(trial):
        # Choose model type
        model_type = trial.suggest_categorical("model_type", ["RandomForest", "GradientBoosting"])
        
        if model_type == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("rf_n_estimators", 50, 500),
                "max_depth": trial.suggest_int("rf_max_depth", 5, 50),
                "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None]),
                "random_state": 42,
                "n_jobs": -1
            }
            model = RandomForestRegressor(**params)
        else:
            params = {
                "n_estimators": trial.suggest_int("gb_n_estimators", 50, 500),
                "max_depth": trial.suggest_int("gb_max_depth", 3, 15),
                "learning_rate": trial.suggest_float("gb_learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("gb_subsample", 0.6, 1.0),
                "min_samples_split": trial.suggest_int("gb_min_samples_split", 2, 20),
                "random_state": 42
            }
            model = GradientBoostingRegressor(**params)
        
        # Cross-validation score (negative MAE, so higher is better)
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
        return -scores.mean()  # Return positive MAE (lower is better, so minimize)
    
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("\n" + "=" * 60)
    print("📊 Best Trial Results:")
    print("=" * 60)
    print(f"  Best MAE: ₪{study.best_trial.value:.2f}")
    print(f"  Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best hyperparameters
    with open(MODEL_DIR / "best_hyperparameters.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
    
    # Train final model with best parameters
    best_params = study.best_trial.params
    if best_params["model_type"] == "RandomForest":
        final_model = RandomForestRegressor(
            n_estimators=best_params.get("rf_n_estimators", 200),
            max_depth=best_params.get("rf_max_depth", 20),
            min_samples_split=best_params.get("rf_min_samples_split", 5),
            min_samples_leaf=best_params.get("rf_min_samples_leaf", 2),
            max_features=best_params.get("rf_max_features"),
            random_state=42,
            n_jobs=-1
        )
    else:
        final_model = GradientBoostingRegressor(
            n_estimators=best_params.get("gb_n_estimators", 200),
            max_depth=best_params.get("gb_max_depth", 7),
            learning_rate=best_params.get("gb_learning_rate", 0.1),
            subsample=best_params.get("gb_subsample", 0.8),
            min_samples_split=best_params.get("gb_min_samples_split", 5),
            random_state=42
        )
    
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_DIR / "cost_estimator_optimized.pkl")
    print(f"\n✅ Saved optimized model to {MODEL_DIR / 'cost_estimator_optimized.pkl'}")
    
    return study.best_trial.params


# ============================================================
# 3. YOLO DETECTION HYPERPARAMETER TUNING
# ============================================================

def tune_yolo_detector(n_trials: int = 20):
    """
    Tune hyperparameters for YOLO detection model.
    Uses Ultralytics tuning or manual grid search.
    """
    print("\n" + "=" * 60)
    print("🔧 Tuning YOLO Detector")
    print("=" * 60)
    
    from ultralytics import YOLO
    
    DATASET_PATH = ROOT / "detection-model" / "dataset-final" / "data.yaml"
    OUTPUT_DIR = ROOT / "detection-model" / "runs" / "hyperparameter_tuning"
    
    def objective(trial):
        # Hyperparameters to tune
        lr0 = trial.suggest_float("lr0", 1e-4, 1e-1, log=True)
        lrf = trial.suggest_float("lrf", 0.01, 0.2)
        momentum = trial.suggest_float("momentum", 0.8, 0.98)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        warmup_epochs = trial.suggest_int("warmup_epochs", 1, 5)
        box_gain = trial.suggest_float("box", 5.0, 10.0)
        cls_gain = trial.suggest_float("cls", 0.3, 1.0)
        
        # Augmentation parameters
        hsv_h = trial.suggest_float("hsv_h", 0.0, 0.1)
        hsv_s = trial.suggest_float("hsv_s", 0.0, 0.9)
        hsv_v = trial.suggest_float("hsv_v", 0.0, 0.9)
        degrees = trial.suggest_float("degrees", 0.0, 45.0)
        translate = trial.suggest_float("translate", 0.0, 0.3)
        scale = trial.suggest_float("scale", 0.0, 0.9)
        fliplr = trial.suggest_float("fliplr", 0.0, 1.0)
        mosaic = trial.suggest_float("mosaic", 0.0, 1.0)
        
        model = YOLO("yolov8s.pt")
        
        try:
            results = model.train(
                data=str(DATASET_PATH),
                epochs=20,  # Shorter for tuning
                imgsz=640,
                batch=16,
                lr0=lr0,
                lrf=lrf,
                momentum=momentum,
                weight_decay=weight_decay,
                warmup_epochs=warmup_epochs,
                box=box_gain,
                cls=cls_gain,
                hsv_h=hsv_h,
                hsv_s=hsv_s,
                hsv_v=hsv_v,
                degrees=degrees,
                translate=translate,
                scale=scale,
                fliplr=fliplr,
                mosaic=mosaic,
                project=str(OUTPUT_DIR),
                name=f"trial_{trial.number}",
                exist_ok=True,
                verbose=False
            )
            
            # Return mAP50 as objective
            return results.results_dict.get("metrics/mAP50(B)", 0.0)
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0
    
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("\n" + "=" * 60)
    print("📊 Best Trial Results:")
    print("=" * 60)
    print(f"  Best mAP50: {study.best_trial.value:.4f}")
    print(f"  Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best hyperparameters
    with open(OUTPUT_DIR / "best_hyperparameters.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
    
    return study.best_trial.params


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for Crash2Cost")
    parser.add_argument("--model", choices=["severity", "cost", "yolo", "all"], 
                       default="all", help="Which model to tune")
    parser.add_argument("--trials", type=int, default=50, 
                       help="Number of Optuna trials")
    args = parser.parse_args()
    
    print("🚀 Crash2Cost Hyperparameter Tuning")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Trials: {args.trials}")
    print("=" * 60)
    
    if args.model in ["severity", "all"]:
        severity_params = tune_severity_classifier(n_trials=args.trials)
    
    if args.model in ["cost", "all"]:
        cost_params = tune_cost_regressor(n_trials=args.trials)
    
    if args.model in ["yolo", "all"]:
        yolo_params = tune_yolo_detector(n_trials=min(args.trials, 20))
    
    print("\n" + "=" * 60)
    print("✅ Hyperparameter tuning complete!")
    print("=" * 60)
