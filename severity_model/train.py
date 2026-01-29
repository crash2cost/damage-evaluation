#!/usr/bin/env python3
"""
Severity Classification Model Training - Car Damage Severity
============================================================
Uses PyTorch and torchvision's pretrained ResNet models.
Leverages transfer learning from ImageNet pretrained weights.

Classes: bumper_dent, bumper_scratch, door_dent, door_scratch, 
         glass_shatter, head_lamp, tail_lamp

Usage:
    python train.py                          # Train with defaults
    python train.py --epochs 50 --batch 32   # Custom settings
    python train.py --backbone resnet50      # Use ResNet50
"""

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "severity_model" / "dataset"
MODEL_DIR = ROOT / "severity_model" / "models"
RUNS_DIR = ROOT / "severity_model" / "runs"

MODEL_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)


@dataclass
class TrainingConfig:
    """Training configuration with sensible defaults."""
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.0003
    weight_decay: float = 0.001  # Increased weight decay
    dropout_rate: float = 0.4
    label_smoothing: float = 0.1
    backbone: str = "resnet18"
    input_size: int = 224
    seed: int = 42
    patience: int = 10
    freeze_layers: int = 3  # Freeze conv1, bn1, layer1 only


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Auto-detect best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_transforms(input_size: int):
    """Get training and validation transforms with strong augmentation."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0)),  # More aggressive crop
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15),  # Stronger color jitter
        transforms.RandomRotation(30),  # Increased from 20 to 30 degrees
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), shear=15),  # Increased translation and shear
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # NEW: viewing angle variations
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),  # blur for generalization
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),  # Increased from 0.2 to 0.3
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform


def build_model(backbone: str, num_classes: int, dropout_rate: float, device: torch.device, freeze_layers: int = 6):
    """
    Build classification model using torchvision pretrained models.
    
    Leverages ImageNet pretrained weights for transfer learning.
    Freezes early layers to prevent overfitting on small datasets.
    """
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        return model.to(device)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    # Freeze early backbone layers to prevent overfitting
    if freeze_layers > 0:
        layers_to_freeze = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4][:freeze_layers]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        print(f"   Frozen first {freeze_layers} backbone layers")
    
    # Replace classifier head with dropout + new FC layer
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes),
    )
    
    return model.to(device)


def get_weighted_sampler(dataset):
    """Create weighted sampler for class imbalance."""
    class_counts = {}
    for label in dataset.targets:
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    total_samples = len(dataset)
    class_weights = []
    for class_name in dataset.classes:
        count = class_counts.get(class_name, 1)
        class_weights.append(total_samples / count)
    
    sample_weights = [class_weights[label] for label in dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    return sampler, class_weights


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix(loss=f"{running_loss/(pbar.n+1):.4f}", acc=f"{100.*correct/total:.2f}%")
    
    return running_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Validating", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100.0 * correct / total


def train(config: TrainingConfig):
    """
    Main training function.
    
    Args:
        config: Training configuration
    """
    set_seed(config.seed)
    device = get_device()
    
    print(f"\n{'='*60}")
    print("🔍 Severity Classification Model Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Backbone: {config.backbone}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Dropout Rate: {config.dropout_rate}")
    print(f"{'='*60}\n")
    
    # Get transforms
    train_transform, val_transform = get_transforms(config.input_size)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=val_transform)
    
    num_classes = len(train_dataset.classes)
    print(f"Classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create weighted sampler for class imbalance
    sampler, class_weights = get_weighted_sampler(train_dataset)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # Build model
    print(f"\nBuilding {config.backbone} model with pretrained ImageNet weights...")
    model = build_model(config.backbone, num_classes, config.dropout_rate, device, config.freeze_layers)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=config.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    
    # Training loop
    best_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    print("\nStarting training...")
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "classes": train_dataset.classes,
                "class_to_idx": train_dataset.class_to_idx,
                "config": vars(config),
            }, MODEL_DIR / "best_model.pt")
            print(f"✅ New best model saved! Accuracy: {best_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
                break
    
    # Save final model and history
    torch.save({
        "epoch": config.epochs,
        "model_state_dict": model.state_dict(),
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "config": vars(config),
    }, MODEL_DIR / "final_model.pt")
    
    with open(MODEL_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Models saved to: {MODEL_DIR}")


def predict(image_path: str, weights: Optional[str] = None):
    """Run inference on a single image."""
    from PIL import Image
    
    device = get_device()
    weights = weights or str(MODEL_DIR / "best_model.pt")
    
    checkpoint = torch.load(weights, map_location=device)
    classes = checkpoint["classes"]
    config = checkpoint.get("config", {})
    
    input_size = config.get("input_size", 224)
    backbone = config.get("backbone", "resnet18")
    dropout_rate = config.get("dropout_rate", 0.3)
    
    model = build_model(backbone, len(classes), dropout_rate, device, freeze_layers=0)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = probs.max(1)
    
    predicted_class = classes[pred.item()]
    confidence = conf.item()
    
    print(f"Predicted: {predicted_class} ({confidence*100:.1f}%)")
    return predicted_class, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Severity Classification Training")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--backbone", type=str, default="resnet18",
                       choices=["resnet18", "resnet50", "efficientnet_b0"],
                       help="Model backbone")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        batch_size=args.batch,
        epochs=args.epochs,
        learning_rate=args.lr,
        backbone=args.backbone,
        dropout_rate=args.dropout,
        patience=args.patience,
        seed=args.seed,
    )
    
    train(config)
