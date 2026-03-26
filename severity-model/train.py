#!/usr/bin/env python3

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "severity-model" / "dataset"
MODEL_DIR = ROOT / "severity-model" / "models"
RUNS_DIR = ROOT / "severity-model" / "runs"

MODEL_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)

DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 80
DEFAULT_LEARNING_RATE = 0.000133
DEFAULT_WEIGHT_DECAY = 0.0001
DEFAULT_DROPOUT_RATE = 0.45
DEFAULT_LABEL_SMOOTHING = 0.1
DEFAULT_BACKBONE = "resnet50"
DEFAULT_INPUT_SIZE = 224
DEFAULT_SEED = 42
DEFAULT_PATIENCE = 15
DEFAULT_FREEZE_LAYERS = 2

SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5

CROP_SCALE_MIN = 0.6
CROP_SCALE_MAX = 1.0
HORIZONTAL_FLIP_PROB = 0.5
VERTICAL_FLIP_PROB = 0.1
COLOR_BRIGHTNESS = 0.4
COLOR_CONTRAST = 0.4
COLOR_SATURATION = 0.3
COLOR_HUE = 0.15
ROTATION_DEGREES = 30
AFFINE_TRANSLATE = 0.15
AFFINE_SHEAR = 15
PERSPECTIVE_DISTORTION = 0.2
PERSPECTIVE_PROB = 0.5
GAUSSIAN_BLUR_KERNEL = 3
GAUSSIAN_BLUR_SIGMA_MIN = 0.1
GAUSSIAN_BLUR_SIGMA_MAX = 2.0
GAUSSIAN_BLUR_PROB = 0.3
RANDOM_ERASING_PROB = 0.3
RANDOM_ERASING_SCALE_MIN = 0.02
RANDOM_ERASING_SCALE_MAX = 0.2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SEPARATOR_WIDTH = 60
PERCENTAGE_MULTIPLIER = 100.0


@dataclass
class TrainingConfig:
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    dropout_rate: float = DEFAULT_DROPOUT_RATE
    label_smoothing: float = DEFAULT_LABEL_SMOOTHING
    backbone: str = DEFAULT_BACKBONE
    input_size: int = DEFAULT_INPUT_SIZE
    seed: int = DEFAULT_SEED
    patience: int = DEFAULT_PATIENCE
    freeze_layers: int = DEFAULT_FREEZE_LAYERS


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_transforms(input_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(CROP_SCALE_MIN, CROP_SCALE_MAX)),
        transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_PROB),
        transforms.RandomVerticalFlip(p=VERTICAL_FLIP_PROB),
        transforms.ColorJitter(
            brightness=COLOR_BRIGHTNESS,
            contrast=COLOR_CONTRAST,
            saturation=COLOR_SATURATION,
            hue=COLOR_HUE
        ),
        transforms.RandomRotation(ROTATION_DEGREES),
        transforms.RandomAffine(degrees=0, translate=(AFFINE_TRANSLATE, AFFINE_TRANSLATE), shear=AFFINE_SHEAR),
        transforms.RandomPerspective(distortion_scale=PERSPECTIVE_DISTORTION, p=PERSPECTIVE_PROB),
        transforms.RandomApply([
            transforms.GaussianBlur(
                kernel_size=GAUSSIAN_BLUR_KERNEL,
                sigma=(GAUSSIAN_BLUR_SIGMA_MIN, GAUSSIAN_BLUR_SIGMA_MAX)
            )
        ], p=GAUSSIAN_BLUR_PROB),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=RANDOM_ERASING_PROB, scale=(RANDOM_ERASING_SCALE_MIN, RANDOM_ERASING_SCALE_MAX)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, val_transform


def build_model(
    backbone: str,
    num_classes: int,
    dropout_rate: float,
    device: torch.device,
    freeze_layers: int = DEFAULT_FREEZE_LAYERS
) -> nn.Module:
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

    if freeze_layers > 0:
        layers_to_freeze = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4][:freeze_layers]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        print(f"   Frozen first {freeze_layers} backbone layers")

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes),
    )

    return model.to(device)


def get_weighted_sampler(dataset: datasets.ImageFolder) -> Tuple[WeightedRandomSampler, List[float]]:
    class_counts: Dict[str, int] = {}
    for label in dataset.targets:
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    total_samples = len(dataset)
    class_weights: List[float] = []
    for class_name in dataset.classes:
        count = class_counts.get(class_name, 1)
        class_weights.append(total_samples / count)

    sample_weights = [class_weights[label] for label in dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    return sampler, class_weights


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(
            loss=f"{running_loss / (pbar.n + 1):.4f}",
            acc=f"{PERCENTAGE_MULTIPLIER * correct / total:.2f}%"
        )

    return running_loss / len(loader), PERCENTAGE_MULTIPLIER * correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
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

    return running_loss / len(loader), PERCENTAGE_MULTIPLIER * correct / total


def train(config: TrainingConfig) -> None:
    set_seed(config.seed)
    device = get_device()

    print(f"\n{'=' * SEPARATOR_WIDTH}")
    print("Severity Classification Model Training")
    print(f"{'=' * SEPARATOR_WIDTH}")
    print(f"Device: {device}")
    print(f"Backbone: {config.backbone}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Dropout Rate: {config.dropout_rate}")
    print(f"{'=' * SEPARATOR_WIDTH}\n")

    train_transform, val_transform = get_transforms(config.input_size)

    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=val_transform)

    num_classes = len(train_dataset.classes)
    print(f"Classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    sampler, class_weights = get_weighted_sampler(train_dataset)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    print(f"\nBuilding {config.backbone} model with pretrained ImageNet weights...")
    model = build_model(config.backbone, num_classes, config.dropout_rate, device, config.freeze_layers)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print(f"\n{'=' * SEPARATOR_WIDTH}")
    print(f"{config.backbone.upper()} Full Architecture")
    print(f"{'=' * SEPARATOR_WIDTH}")
    print(model)
    print(f"\nLayer breakdown:")
    layer_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            layer_count += 1
            params = sum(p.numel() for p in module.parameters())
            frozen = not any(p.requires_grad for p in module.parameters())
            status = " [FROZEN]" if frozen else ""
            print(f"  {layer_count:3d}. {name:40s} {module.__class__.__name__:20s} params={params:>10,}{status}")
    print(f"\n  Total neural layers (Conv2d + Linear + BatchNorm2d): {layer_count}")
    print(f"{'=' * SEPARATOR_WIDTH}")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=config.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR
    )

    best_acc = 0.0
    patience_counter = 0
    history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("\nStarting training...")
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")

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
            print(f"New best model saved! Accuracy: {best_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

    torch.save({
        "epoch": config.epochs,
        "model_state_dict": model.state_dict(),
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "config": vars(config),
    }, MODEL_DIR / "final_model.pt")

    with open(MODEL_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Models saved to: {MODEL_DIR}")


def predict(image_path: str, weights: Optional[str] = None) -> Tuple[str, float]:
    device = get_device()
    weights = weights or str(MODEL_DIR / "best_model.pt")

    checkpoint = torch.load(weights, map_location=device)
    classes = checkpoint["classes"]
    config = checkpoint.get("config", {})

    input_size = config.get("input_size", DEFAULT_INPUT_SIZE)
    backbone = config.get("backbone", DEFAULT_BACKBONE)
    dropout_rate = config.get("dropout_rate", DEFAULT_DROPOUT_RATE)

    model = build_model(backbone, len(classes), dropout_rate, device, freeze_layers=0)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = probs.max(1)

    predicted_class = classes[pred.item()]
    confidence = conf.item()

    print(f"Predicted: {predicted_class} ({confidence * PERCENTAGE_MULTIPLIER:.1f}%)")
    return predicted_class, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Severity Classification Training")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE,
                        choices=["resnet18", "resnet50", "efficientnet_b0"],
                        help="Model backbone")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT_RATE, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")

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
