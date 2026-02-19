#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "severity-model" / "dataset"
MODEL_PATH = ROOT / "severity-model" / "models" / "best_model.pt"

CLASSES = [
    "bumper_dent", "bumper_scratch", "door_dent", "door_scratch",
    "glass_shatter", "head_lamp", "tail_lamp"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224

def build_model(num_classes):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )
    return model

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    val_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_dir = DATA_DIR / "test"
    if not test_dir.exists():
        test_dir = DATA_DIR / "val"
        print(f"No test/ dir found, using val/")

    test_dataset = datasets.ImageFolder(str(test_dir), transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Test samples: {len(test_dataset)}")

    model = build_model(num_classes)
    checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print(f"Loaded model from: {MODEL_PATH}")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\n" + "=" * 80)
    print("PER-PART CLASSIFICATION REPORT")
    print("=" * 80)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    print("=" * 80)
    print("CONFUSION MATRIX (raw counts)")
    print("=" * 80)
    header = "Actual \\ Pred"
    print(f"{header:>16s}", end="")
    for name in class_names:
        print(f"{name[:14]:>15s}", end="")
    print()
    print("-" * (16 + 15 * len(class_names)))
    for i, name in enumerate(class_names):
        print(f"{name[:16]:>16s}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i][j]:>15d}", end="")
        print()

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={"label": "Count"})
    axes[0].set_title("Confusion Matrix (Raw Counts)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted", fontsize=12)
    axes[0].set_ylabel("Actual", fontsize=12)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={"label": "Rate"}, vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted", fontsize=12)
    axes[1].set_ylabel("Actual", fontsize=12)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    out_path = ROOT / "results" / "per_part_confusion_matrix.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved to: {out_path}")
    plt.close()

    print("\n" + "=" * 80)
    print("PER-PART ACCURACY")
    print("=" * 80)
    for i, name in enumerate(class_names):
        acc = cm_normalized[i][i] * 100
        total = cm[i].sum()
        correct = cm[i][i]
        print(f"  {name:<20s}: {acc:6.2f}% ({correct}/{total})")
    overall = np.trace(cm) / cm.sum() * 100
    print(f"\n  {'OVERALL':<20s}: {overall:6.2f}% ({np.trace(cm)}/{cm.sum()})")
    print("=" * 80)


if __name__ == "__main__":
    main()
