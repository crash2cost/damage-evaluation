#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

from train import TrainingConfig, train, set_seed, get_device, get_transforms, build_model
from train import DATA_DIR, MODEL_DIR, RUNS_DIR
from train import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE
from train import DEFAULT_WEIGHT_DECAY, DEFAULT_DROPOUT_RATE, DEFAULT_PATIENCE
from train import DEFAULT_FREEZE_LAYERS, DEFAULT_LABEL_SMOOTHING, DEFAULT_SEED
from train import DEFAULT_INPUT_SIZE, SEPARATOR_WIDTH

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

BACKBONES = ["resnet18", "resnet50", "efficientnet_b0"]
COMPARISON_DIR = RUNS_DIR / "backbone_comparison"
COMPARISON_DIR.mkdir(exist_ok=True)


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def evaluate_test(model, test_loader, device, classes):
    model.eval()
    correct = 0
    total = 0
    class_correct = {c: 0 for c in classes}
    class_total = {c: 0 for c in classes}

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for i in range(labels.size(0)):
                label = classes[labels[i].item()]
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1

    overall_acc = 100.0 * correct / total
    per_class = {}
    for c in classes:
        if class_total[c] > 0:
            per_class[c] = 100.0 * class_correct[c] / class_total[c]
        else:
            per_class[c] = 0.0

    return overall_acc, per_class


def run_comparison(epochs: int, seed: int):
    results = {}

    for backbone in BACKBONES:
        print(f"\n{'=' * SEPARATOR_WIDTH}")
        print(f"  Training: {backbone.upper()}")
        print(f"{'=' * SEPARATOR_WIDTH}\n")

        set_seed(seed)

        config = TrainingConfig(
            batch_size=DEFAULT_BATCH_SIZE,
            epochs=epochs,
            learning_rate=DEFAULT_LEARNING_RATE,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            dropout_rate=DEFAULT_DROPOUT_RATE,
            label_smoothing=DEFAULT_LABEL_SMOOTHING,
            backbone=backbone,
            input_size=DEFAULT_INPUT_SIZE,
            seed=seed,
            patience=DEFAULT_PATIENCE,
            freeze_layers=DEFAULT_FREEZE_LAYERS,
        )

        device = get_device()
        _, val_transform = get_transforms(config.input_size)
        train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=val_transform)
        num_classes = len(train_dataset.classes)
        model = build_model(backbone, num_classes, config.dropout_rate, device, config.freeze_layers)
        total_params, trainable_params = count_parameters(model)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        start_time = time.time()
        train(config)
        train_time = time.time() - start_time

        history_path = MODEL_DIR / "training_history.json"
        with open(history_path, "r") as f:
            history = json.load(f)

        best_val_acc = max(history["val_acc"])
        best_epoch = history["val_acc"].index(best_val_acc) + 1
        final_train_acc = history["train_acc"][-1]

        checkpoint = torch.load(MODEL_DIR / "best_model.pt", map_location=device)
        model = build_model(backbone, num_classes, config.dropout_rate, device, freeze_layers=0)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        test_acc, per_class_acc = evaluate_test(model, test_loader, device, train_dataset.classes)

        backbone_result = {
            "backbone": backbone,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "best_val_acc": round(best_val_acc, 2),
            "test_acc": round(test_acc, 2),
            "final_train_acc": round(final_train_acc, 2),
            "best_epoch": best_epoch,
            "total_epochs": len(history["train_acc"]),
            "train_time_seconds": round(train_time, 1),
            "overfitting_gap": round(final_train_acc - best_val_acc, 2),
            "per_class_test_acc": {k: round(v, 2) for k, v in per_class_acc.items()},
            "history": history,
        }
        results[backbone] = backbone_result

        import shutil
        backbone_dir = COMPARISON_DIR / backbone
        backbone_dir.mkdir(exist_ok=True)
        shutil.copy2(MODEL_DIR / "best_model.pt", backbone_dir / "best_model.pt")
        with open(backbone_dir / "results.json", "w") as f:
            json.dump(backbone_result, f, indent=2)

        print(f"\n{backbone}: Val={best_val_acc:.2f}%, Test={test_acc:.2f}%, "
              f"Params={total_params:,}, Time={train_time:.0f}s")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n{'=' * 80}")
    print("BACKBONE COMPARISON RESULTS")
    print(f"{'=' * 80}")
    print(f"{'Backbone':<18} {'Params':>10} {'Trainable':>10} {'Val Acc':>9} {'Test Acc':>9} {'Overfit':>8} {'Time':>8}")
    print(f"{'-'*18} {'-'*10} {'-'*10} {'-'*9} {'-'*9} {'-'*8} {'-'*8}")

    for bb in BACKBONES:
        r = results[bb]
        print(f"{bb:<18} {r['total_params']:>10,} {r['trainable_params']:>10,} "
              f"{r['best_val_acc']:>8.2f}% {r['test_acc']:>8.2f}% "
              f"{r['overfitting_gap']:>7.2f}% {r['train_time_seconds']:>6.0f}s")

    summary = {bb: {k: v for k, v in r.items() if k != "history"} for bb, r in results.items()}
    with open(COMPARISON_DIR / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {COMPARISON_DIR}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare backbone architectures")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs per backbone")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    args = parser.parse_args()

    run_comparison(args.epochs, args.seed)
