#!/usr/bin/env python3
"""
Model Registry - Track and manage model versions
=================================================
Simple file-based registry for model versioning, comparison, and rollback.

Usage:
    python model_registry.py register --name v2.0 --weights path/to/best.pt --dataset multiclass-v2
    python model_registry.py list
    python model_registry.py promote v2.0
    python model_registry.py rollback
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_DIR = ROOT / "models" / "registry"
REGISTRY_FILE = REGISTRY_DIR / "registry.json"
PRODUCTION_LINK = ROOT / "detection-model" / "runs" / "train" / "weights" / "best.pt"


def _load_registry() -> dict:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    return {"current_production": None, "previous_production": None, "models": {}}


def _save_registry(registry: dict) -> None:
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def register_model(
    name: str,
    weights_path: str,
    dataset: str = "",
    metrics: dict = None,
    notes: str = "",
) -> None:
    """Register a new model version."""
    registry = _load_registry()
    weights = Path(weights_path)

    if not weights.exists():
        print(f"ERROR: Weights not found: {weights}")
        return

    # Create version directory
    version_dir = REGISTRY_DIR / name
    version_dir.mkdir(parents=True, exist_ok=True)

    # Copy weights
    shutil.copy2(weights, version_dir / "best.pt")

    # Save metadata
    meta = {
        "name": name,
        "registered": datetime.now().isoformat(),
        "dataset": dataset,
        "weights_size_mb": round(weights.stat().st_size / 1024 / 1024, 2),
        "metrics": metrics or {},
        "notes": notes,
    }
    with open(version_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    registry["models"][name] = meta
    _save_registry(registry)
    print(f"Registered model: {name} at {version_dir}")


def list_models() -> None:
    """List all registered models."""
    registry = _load_registry()
    prod = registry.get("current_production")

    if not registry["models"]:
        print("No models registered.")
        return

    print(f"\n{'Name':<15s} {'Date':<12s} {'Size':>8s} {'Dataset':<20s} {'Status'}")
    print("-" * 70)
    for name, meta in sorted(registry["models"].items()):
        date = meta.get("registered", "")[:10]
        size = f"{meta.get('weights_size_mb', '?')} MB"
        dataset = meta.get("dataset", "")[:20]
        status = " [PRODUCTION]" if name == prod else ""
        print(f"{name:<15s} {date:<12s} {size:>8s} {dataset:<20s}{status}")


def promote_model(name: str) -> None:
    """Promote a model version to production."""
    registry = _load_registry()

    if name not in registry["models"]:
        print(f"ERROR: Model '{name}' not found.")
        return

    version_dir = REGISTRY_DIR / name / "best.pt"
    if not version_dir.exists():
        print(f"ERROR: Weights not found for '{name}'.")
        return

    registry["previous_production"] = registry.get("current_production")
    registry["current_production"] = name
    _save_registry(registry)

    # Copy to production location
    PRODUCTION_LINK.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(version_dir, PRODUCTION_LINK)
    print(f"Promoted '{name}' to production at {PRODUCTION_LINK}")


def rollback() -> None:
    """Rollback to previous production model."""
    registry = _load_registry()
    prev = registry.get("previous_production")

    if not prev:
        print("No previous production model to rollback to.")
        return

    promote_model(prev)
    print(f"Rolled back to: {prev}")


def main():
    parser = argparse.ArgumentParser(description="Model Registry")
    sub = parser.add_subparsers(dest="command")

    reg = sub.add_parser("register", help="Register a model")
    reg.add_argument("--name", required=True)
    reg.add_argument("--weights", required=True)
    reg.add_argument("--dataset", default="")
    reg.add_argument("--notes", default="")

    sub.add_parser("list", help="List models")

    prom = sub.add_parser("promote", help="Promote to production")
    prom.add_argument("name")

    sub.add_parser("rollback", help="Rollback to previous")

    args = parser.parse_args()

    if args.command == "register":
        register_model(args.name, args.weights, args.dataset, notes=args.notes)
    elif args.command == "list":
        list_models()
    elif args.command == "promote":
        promote_model(args.name)
    elif args.command == "rollback":
        rollback()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
