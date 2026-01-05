from pathlib import Path
import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data_yaml = root / "dataset" / "data.yaml"

    model = YOLO(args.model)
    
    print("\n" + "="*70)
    print("YOLO Training - Car Damage Detection")
    print("="*70)
    print(f"Training Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image Size: {args.imgsz}x{args.imgsz}")
    print(f"  Batch Size: {args.batch}")
    print(f"  Learning Rate: {args.lr0}")
    print(f"  Patience: {args.patience}")
    print("="*70 + "\n")
    
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        lr0=args.lr0,
        project=str(root / "runs"),
        name="damage-detector",
        verbose=True,
        plots=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.2,
        scale=0.9,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.5,
        copy_paste=0.3,
    )


if __name__ == "__main__":
    main()