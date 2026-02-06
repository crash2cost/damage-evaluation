from pathlib import Path
import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained model weights (.pt file)")
    parser.add_argument("--source", required=True, help="Path to image or folder of images")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save", action="store_true", help="Save results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Load the trained model
    model = YOLO(args.model)
    
    # Run prediction
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=args.save,
        show=False
    )
    
    # Print results
    for result in results:
        boxes = result.boxes
        print(f"\nImage: {result.path}")
        print(f"Detections: {len(boxes)}")
        for box in boxes:
            conf = box.conf[0]
            cls = box.cls[0]
            print(f"  - Class: {model.names[int(cls)]}, Confidence: {conf:.2f}")


if __name__ == "__main__":
    main()
