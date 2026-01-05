import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import joblib
import argparse
from pathlib import Path
import json
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "detection-model" / "src"))
from custom_inference import load_custom_yolo
DETECTION_WEIGHTS = ROOT / "detection-model" / "runs" / "custom-yolo" / "best.pt"
CLASSIFIER_PATH = ROOT / "regression-model" / "models" / "damage_classifier_best.pt"
COST_MODEL_PATH = ROOT / "severity_model" / "models" / "cost_estimator.pkl"
PART_ENCODER_PATH = ROOT / "severity_model" / "models" / "part_encoder.pkl"
SEGMENT_ENCODER_PATH = ROOT / "severity_model" / "models" / "segment_encoder.pkl"
METADATA_PATH = ROOT / "severity_model" / "models" / "metadata.json"
DAMAGE_TO_PART = {
    'bumper_dent': 'Front Bumper',
    'bumper_scratch': 'Front Bumper',
    'door_dent': 'Front Door',
    'door_scratch': 'Front Door',
    'glass_shatter': 'Windshield',
    'head_lamp': 'Headlight',
    'tail_lamp': 'Tail Light'
}
class DamageDetector:
    def __init__(self, weights_path, conf_threshold=0.25):
        self.model = load_custom_yolo(weights_path, num_classes=1)
        self.conf_threshold = conf_threshold
    def detect(self, image_path):
        results = self.model.predict(image_path, conf=self.conf_threshold)
        return results
    def crop_detections(self, image_path, results):
        img = Image.open(image_path).convert('RGB')
        crops = []
        if len(results.boxes) > 0:
            for box in results.boxes.xyxy:
                x1, y1, x2, y2 = box.int().tolist()
                pad = 10
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(img.width, x2 + pad)
                y2 = min(img.height, y2 + pad)
                crop = img.crop((x1, y1, x2, y2))
                crops.append(crop)
        else:
            crops.append(img)
        return crops
    def visualize(self, image_path, results, save_path=None):
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        if len(results.boxes) > 0:
            for i, (box, conf) in enumerate(zip(results.boxes.xyxy, results.boxes.conf)):
                x1, y1, x2, y2 = box.int().tolist()
                draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                label = f"Damage {conf:.2f}"
                draw.text((x1, y1-20), label, fill='red')
        if save_path:
            img.save(save_path)
            print(f"   Saved detection visualization to {save_path}")
        return img
class DamageClassifier:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        checkpoint = torch.load(model_path, map_location=device)
        self.classes = checkpoint['classes']
        num_classes = len(self.classes)
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def predict(self, image):
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        damage_type = self.classes[predicted_idx.item()]
        return damage_type, confidence.item()
class CostEstimator:
    def __init__(self, model_path, part_encoder_path, segment_encoder_path):
        self.model = joblib.load(model_path)
        self.part_encoder = joblib.load(part_encoder_path)
        self.segment_encoder = joblib.load(segment_encoder_path)
    def estimate(self, part_name, severity, car_segment):
        try:
            part_enc = self.part_encoder.transform([part_name])[0]
            seg_enc = self.segment_encoder.transform([car_segment])[0]
            cost = self.model.predict([[part_enc, severity, seg_enc]])[0]
            return int(cost)
        except ValueError as e:
            return None
def main():
    parser = argparse.ArgumentParser(description='Crash damage cost estimation with custom YOLO')
    parser.add_argument('--image', required=True, help='Path to damage image')
    parser.add_argument('--severity', type=int, choices=[1,2,3,4,5], default=3,
                       help='Damage severity (1=minor, 5=severe)')
    parser.add_argument('--car-segment', choices=['Micro', 'Family', 'Executive', 'Luxury', 'SUV'],
                       default='Family', help='Car segment')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'],
                       help='Device to run inference on')
    parser.add_argument('--conf', type=float, default=0.25, help='Detection confidence threshold')
    parser.add_argument('--save-viz', action='store_true', help='Save visualization')
    args = parser.parse_args()
    print("🚗 Crash2Cost - Complete Damage Assessment System (Custom YOLO)")
    print("=" * 70)
    print("Loading models...")
    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    print("  [1/3] Loading custom YOLO detector...")
    detector = DamageDetector(DETECTION_WEIGHTS, conf_threshold=args.conf)
    print("  [2/3] Loading damage classifier...")
    classifier = DamageClassifier(CLASSIFIER_PATH, device=device)
    print("  [3/3] Loading cost estimator...")
    cost_estimator = CostEstimator(COST_MODEL_PATH, PART_ENCODER_PATH, SEGMENT_ENCODER_PATH)
    print("\n" + "=" * 70)
    print(f"\n📸 STEP 1: Detecting damage regions in {Path(args.image).name}")
    detection_results = detector.detect(args.image)
    num_detections = len(detection_results.boxes)
    print(f"   Found {num_detections} damage region(s)")
    if num_detections > 0:
        for i, (box, conf) in enumerate(zip(detection_results.boxes.xyxy, detection_results.boxes.conf)):
            x1, y1, x2, y2 = box.int().tolist()
            print(f"   - Region {i+1}: [{x1}, {y1}, {x2}, {y2}] (confidence: {conf:.2%})")
    if args.save_viz:
        viz_path = Path(args.image).parent / f"{Path(args.image).stem}_detection.jpg"
        detector.visualize(args.image, detection_results, save_path=viz_path)
    print(f"\n🔍 STEP 2: Classifying damage type")
    crops = detector.crop_detections(args.image, detection_results)
    damage_type, confidence = classifier.predict(crops[0])
    print(f"   Detected damage: {damage_type}")
    print(f"   Classification confidence: {confidence*100:.1f}%")
    part_name = DAMAGE_TO_PART.get(damage_type, 'Front Bumper')
    print(f"   Affected part: {part_name}")
    print(f"\n💰 STEP 3: Estimating repair cost")
    print(f"   Severity level: {args.severity}/5")
    print(f"   Car segment: {args.car_segment}")
    cost = cost_estimator.estimate(part_name, args.severity, args.car_segment)
    print("\n" + "=" * 70)
    print("📋 FINAL ASSESSMENT")
    print("=" * 70)
    if cost:
        print(f"\n✅ Estimated repair cost: ₪{cost:,}")
        print(f"   (Approximately ${int(cost/3.7):,} USD)")
    else:
        print("\n❌ Could not estimate cost for this configuration")
    print(f"\n📊 Summary:")
    print(f"   • Damage regions detected: {num_detections}")
    print(f"   • Damage type: {damage_type}")
    print(f"   • Affected part: {part_name}")
    print(f"   • Severity: {args.severity}/5")
    print(f"   • Car segment: {args.car_segment}")
    print("\n" + "=" * 70)
    print("\n⚠️  NOTE: This is an automated estimate. Actual costs may vary.")
    print("   Consult a professional for accurate assessment.")
    print("\n✨ System powered by:")
    print("   - Custom YOLO (Built from scratch with PyTorch)")
    print("   - ResNet18 Damage Classifier")
    print("   - XGBoost Cost Estimator")
if __name__ == "__main__":
    main()