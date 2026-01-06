import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import joblib
import argparse
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parent
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
    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.predict_image(image)

    def predict_image(self, image):
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
    parser = argparse.ArgumentParser(description='Crash damage cost estimation')
    parser.add_argument('--image', required=True, help='Path to damage image')
    parser.add_argument('--severity', type=int, choices=[1,2,3,4,5], default=3,
                       help='Damage severity (1=minor, 5=severe)')
    parser.add_argument('--car-segment', choices=['Micro', 'Family', 'Executive', 'Luxury', 'SUV'],
                       default='Family', help='Car segment')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'],
                       help='Device to run inference on')
    args = parser.parse_args()
    print("🚗 Crash2Cost - Damage Assessment System")
    print("=" * 50)
    print("Loading models...")
    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    classifier = DamageClassifier(CLASSIFIER_PATH, device=device)
    cost_estimator = CostEstimator(COST_MODEL_PATH, PART_ENCODER_PATH, SEGMENT_ENCODER_PATH)
    print(f"\nAnalyzing image: {args.image}")
    damage_type, confidence = classifier.predict(args.image)
    print(f"✅ Detected damage: {damage_type} (confidence: {confidence*100:.1f}%)")
    part_name = DAMAGE_TO_PART.get(damage_type, 'Front Bumper')
    print(f"   Affected part: {part_name}")
    print(f"\nCost estimation:")
    print(f"   Severity level: {args.severity}/5")
    print(f"   Car segment: {args.car_segment}")
    cost = cost_estimator.estimate(part_name, args.severity, args.car_segment)
    if cost:
        print(f"\n💰 Estimated repair cost: ₪{cost:,}")
        print(f"   (Approximately ${int(cost/3.7):,} USD)")
    else:
        print("\n❌ Could not estimate cost for this configuration")
    print("\n" + "=" * 50)
    print("\nNOTE: This is an automated estimate based on:")
    print("  - AI-detected damage type")
    print("  - User-specified severity level")
    print("  - Car segment pricing")
    print("  Actual repair costs may vary. Consult a professional.")
if __name__ == "__main__":
    main()
