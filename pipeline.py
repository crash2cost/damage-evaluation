#!/usr/bin/env python3
"""
Crash2Cost Pipeline - Full Car Damage Assessment
=================================================
Combines all three models for end-to-end damage assessment:
1. Detection: YOLOv8 detects damage regions
2. Severity: ResNet classifies damage type/severity
3. Cost: RandomForest/GradientBoosting estimates repair cost

Usage:
    python pipeline.py --image path/to/image.jpg
    python pipeline.py --image path/to/image.jpg --car-segment Luxury
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import time

import joblib
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# Paths
ROOT = Path(__file__).resolve().parent

DETECTION_WEIGHTS = ROOT / "detection-model" / "runs" / "train" / "weights" / "best.pt"
SEVERITY_WEIGHTS = ROOT / "severity_model" / "models" / "best_model.pt"
COST_MODEL = ROOT / "cost_model" / "models" / "cost_estimator.pkl"
PART_ENCODER = ROOT / "cost_model" / "models" / "part_encoder.pkl"
SEGMENT_ENCODER = ROOT / "cost_model" / "models" / "segment_encoder.pkl"
CONFIG_FILE = ROOT / "config" / "damage_mappings.json"


@dataclass
class DamageDetection:
    """Single detected damage region."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str


@dataclass
class DamageAssessment:
    """Complete assessment for a detected damage."""
    detection: DamageDetection
    damage_type: str
    damage_confidence: float
    severity: int
    estimated_cost: float
    repair_or_replace: str


@dataclass
class VehicleAssessment:
    """Full vehicle assessment with all damages."""
    damages: List[DamageAssessment]
    total_cost: float
    inference_time_ms: float


# Fallback mappings (used if config file not found)
# These are loaded from config/damage_mappings.json by default
FALLBACK_DAMAGE_TO_PART = {
    "bumper_dent": "Front Bumper",
    "bumper_scratch": "Front Bumper",
    "door_dent": "Front Door",
    "door_scratch": "Front Door",
    "glass_shatter": "Windshield",
    "head_lamp": "Headlight",
    "tail_lamp": "Tail Light",
}

FALLBACK_DAMAGE_TO_SEVERITY = {
    "scratch": 2,
    "dent": 3,
    "shatter": 4,
    "lamp": 3,
}


def get_device():
    """Auto-detect best device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Crash2CostPipeline:
    """End-to-end car damage assessment pipeline."""
    
    def __init__(
        self,
        detection_weights: Optional[str] = None,
        severity_weights: Optional[str] = None,
        cost_model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "auto",
    ):
        self.device = get_device() if device == "auto" else torch.device(device)
        print(f"Using device: {self.device}")

        # Load configuration
        self._load_config(config_path or str(CONFIG_FILE))

        # Load models
        self._load_detection_model(detection_weights or str(DETECTION_WEIGHTS))
        self._load_severity_model(severity_weights or str(SEVERITY_WEIGHTS))
        self._load_cost_model(cost_model_path or str(COST_MODEL))

        # Transforms
        self.severity_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _load_config(self, config_path: str):
        """Load damage mappings from configuration file."""
        self.damage_to_part: Dict[str, str] = {}
        self.damage_to_severity: Dict[str, int] = {}
        self.default_unknown_part = "Front Bumper"
        self.default_unknown_severity = 3
        self.fallback_cost = 1000

        if not Path(config_path).exists():
            print(f"⚠️ Config file not found: {config_path}")
            print("   Using fallback hardcoded mappings")
            self.damage_to_part = FALLBACK_DAMAGE_TO_PART.copy()
            self.damage_to_severity = FALLBACK_DAMAGE_TO_SEVERITY.copy()
            return

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Load main mappings (filter out _comment keys)
            self.damage_to_part = {
                k: v for k, v in config.get("damage_to_part", {}).items()
                if not k.startswith("_")
            }
            self.damage_to_severity = {
                k: v for k, v in config.get("damage_to_severity", {}).items()
                if not k.startswith("_")
            }

            # Load defaults
            defaults = config.get("default_mappings", {})
            self.default_unknown_part = defaults.get("unknown_part", "Front Bumper")
            self.default_unknown_severity = defaults.get("unknown_severity", 3)
            self.fallback_cost = defaults.get("fallback_cost", 1000)

            print(f"✅ Loaded configuration: {config_path}")
            print(f"   Damage types supported: {len(self.damage_to_part)}")
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            print("   Using fallback hardcoded mappings")
            self.damage_to_part = FALLBACK_DAMAGE_TO_PART.copy()
            self.damage_to_severity = FALLBACK_DAMAGE_TO_SEVERITY.copy()

    def _load_detection_model(self, weights_path: str):
        """Load YOLO detection model."""
        try:
            from ultralytics import YOLO
            if Path(weights_path).exists():
                self.detection_model = YOLO(weights_path)
                print(f"✅ Loaded detection model: {weights_path}")
            else:
                print(f"⚠️ Detection weights not found: {weights_path}")
                print("   Using pretrained YOLOv8s as fallback")
                self.detection_model = YOLO("yolov8s.pt")
        except ImportError:
            print("❌ Ultralytics not installed. Run: pip install ultralytics")
            self.detection_model = None
    
    def _load_severity_model(self, weights_path: str):
        """Load severity classification model (supports ResNet and EfficientNet)."""
        self.severity_model = None
        self.severity_classes = []

        if not Path(weights_path).exists():
            print(f"⚠️ Severity weights not found: {weights_path}")
            print("   Run: python severity_model/train.py")
            return

        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.severity_classes = checkpoint["classes"]

            # Handle both old and new checkpoint formats
            config = checkpoint.get("config", checkpoint.get("hyperparameters", {}))

            num_classes = len(self.severity_classes)
            backbone = config.get("backbone", "resnet18")
            dropout_rate = config.get("dropout_rate", 0.3)

            # Create model based on backbone type
            if backbone.startswith("efficientnet"):
                # EfficientNet models
                if backbone == "efficientnet_b0":
                    model = models.efficientnet_b0(weights=None)
                elif backbone == "efficientnet_b1":
                    model = models.efficientnet_b1(weights=None)
                elif backbone == "efficientnet_b2":
                    model = models.efficientnet_b2(weights=None)
                else:
                    model = models.efficientnet_b0(weights=None)

                # EfficientNet uses classifier instead of fc
                in_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(in_features, num_classes),
                )
            elif backbone == "resnet50":
                model = models.resnet50(weights=None)
                model.fc = nn.Sequential(
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(model.fc.in_features, num_classes),
                )
            else:
                # Default to ResNet18
                model = models.resnet18(weights=None)
                model.fc = nn.Sequential(
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(model.fc.in_features, num_classes),
                )

            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()

            self.severity_model = model
            print(f"✅ Loaded severity model: {weights_path}")
            print(f"   Backbone: {backbone}")
            print(f"   Classes: {self.severity_classes}")
        except Exception as e:
            print(f"❌ Failed to load severity model: {e}")
    
    def _load_cost_model(self, model_path: str):
        """Load cost estimation model."""
        self.cost_model = None
        self.part_encoder = None
        self.segment_encoder = None
        
        if not Path(model_path).exists():
            print(f"⚠️ Cost model not found: {model_path}")
            return
        
        try:
            self.cost_model = joblib.load(model_path)
            self.part_encoder = joblib.load(PART_ENCODER)
            self.segment_encoder = joblib.load(SEGMENT_ENCODER)
            print(f"✅ Loaded cost model: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load cost model: {e}")
    
    def detect_damage(self, image: Image.Image, conf_threshold: float = 0.25) -> List[DamageDetection]:
        """Detect damage regions in image."""
        if self.detection_model is None:
            return []
        
        results = self.detection_model.predict(image, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = box.cpu().numpy()
                class_id = int(cls.cpu().item())
                class_name = result.names.get(class_id, "damage")
                
                detections.append(DamageDetection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(conf.cpu().item()),
                    class_id=class_id,
                    class_name=class_name,
                ))
        
        return detections
    
    def classify_damage(self, image: Image.Image, bbox: Tuple[float, float, float, float]) -> Tuple[str, float]:
        """Classify damage type from cropped region."""
        if self.severity_model is None:
            return "unknown", 0.0
        
        # Crop to bounding box
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(image.width, int(x2)), min(image.height, int(y2))
        
        cropped = image.crop((x1, y1, x2, y2))
        input_tensor = self.severity_transform(cropped).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.severity_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = probs.max(1)
        
        damage_type = self.severity_classes[pred.item()]
        confidence = conf.item()
        
        return damage_type, confidence
    
    def estimate_cost(self, damage_type: str, severity: int, car_segment: str = "Family") -> float:
        """Estimate repair cost."""
        if self.cost_model is None:
            # Fallback estimation
            base_costs = {"scratch": 500, "dent": 1500, "shatter": 3000, "lamp": 800}
            for key, cost in base_costs.items():
                if key in damage_type.lower():
                    return cost * severity / 3
            return 1000
        
        # Map damage type to part name
        part_name = self.damage_to_part.get(damage_type, self.default_unknown_part)

        try:
            part_enc = self.part_encoder.transform([part_name])[0]
            seg_enc = self.segment_encoder.transform([car_segment])[0]
            cost = self.cost_model.predict([[part_enc, severity, seg_enc]])[0]
            return float(cost)
        except ValueError:
            return float(self.fallback_cost)
    
    def get_severity(self, damage_type: str) -> int:
        """Get severity level from damage type."""
        for key, severity in self.damage_to_severity.items():
            if key in damage_type.lower():
                return severity
        return self.default_unknown_severity
    
    def assess_damage(
        self,
        image_input,
        car_segment: str = "Family",
        conf_threshold: float = 0.25,
    ) -> VehicleAssessment:
        """Run full damage assessment on an image.
        
        Args:
            image_input: Either a file path (str) or a PIL Image object
            car_segment: Car segment for cost estimation
            conf_threshold: Detection confidence threshold
        """
        start_time = time.time()
        
        # Load image - support both path and PIL Image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be a file path or PIL Image")
        
        # Detect damages
        detections = self.detect_damage(image, conf_threshold)
        if not detections:
            # Retry with progressively lower thresholds to avoid false "no damage" results
            fallback_thresholds = (0.1, 0.05, 0.02, 0.01)
            for threshold in fallback_thresholds:
                if threshold >= conf_threshold:
                    continue
                detections = self.detect_damage(image, threshold)
                if detections:
                    print(f"⚠️ No detections at conf={conf_threshold}. Retrying at conf={threshold} ({len(detections)} found).")
                    break
        
        # Process each detection
        assessments = []
        for detection in detections:
            # Classify damage type
            damage_type, damage_conf = self.classify_damage(image, detection.bbox)
            
            # Get severity
            severity = self.get_severity(damage_type)
            
            # Estimate cost
            cost = self.estimate_cost(damage_type, severity, car_segment)
            
            # Determine repair or replace
            repair_or_replace = "Replace" if severity >= 4 else "Repair"
            
            assessments.append(DamageAssessment(
                detection=detection,
                damage_type=damage_type,
                damage_confidence=damage_conf,
                severity=severity,
                estimated_cost=cost,
                repair_or_replace=repair_or_replace,
            ))
        
        # Calculate totals
        total_cost = sum(a.estimated_cost for a in assessments)
        inference_time = (time.time() - start_time) * 1000
        
        return VehicleAssessment(
            damages=assessments,
            total_cost=total_cost,
            inference_time_ms=inference_time,
        )
    
    def print_assessment(self, assessment: VehicleAssessment):
        """Print formatted assessment results."""
        print(f"\n{'='*60}")
        print("📋 DAMAGE ASSESSMENT REPORT")
        print(f"{'='*60}")
        
        if not assessment.damages:
            print("\n✅ No damage detected!")
            return
        
        for i, damage in enumerate(assessment.damages, 1):
            print(f"\n🔍 Damage #{i}")
            print(f"   Type: {damage.damage_type}")
            print(f"   Confidence: {damage.damage_confidence*100:.1f}%")
            print(f"   Severity: {damage.severity}/5")
            print(f"   Action: {damage.repair_or_replace}")
            print(f"   Estimated Cost: ₪{damage.estimated_cost:,.0f}")
        
        print(f"\n{'='*60}")
        print(f"💰 TOTAL ESTIMATED COST: ₪{assessment.total_cost:,.0f}")
        print(f"⏱️  Inference Time: {assessment.inference_time_ms:.0f}ms")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Crash2Cost Damage Assessment")
    parser.add_argument("--image", required=True, help="Path to damage image")
    parser.add_argument("--car-segment", choices=["Micro", "Family", "Executive", "Luxury", "SUV"],
                       default="Family", help="Car segment")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    
    args = parser.parse_args()
    
    print("🚗 Crash2Cost - Car Damage Assessment System")
    print("=" * 60)
    print("\nLoading models...")
    
    pipeline = Crash2CostPipeline(device=args.device)
    
    print(f"\n📸 Analyzing: {args.image}")
    assessment = pipeline.assess_damage(
        args.image,
        car_segment=args.car_segment,
        conf_threshold=args.conf,
    )
    
    pipeline.print_assessment(assessment)
    
    print("\n⚠️ NOTE: This is an automated estimate. Actual costs may vary.")
    print("   Consult a professional for accurate assessment.")


if __name__ == "__main__":
    main()
