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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# =============================================================================
# Constants
# =============================================================================

# Paths
ROOT = Path(__file__).resolve().parent

DETECTION_WEIGHTS = ROOT / "detection-model" / "runs" / "train" / "weights" / "best.pt"
SEVERITY_WEIGHTS = ROOT / "severity_model" / "models" / "best_model.pt"
COST_MODEL = ROOT / "cost_model" / "models" / "cost_estimator.pkl"
PART_ENCODER = ROOT / "cost_model" / "models" / "part_encoder.pkl"
DAMAGE_ENCODER = ROOT / "cost_model" / "models" / "damage_encoder.pkl"
CONFIG_FILE = ROOT / "config" / "damage_mappings.json"

# Image processing
SEVERITY_INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Detection thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
FALLBACK_CONFIDENCE_THRESHOLDS = (0.1, 0.05, 0.02, 0.01)

# Severity thresholds
REPLACE_SEVERITY_THRESHOLD = 4
DEFAULT_UNKNOWN_SEVERITY = 3
MAX_SEVERITY = 5

# Cost defaults
DEFAULT_FALLBACK_COST = 1000
DEFAULT_UNKNOWN_PART = "Front Bumper"
FALLBACK_BASE_COSTS = {"scratch": 500, "dent": 1500, "shatter": 3000, "lamp": 800}
SEVERITY_DIVISOR = 3

# Model defaults
DEFAULT_BACKBONE = "resnet18"
DEFAULT_DROPOUT_RATE = 0.3

# Display
SEPARATOR_WIDTH = 60
MS_PER_SECOND = 1000


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
FALLBACK_DAMAGE_TO_PART: Dict[str, str] = {
    "bumper_dent": "Front Bumper",
    "bumper_scratch": "Front Bumper",
    "door_dent": "Front Door",
    "door_scratch": "Front Door",
    "glass_shatter": "Windshield",
    "head_lamp": "Headlight",
    "tail_lamp": "Tail Light",
}

FALLBACK_DAMAGE_TO_SEVERITY: Dict[str, int] = {
    "scratch": 2,
    "dent": 3,
    "shatter": 4,
    "lamp": 3,
}


def get_device() -> torch.device:
    """
    Auto-detect best available device for inference.

    Returns:
        torch.device: Best available device (mps, cuda, or cpu).
    """
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
    ) -> None:
        """
        Initialize the Crash2Cost pipeline.

        Args:
            detection_weights: Path to YOLO detection model weights.
            severity_weights: Path to severity classification model weights.
            cost_model_path: Path to cost estimation model.
            config_path: Path to damage mappings configuration file.
            device: Device to use ('auto', 'cpu', 'cuda', 'mps').
        """
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
            transforms.Resize((SEVERITY_INPUT_SIZE, SEVERITY_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def _load_config(self, config_path: str) -> None:
        """
        Load damage mappings from configuration file.

        Args:
            config_path: Path to the JSON configuration file.
        """
        self.damage_to_part: Dict[str, str] = {}
        self.damage_to_severity: Dict[str, int] = {}
        self.default_unknown_part = DEFAULT_UNKNOWN_PART
        self.default_unknown_severity = DEFAULT_UNKNOWN_SEVERITY
        self.fallback_cost = DEFAULT_FALLBACK_COST

        if not Path(config_path).exists():
            print(f"Config file not found: {config_path}")
            print("   Using fallback hardcoded mappings")
            self.damage_to_part = FALLBACK_DAMAGE_TO_PART.copy()
            self.damage_to_severity = FALLBACK_DAMAGE_TO_SEVERITY.copy()
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
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
            self.default_unknown_part = defaults.get("unknown_part", DEFAULT_UNKNOWN_PART)
            self.default_unknown_severity = defaults.get("unknown_severity", DEFAULT_UNKNOWN_SEVERITY)
            self.fallback_cost = defaults.get("fallback_cost", DEFAULT_FALLBACK_COST)

            print(f"Loaded configuration: {config_path}")
            print(f"   Damage types supported: {len(self.damage_to_part)}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to load config: {e}")
            print("   Using fallback hardcoded mappings")
            self.damage_to_part = FALLBACK_DAMAGE_TO_PART.copy()
            self.damage_to_severity = FALLBACK_DAMAGE_TO_SEVERITY.copy()

    def _load_detection_model(self, weights_path: str) -> None:
        """
        Load YOLO detection model.

        Args:
            weights_path: Path to the YOLO model weights file.
        """
        try:
            from ultralytics import YOLO
            if Path(weights_path).exists():
                self.detection_model = YOLO(weights_path)
                print(f"Loaded detection model: {weights_path}")
            else:
                print(f"Detection weights not found: {weights_path}")
                print("   Using pretrained YOLOv8s as fallback")
                pretrained_path = Path(__file__).parent / "detection-model" / "pretrained" / "yolov8s.pt"
                pretrained_path.parent.mkdir(parents=True, exist_ok=True)
                self.detection_model = YOLO(str(pretrained_path) if pretrained_path.exists() else "yolov8s.pt")
        except ImportError:
            print("Ultralytics not installed. Run: pip install ultralytics")
            self.detection_model = None

    def _load_severity_model(self, weights_path: str) -> None:
        """
        Load severity classification model (supports ResNet and EfficientNet).

        Args:
            weights_path: Path to the severity model weights file.
        """
        self.severity_model: Optional[nn.Module] = None
        self.severity_classes: List[str] = []

        if not Path(weights_path).exists():
            print(f"Severity weights not found: {weights_path}")
            print("   Run: python severity_model/train.py")
            return

        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.severity_classes = checkpoint["classes"]

            # Handle both old and new checkpoint formats
            config = checkpoint.get("config", checkpoint.get("hyperparameters", {}))

            num_classes = len(self.severity_classes)
            backbone = config.get("backbone", DEFAULT_BACKBONE)
            dropout_rate = config.get("dropout_rate", DEFAULT_DROPOUT_RATE)

            # Create model based on backbone type
            if backbone.startswith("efficientnet"):
                model = self._create_efficientnet_model(backbone, num_classes, dropout_rate)
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
            print(f"Loaded severity model: {weights_path}")
            print(f"   Backbone: {backbone}")
            print(f"   Classes: {self.severity_classes}")
        except (KeyError, RuntimeError) as e:
            print(f"Failed to load severity model: {e}")

    def _create_efficientnet_model(
        self, backbone: str, num_classes: int, dropout_rate: float
    ) -> nn.Module:
        """
        Create an EfficientNet model with custom classifier head.

        Args:
            backbone: EfficientNet variant name.
            num_classes: Number of output classes.
            dropout_rate: Dropout probability for the classifier.

        Returns:
            Configured EfficientNet model.
        """
        if backbone == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
        elif backbone == "efficientnet_b1":
            model = models.efficientnet_b1(weights=None)
        elif backbone == "efficientnet_b2":
            model = models.efficientnet_b2(weights=None)
        else:
            model = models.efficientnet_b0(weights=None)

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        return model

    def _load_cost_model(self, model_path: str) -> None:
        """
        Load cost estimation model (v2 with damage_type).

        Args:
            model_path: Path to the cost estimation model file.
        """
        self.cost_model = None
        self.part_encoder = None
        self.damage_encoder = None

        if not Path(model_path).exists():
            print(f"Cost model not found: {model_path}")
            return

        try:
            self.cost_model = joblib.load(model_path)
            self.part_encoder = joblib.load(PART_ENCODER)
            self.damage_encoder = joblib.load(DAMAGE_ENCODER)
            print(f"Loaded cost model: {model_path}")
            print("   Features: Part_Name, Damage_Type, Severity")
        except (FileNotFoundError, ValueError) as e:
            print(f"Failed to load cost model: {e}")

    def detect_damage(
        self, image: Image.Image, conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ) -> List[DamageDetection]:
        """
        Detect damage regions in image.

        Args:
            image: PIL Image to analyze.
            conf_threshold: Minimum confidence for detections.

        Returns:
            List of detected damage regions.
        """
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

    def classify_damage(
        self, image: Image.Image, bbox: Tuple[float, float, float, float]
    ) -> Tuple[str, float]:
        """
        Classify damage type from cropped region.

        Args:
            image: Full PIL Image.
            bbox: Bounding box coordinates (x1, y1, x2, y2).

        Returns:
            Tuple of (damage_type, confidence).
        """
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

    def estimate_cost(
        self, damage_type: str, severity: int, car_segment: str = "Family"
    ) -> float:
        """
        Estimate repair cost using Part + Damage_Type + Severity.

        Args:
            damage_type: Type of damage detected.
            severity: Severity level (1-5).
            car_segment: Vehicle segment category.

        Returns:
            Estimated repair cost in local currency.
        """
        if self.cost_model is None or self.damage_encoder is None:
            # Fallback estimation
            for key, cost in FALLBACK_BASE_COSTS.items():
                if key in damage_type.lower():
                    return cost * severity / SEVERITY_DIVISOR
            return float(DEFAULT_FALLBACK_COST)

        # Map damage type to part name
        part_name = self.damage_to_part.get(damage_type, self.default_unknown_part)

        try:
            part_enc = self.part_encoder.transform([part_name])[0]
            damage_enc = self.damage_encoder.transform([damage_type])[0]
            # v2 model: [Part_Encoded, Damage_Encoded, Severity]
            cost = self.cost_model.predict([[part_enc, damage_enc, severity]])[0]
            return float(cost)
        except ValueError:
            # If damage_type not in encoder, use fallback
            return float(self.fallback_cost)

    def get_severity(self, damage_type: str) -> int:
        """
        Get severity level from damage type.

        Args:
            damage_type: Type of damage detected.

        Returns:
            Severity level (1-5).
        """
        for key, severity in self.damage_to_severity.items():
            if key in damage_type.lower():
                return severity
        return self.default_unknown_severity

    def assess_damage(
        self,
        image_input: Union[str, Image.Image],
        car_segment: str = "Family",
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> VehicleAssessment:
        """
        Run full damage assessment on an image.

        Args:
            image_input: Either a file path (str) or a PIL Image object.
            car_segment: Car segment for cost estimation.
            conf_threshold: Detection confidence threshold.

        Returns:
            Complete vehicle assessment with all damages and costs.

        Raises:
            ValueError: If image_input is neither a string path nor PIL Image.
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
            for threshold in FALLBACK_CONFIDENCE_THRESHOLDS:
                if threshold >= conf_threshold:
                    continue
                detections = self.detect_damage(image, threshold)
                if detections:
                    print(f"No detections at conf={conf_threshold}. "
                          f"Retrying at conf={threshold} ({len(detections)} found).")
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
            repair_or_replace = "Replace" if severity >= REPLACE_SEVERITY_THRESHOLD else "Repair"

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
        inference_time = (time.time() - start_time) * MS_PER_SECOND

        return VehicleAssessment(
            damages=assessments,
            total_cost=total_cost,
            inference_time_ms=inference_time,
        )

    def print_assessment(self, assessment: VehicleAssessment) -> None:
        """
        Print formatted assessment results.

        Args:
            assessment: Vehicle assessment to display.
        """
        print(f"\n{'=' * SEPARATOR_WIDTH}")
        print("DAMAGE ASSESSMENT REPORT")
        print(f"{'=' * SEPARATOR_WIDTH}")

        if not assessment.damages:
            print("\nNo damage detected!")
            return

        for i, damage in enumerate(assessment.damages, 1):
            print(f"\nDamage #{i}")
            print(f"   Type: {damage.damage_type}")
            print(f"   Confidence: {damage.damage_confidence * 100:.1f}%")
            print(f"   Severity: {damage.severity}/{MAX_SEVERITY}")
            print(f"   Action: {damage.repair_or_replace}")
            print(f"   Estimated Cost: {damage.estimated_cost:,.0f} ILS")

        print(f"\n{'=' * SEPARATOR_WIDTH}")
        print(f"TOTAL ESTIMATED COST: {assessment.total_cost:,.0f} ILS")
        print(f"Inference Time: {assessment.inference_time_ms:.0f}ms")
        print(f"{'=' * SEPARATOR_WIDTH}")


def main() -> None:
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description="Crash2Cost Damage Assessment")
    parser.add_argument("--image", required=True, help="Path to damage image")
    parser.add_argument("--car-segment", choices=["Micro", "Family", "Executive", "Luxury", "SUV"],
                        default="Family", help="Car segment")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help="Detection confidence threshold")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])

    args = parser.parse_args()

    print("Crash2Cost - Car Damage Assessment System")
    print("=" * SEPARATOR_WIDTH)
    print("\nLoading models...")

    pipeline = Crash2CostPipeline(device=args.device)

    print(f"\nAnalyzing: {args.image}")
    assessment = pipeline.assess_damage(
        args.image,
        car_segment=args.car_segment,
        conf_threshold=args.conf,
    )

    pipeline.print_assessment(assessment)

    print("\nNOTE: This is an automated estimate. Actual costs may vary.")
    print("   Consult a professional for accurate assessment.")


if __name__ == "__main__":
    main()
