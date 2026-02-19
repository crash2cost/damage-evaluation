#!/usr/bin/env python3

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

ROOT = Path(__file__).resolve().parent

DETECTION_WEIGHTS = ROOT / "detection-model" / "runs" / "train" / "weights" / "best.pt"
SEVERITY_WEIGHTS = ROOT / "severity-model" / "models" / "best_model.pt"
COST_MODEL = ROOT / "cost-model" / "models" / "cost_estimator.pkl"
PART_ENCODER = ROOT / "cost-model" / "models" / "part_encoder.pkl"
DAMAGE_ENCODER = ROOT / "cost-model" / "models" / "damage_encoder.pkl"
CONFIG_FILE = ROOT / "config" / "damage_mappings.json"

SEVERITY_INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_CONFIDENCE_THRESHOLD = 0.25
FALLBACK_CONFIDENCE_THRESHOLDS = (0.1, 0.05, 0.02, 0.01)
SEVERITY_FALLBACK_THRESHOLD = 0.4  # Below this YOLO conf, use severity model as fallback

REPLACE_SEVERITY_THRESHOLD = 4
DEFAULT_UNKNOWN_SEVERITY = 3
MAX_SEVERITY = 5

DEFAULT_FALLBACK_COST = 1000
DEFAULT_UNKNOWN_PART = "Front Bumper"
FALLBACK_BASE_COSTS = {"scratch": 500, "dent": 1500, "shatter": 3000, "lamp": 800}
SEVERITY_DIVISOR = 3

DEFAULT_BACKBONE = "resnet18"
DEFAULT_DROPOUT_RATE = 0.3

SEPARATOR_WIDTH = 60
MS_PER_SECOND = 1000

# TTA configuration
TTA_SCALES = (0.8, 1.2)
TTA_NMS_IOU_THRESHOLD = 0.5


@dataclass
class DamageDetection:
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


@dataclass
class DamageAssessment:
    detection: DamageDetection
    damage_type: str
    damage_confidence: float
    severity: int
    estimated_cost: float
    repair_or_replace: str


@dataclass
class VehicleAssessment:
    damages: List[DamageAssessment]
    total_cost: float
    inference_time_ms: float


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


def _compute_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Crash2CostPipeline:

    def __init__(
        self,
        detection_weights: Optional[str] = None,
        severity_weights: Optional[str] = None,
        cost_model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        self.device = get_device() if device == "auto" else torch.device(device)
        print(f"Using device: {self.device}")

        self._load_config(config_path or str(CONFIG_FILE))

        self._load_detection_model(detection_weights or str(DETECTION_WEIGHTS))
        self._load_severity_model(severity_weights or str(SEVERITY_WEIGHTS))
        self._load_cost_model(cost_model_path or str(COST_MODEL))

        self.severity_transform = transforms.Compose([
            transforms.Resize((SEVERITY_INPUT_SIZE, SEVERITY_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def _load_config(self, config_path: str) -> None:
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

            self.damage_to_part = {
                k: v for k, v in config.get("damage_to_part", {}).items()
                if not k.startswith("_")
            }
            self.damage_to_severity = {
                k: v for k, v in config.get("damage_to_severity", {}).items()
                if not k.startswith("_")
            }

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
        self.severity_model: Optional[nn.Module] = None
        self.severity_classes: List[str] = []

        if not Path(weights_path).exists():
            print(f"Severity weights not found: {weights_path}")
            print("   Run: python severity-model/train.py")
            return

        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.severity_classes = checkpoint["classes"]

            config = checkpoint.get("config", checkpoint.get("hyperparameters", {}))

            num_classes = len(self.severity_classes)
            backbone = config.get("backbone", DEFAULT_BACKBONE)
            dropout_rate = config.get("dropout_rate", DEFAULT_DROPOUT_RATE)

            if backbone.startswith("efficientnet"):
                model = self._create_efficientnet_model(backbone, num_classes, dropout_rate)
            elif backbone == "resnet50":
                model = models.resnet50(weights=None)
                model.fc = nn.Sequential(
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(model.fc.in_features, num_classes),
                )
            else:
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

    def detect_damage_tta(
        self, image: Image.Image, conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    ) -> List[DamageDetection]:
        """Run detection with test-time augmentation for higher accuracy."""
        all_detections = []

        # Original image
        all_detections.extend(self.detect_damage(image, conf_threshold))

        # Horizontally flipped
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        flip_dets = self.detect_damage(flipped, conf_threshold)
        for det in flip_dets:
            x1, y1, x2, y2 = det.bbox
            det.bbox = (image.width - x2, y1, image.width - x1, y2)
        all_detections.extend(flip_dets)

        # Multi-scale
        for scale in TTA_SCALES:
            w, h = int(image.width * scale), int(image.height * scale)
            if w < 32 or h < 32:
                continue
            scaled = image.resize((w, h), Image.BILINEAR)
            scale_dets = self.detect_damage(scaled, conf_threshold)
            for det in scale_dets:
                x1, y1, x2, y2 = det.bbox
                det.bbox = (x1 / scale, y1 / scale, x2 / scale, y2 / scale)
            all_detections.extend(scale_dets)

        # Simple NMS to merge overlapping detections
        return self._nms_detections(all_detections)

    @staticmethod
    def _nms_detections(detections: List[DamageDetection]) -> List[DamageDetection]:
        """Non-maximum suppression across TTA detections."""
        if not detections:
            return []

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        kept = []

        for det in detections:
            suppress = False
            for kept_det in kept:
                iou = _compute_iou(det.bbox, kept_det.bbox)
                if iou > TTA_NMS_IOU_THRESHOLD:
                    suppress = True
                    break
            if not suppress:
                kept.append(det)

        return kept

    def classify_damage(
        self, image: Image.Image, bbox: Tuple[float, float, float, float]
    ) -> Tuple[str, float]:
        if self.severity_model is None:
            return "unknown", 0.0

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
        if self.cost_model is None or self.damage_encoder is None:
            for key, cost in FALLBACK_BASE_COSTS.items():
                if key in damage_type.lower():
                    return cost * severity / SEVERITY_DIVISOR
            return float(DEFAULT_FALLBACK_COST)

        part_name = self.damage_to_part.get(damage_type, self.default_unknown_part)

        try:
            part_enc = self.part_encoder.transform([part_name])[0]
            damage_enc = self.damage_encoder.transform([damage_type])[0]
            cost = self.cost_model.predict([[part_enc, damage_enc, severity]])[0]
            return float(cost)
        except ValueError:
            return float(self.fallback_cost)

    def get_severity(self, damage_type: str) -> int:
        for key, severity in self.damage_to_severity.items():
            if key in damage_type.lower():
                return severity
        return self.default_unknown_severity

    def assess_damage(
        self,
        image_input: Union[str, Image.Image],
        car_segment: str = "Family",
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        use_tta: bool = False,
    ) -> VehicleAssessment:
        start_time = time.time()

        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be a file path or PIL Image")

        detect_fn = self.detect_damage_tta if use_tta else self.detect_damage
        detections = detect_fn(image, conf_threshold)
        if not detections:
            for threshold in FALLBACK_CONFIDENCE_THRESHOLDS:
                if threshold >= conf_threshold:
                    continue
                detections = detect_fn(image, threshold)
                if detections:
                    print(f"No detections at conf={conf_threshold}. "
                          f"Retrying at conf={threshold} ({len(detections)} found).")
                    break

        assessments = []
        for detection in detections:
            # Single-stage: YOLO directly provides damage type for multiclass models
            # Fall back to severity classifier if YOLO class is generic "damage"
            # or confidence is low
            if (detection.class_name in ("damage", "unknown")
                    or detection.confidence < SEVERITY_FALLBACK_THRESHOLD):
                damage_type, damage_conf = self.classify_damage(image, detection.bbox)
            else:
                damage_type = detection.class_name
                damage_conf = detection.confidence

            severity = self.get_severity(damage_type)

            cost = self.estimate_cost(damage_type, severity, car_segment)

            repair_or_replace = "Replace" if severity >= REPLACE_SEVERITY_THRESHOLD else "Repair"

            assessments.append(DamageAssessment(
                detection=detection,
                damage_type=damage_type,
                damage_confidence=damage_conf,
                severity=severity,
                estimated_cost=cost,
                repair_or_replace=repair_or_replace,
            ))

        total_cost = sum(a.estimated_cost for a in assessments)
        inference_time = (time.time() - start_time) * MS_PER_SECOND

        return VehicleAssessment(
            damages=assessments,
            total_cost=total_cost,
            inference_time_ms=inference_time,
        )

    def print_assessment(self, assessment: VehicleAssessment) -> None:
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
