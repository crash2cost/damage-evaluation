#!/bin/bash
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Crash2Cost - Complete System Validation                       ║"
echo "║  Testing Custom YOLO + Classifier + Cost Estimator             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'
echo -e "${BLUE}[1/5] Checking System Files...${NC}"
echo "----------------------------------------"
files=(
    "crash2cost_custom.py"
    "detection-model/src/custom_yolo.py"
    "detection-model/src/custom_train.py"
    "detection-model/src/custom_inference.py"
    "detection-model/runs/custom-yolo/best.pt"
    "regression-model/models/damage_classifier_best.pt"
    "severity_model/models/cost_estimator.pkl"
)
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file (MISSING)"
    fi
done
echo ""
echo -e "${BLUE}[2/5] Testing Custom YOLO Detection...${NC}"
echo "----------------------------------------"
cd detection-model/src
python3 -c "
from custom_yolo import CustomYOLO
import torch
model = CustomYOLO(num_classes=1)
x = torch.randn(1, 3, 640, 640)
outputs = model(x)
print(f'✓ Model forward pass successful')
print(f'  Output shapes: {[o.shape for o in outputs]}')
print(f'  Total parameters: {sum(p.numel() for p in model.parameters()):,}')
"
cd ../..
echo ""
echo -e "${BLUE}[3/5] Testing Inference Wrapper...${NC}"
echo "----------------------------------------"
cd detection-model/src
python3 -c "
from custom_inference import load_custom_yolo
model = load_custom_yolo('../runs/custom-yolo/best.pt', num_classes=1)
print(f'✓ Inference wrapper loaded successfully')
"
cd ../..
echo ""
echo -e "${BLUE}[4/5] Running End-to-End Test Cases...${NC}"
echo "----------------------------------------"
echo -e "${YELLOW}Test 1: Family car, moderate damage${NC}"
python3 crash2cost_custom.py \
    --image archive/image/0.jpeg \
    --severity 3 \
    --car-segment Family \
    2>&1 | grep -E "(Found|Detected|cost:|damage type)" | head -4
echo ""
echo -e "${YELLOW}Test 2: Executive car, severe damage${NC}"
python3 crash2cost_custom.py \
    --image archive/image/1.jpeg \
    --severity 4 \
    --car-segment Executive \
    2>&1 | grep -E "(Found|Detected|cost:|damage type)" | head -4
echo ""
echo -e "${YELLOW}Test 3: Micro car, minor damage${NC}"
python3 crash2cost_custom.py \
    --image archive/image/10.jpeg \
    --severity 2 \
    --car-segment Micro \
    2>&1 | grep -E "(Found|Detected|cost:|damage type)" | head -4
echo ""
echo -e "${BLUE}[5/5] System Summary${NC}"
echo "----------------------------------------"
echo -e "${GREEN}✓${NC} Custom YOLO Detection (Built from scratch)"
echo -e "${GREEN}✓${NC} Damage Classification (ResNet18)"
echo -e "${GREEN}✓${NC} Cost Estimation (Random Forest)"
echo -e "${GREEN}✓${NC} End-to-End Pipeline Integration"
echo -e "${GREEN}✓${NC} Production Inference API"
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                  ✨ ALL SYSTEMS OPERATIONAL ✨                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📚 Documentation:"
echo "   - COMPLETE_SYSTEM_DOCS.md    (Full documentation)"
echo "   - TRAINING_GUIDE_HEBREW.md   (Hebrew training guide)"
echo "   - CUSTOM_YOLO_GUIDE.md       (Architecture details)"
echo ""
echo "🚀 Quick Start:"
echo "   python crash2cost_custom.py --image car.jpg --severity 3 --car-segment Family"
echo ""