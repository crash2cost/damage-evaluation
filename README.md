# Crash2Cost - AI-Powered Vehicle Damage Cost Estimation

##  Overview
Automated system that analyzes car damage images and estimates repair costs using computer vision and machine learning.

##  Project Structure

```
machines/
├── pipeline.py              # Main inference pipeline (uses all 3 models)
├── detection-model/
│   ├── train.py            # YOLOv8 training script (uses ultralytics)
│   ├── dataset/            # Training data
│   └── runs/               # Training outputs & weights
├── severity_model/
│   ├── train.py            # ResNet severity classification training
│   ├── dataset/            # Training data (train/val/test splits)
│   └── models/             # Saved model weights
└── cost_model/
    ├── train.py            # Random Forest/Gradient Boosting training
    ├── dataset/            # Cost estimation training data
    └── models/             # Saved models & encoders
```

##  System Components

### 1. **Damage Detection Model** 
- **Architecture**: YOLOv8 (via Ultralytics library)
- **Output**: Bounding boxes around damage regions
- **Training**: `python detection-model/train.py`

### 2. **Severity Classification Model** 
- **Architecture**: ResNet18/50 (torchvision pretrained)
- **Classes**: bumper_dent, bumper_scratch, door_dent, door_scratch, glass_shatter, head_lamp, tail_lamp
- **Training**: `python severity_model/train.py`

### 3. **Cost Estimation Model** 
- **Algorithm**: Random Forest or Gradient Boosting (sklearn)
- **Features**: Part type, Severity level (1-5), Car segment
- **Training**: `python cost_model/train.py`

##  Quick Start

### Installation
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install torch torchvision ultralytics scikit-learn joblib pandas pillow tqdm
```

### Training Each Model

```bash
# 1. Train Detection Model (YOLOv8)
python detection-model/train.py --epochs 50 --batch 32

# 2. Train Severity Classifier
python severity_model/train.py --epochs 50 --backbone resnet18

# 3. Train Cost Estimator
python cost_model/train.py
```

### Run Inference
```bash
# Assess damage in an image
python pipeline.py --image path/to/damage.jpg --car-segment Family
```

### Example Output
```
 Crash2Cost - Car Damage Assessment System
============================================================

 DAMAGE ASSESSMENT REPORT
============================================================

 Damage #1
   Type: bumper_dent
   Confidence: 87.5%
   Severity: 3/5
   Action: Repair
   Estimated Cost: ₪1,250

============================================================
 TOTAL ESTIMATED COST: ₪1,250
Inference Time: 145ms
============================================================
```

##  Libraries Used

| Component | Library | Purpose |
|-----------|---------|---------|
| Detection | `ultralytics` | YOLOv8 object detection |
| Classification | `torchvision.models` | Pretrained ResNet for transfer learning |
| Cost Estimation | `sklearn` | RandomForest & GradientBoosting regressors |
| Data Loading | `torch.utils.data` | Efficient data loading & augmentation |
| Image Processing | `PIL`, `torchvision.transforms` | Image preprocessing |

##  Training Options

### Detection Model
```bash
python detection-model/train.py \
  --epochs 100 \
  --batch 16 \
  --img-size 640 \
  --model s \          # n/s/m/l/x for nano to xlarge
  --patience 15 \
  --resume             # Resume from checkpoint
```

### Severity Classifier
```bash
python severity_model/train.py \
  --epochs 50 \
  --batch 32 \
  --backbone resnet50 \  # resnet18, resnet50, efficientnet_b0
  --dropout 0.3 \
  --lr 0.0003 \
  --patience 10
```

### Cost Estimator
```bash
python cost_model/train.py \
  --data-path path/to/costs.csv \
  --test-size 0.3
```

##  Expected Performance

| Model | Metric | Value |
|-------|--------|-------|
| Detection | mAP50 | ~0.85 |
| Severity | Accuracy | ~80% |
| Cost | MAE | ~₪100 |
| Cost | R² | ~0.99 |

##  Device Support

All training scripts auto-detect the best available device:
- **Apple Silicon**: Uses MPS (Metal Performance Shaders)
- **NVIDIA GPU**: Uses CUDA
- **CPU**: Fallback option

##  Notes

- The pipeline gracefully handles missing models with fallback logic
- All training scripts support early stopping to prevent overfitting
- Pretrained ImageNet weights are used for transfer learning in severity classification
- Class imbalance is handled via weighted sampling
