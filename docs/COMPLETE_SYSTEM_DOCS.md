# Crash2Cost - Complete System Documentation

## 🎯 Overview

Crash2Cost is an end-to-end AI-powered system for automated car damage assessment and repair cost estimation. The system combines three machine learning models to provide accurate damage detection, classification, and cost prediction.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   INPUT: Car Damage Image                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: Damage Detection (Custom YOLO - Built from Scratch)│
│  • Locates damage regions in the image                      │
│  • Returns bounding boxes with confidence scores            │
│  • 3-scale detection (80x80, 40x40, 20x20)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: Damage Classification (ResNet18)                  │
│  • Classifies damage type from detected regions             │
│  • 7 damage categories: bumper_dent, bumper_scratch,        │
│    door_dent, door_scratch, glass_shatter, head_lamp,       │
│    tail_lamp                                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: Cost Estimation (Random Forest Regressor)         │
│  • Predicts repair cost based on:                           │
│    - Damage type → Affected part                            │
│    - Severity level (1-5)                                   │
│    - Car segment (Micro/Family/Executive/Luxury/SUV)        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           OUTPUT: Estimated Repair Cost (₪ / $)              │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Components

### 1. Custom YOLO Detection Model

**Location**: `detection-model/`

**Key Features**:
- ✅ Built completely from scratch using PyTorch primitives (no pre-built YOLO)
- ✅ YOLOv8-nano architecture (3M parameters)
- ✅ Custom training with data augmentation
- ✅ 100 epochs trained on 58 images
- ✅ Final validation loss: 0.0508

**Files**:
- `src/custom_yolo.py` - Full YOLO architecture (ConvBlock, C2fBlock, SPPF, Backbone, Neck, Head)
- `src/custom_train.py` - Training script with augmentation
- `src/custom_loss.py` - YOLO loss function
- `src/custom_inference.py` - Inference wrapper with NMS
- `runs/custom-yolo/best.pt` - Trained weights

**Data Augmentation**:
- Random horizontal flip (50%)
- Color jitter (brightness, contrast, saturation, hue)
- Random affine (rotation, translation, scale, shear)
- Random perspective distortion
- Random erasing

### 2. Damage Classification Model

**Location**: `regression-model/`

**Key Features**:
- ResNet18 architecture
- Pre-trained on ImageNet, fine-tuned on car damage
- 7 damage type categories
- 1,048 training images

**Files**:
- `models/damage_classifier_best.pt` - Trained weights
- `dataset/` - Classification dataset

### 3. Cost Estimation Model

**Location**: `severity_model/`

**Key Features**:
- Random Forest Regressor
- Inputs: part name, severity, car segment
- Outputs: Repair cost in ₪ (Israeli Shekel)

**Files**:
- `models/cost_estimator.pkl` - Trained model
- `models/part_encoder.pkl` - Label encoder for parts
- `models/segment_encoder.pkl` - Label encoder for segments

## 🚀 Usage

### Quick Start

Run the complete custom pipeline:

```bash
python crash2cost_custom.py --image <path_to_image> --severity 3 --car-segment Family --save-viz
```

### Parameters

- `--image` (required): Path to car damage image
- `--severity` (1-5): Damage severity level
  - 1 = Very minor scratch
  - 2 = Minor dent
  - 3 = Moderate damage (default)
  - 4 = Significant damage
  - 5 = Severe/structural damage
- `--car-segment`: Car type
  - `Micro` - Small economy cars
  - `Family` - Standard sedans (default)
  - `Executive` - Premium sedans
  - `Luxury` - High-end vehicles
  - `SUV` - Sport utility vehicles
- `--conf`: Detection confidence threshold (default: 0.25)
- `--save-viz`: Save detection visualization
- `--device`: Computing device (cpu/cuda/mps)

### Example Commands

```bash
# Basic usage
python crash2cost_custom.py --image car_damage.jpg --severity 3 --car-segment Family

# Luxury car with severe damage
python crash2cost_custom.py --image tesla_crash.jpg --severity 5 --car-segment Luxury --save-viz

# Micro car with minor scratch
python crash2cost_custom.py --image fiat_scratch.jpg --severity 1 --car-segment Micro
```

### Output Example

```
🚗 Crash2Cost - Complete Damage Assessment System (Custom YOLO)
======================================================================
Loading models...
  [1/3] Loading custom YOLO detector...
  ✓ Loaded custom YOLO model
  [2/3] Loading damage classifier...
  [3/3] Loading cost estimator...

======================================================================

📸 STEP 1: Detecting damage regions in car_damage.jpg
   Found 1 damage region(s)
   - Region 1: [55, 56, 168, 168] (confidence: 50.23%)
   Saved detection visualization to car_damage_detection.jpg

🔍 STEP 2: Classifying damage type
   Detected damage: door_dent
   Classification confidence: 32.4%
   Affected part: Front Door

💰 STEP 3: Estimating repair cost
   Severity level: 3/5
   Car segment: Family

======================================================================
📋 FINAL ASSESSMENT
======================================================================

✅ Estimated repair cost: ₪2,077
   (Approximately $561 USD)

📊 Summary:
   • Damage regions detected: 1
   • Damage type: door_dent
   • Affected part: Front Door
   • Severity: 3/5
   • Car segment: Family
```

## 🛠️ Training Custom Models

### Train Detection Model

```bash
cd detection-model/src
python custom_train.py \
  --data ../dataset/data.yaml \
  --epochs 100 \
  --batch-size 8 \
  --lr 0.0005 \
  --num-classes 1 \
  --workers 0
```

### Test Detection Model

```bash
cd detection-model/src
python custom_predict.py \
  --source ../dataset/valid/images \
  --weights ../runs/custom-yolo/best.pt
```

## 📊 Model Performance

### Custom YOLO Detection
- **Training Loss**: 0.0508 (from 2.22)
- **Validation Loss**: 0.0508 (from 2.09)
- **Training Time**: ~15 minutes on Apple M4 Max (CPU)
- **Inference Speed**: ~1 second per image
- **Architecture**: YOLOv8-nano (3,011,027 parameters)

### Damage Classifier
- **Dataset**: 1,048 images across 7 classes
- **Accuracy**: ~85% on validation set
- **Architecture**: ResNet18 (pre-trained)

### Cost Estimator
- **Model**: Random Forest Regressor
- **Features**: Part type, severity, car segment
- **Output Range**: ₪500 - ₪15,000

## 📁 Project Structure

```
crash2cost/
├── crash2cost.py                 # Original pipeline (Ultralytics YOLO)
├── crash2cost_custom.py          # Custom pipeline (Custom YOLO)
├── detection-model/
│   ├── dataset/
│   │   ├── train/                # 58 training images
│   │   └── valid/                # 11 validation images
│   ├── src/
│   │   ├── custom_yolo.py        # Custom YOLO architecture
│   │   ├── custom_train.py       # Training script
│   │   ├── custom_loss.py        # Loss functions
│   │   ├── custom_inference.py   # Inference wrapper
│   │   └── custom_predict.py     # Prediction script
│   └── runs/
│       └── custom-yolo/
│           └── best.pt           # Trained weights
├── regression-model/
│   ├── dataset/                  # Classification dataset
│   ├── models/
│   │   └── damage_classifier_best.pt
│   └── src/
│       ├── train_cnn.py
│       └── evaluate.py
├── severity_model/
│   ├── models/
│   │   ├── cost_estimator.pkl
│   │   ├── part_encoder.pkl
│   │   └── segment_encoder.pkl
│   └── src/
│       ├── train_regressor.py
│       └── price_logic.py
└── archive/
    └── image/                    # Test images
```

## 🧪 Testing the System

### End-to-End Test

```bash
# Test with sample images
python crash2cost_custom.py --image archive/image/0.jpeg --severity 3 --car-segment Family --save-viz
python crash2cost_custom.py --image archive/image/1.jpeg --severity 4 --car-segment Executive --save-viz
```

### Component Tests

```bash
# Test detection only
cd detection-model/src
python custom_predict.py --source ../../archive/image/0.jpeg

# Test classification (requires detection first)
cd regression-model/src
python evaluate.py --image ../../archive/image/0.jpeg
```

## 🔧 Technical Details

### Custom YOLO Architecture

**Backbone** (Feature Extraction):
1. ConvBlock (3→16, stride=2) + C2fBlock(16→16)
2. ConvBlock (16→32, stride=2) + C2fBlock(32→32)
3. ConvBlock (32→64, stride=2) + C2fBlock(64→64)
4. ConvBlock (64→128, stride=2) + C2fBlock(128→128)
5. ConvBlock (128→256, stride=2) + C2fBlock(256→256) + SPPF

**Neck** (Feature Pyramid):
- Upsampling + concatenation for multi-scale features
- C2fBlock at each scale

**Head** (Detection):
- 3 detection heads at different scales (80×80, 40×40, 20×20)
- Each outputs: [batch, 65, H, W]
  - 4 bbox coordinates (x, y, w, h)
  - 1 objectness score
  - 60 class probabilities (expandable)

### Post-Processing Pipeline

1. **Prediction Decoding**:
   - Convert YOLO outputs to bounding boxes
   - Apply sigmoid activation to coordinates
   - Calculate confidence scores (objectness × class score)

2. **Non-Maximum Suppression (NMS)**:
   - Filter overlapping detections
   - IoU threshold: 0.45
   - Confidence threshold: 0.25 (default)

3. **Coordinate Scaling**:
   - Scale from model space (640×640) to original image size

## 📝 Key Implementation Details

### Data Augmentation (Training Only)

```python
transform = T.Compose([
    T.Resize((640, 640)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    T.RandomPerspective(distortion_scale=0.2, p=0.3),
    T.ToTensor(),
    T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])
```

### Training Configuration

- **Optimizer**: Adam (lr=0.0005, weight_decay=0.0005)
- **Scheduler**: Cosine annealing
- **Batch Size**: 8
- **Epochs**: 100
- **Loss Function**: Combined box loss + classification loss
- **Early Stopping**: Best model saved based on validation loss

## 🎓 Learning Resources

### Understanding YOLO Architecture
- See `TRAINING_GUIDE_HEBREW.md` for detailed Hebrew explanation
- See `CUSTOM_YOLO_GUIDE.md` for architecture details

### Understanding the Code
- All code is fully documented with docstrings
- Each component is modular and can be studied independently
- Custom implementation allows full control and understanding

## 🚧 Known Limitations

1. **Small Dataset**: Only 58 training images for detection (augmentation helps)
2. **Single Class Detection**: Currently detects "damage" as one class
3. **Simplified Loss**: Basic YOLO loss implementation
4. **No Anchor Boxes**: Simplified anchor-free approach
5. **Detection Confidence**: May vary due to small dataset

## 🔮 Future Improvements

1. **Multi-Class Detection**: Detect specific damage types (dent, scratch, crack)
2. **Larger Dataset**: Collect more diverse damage images
3. **Advanced Augmentation**: Mosaic, MixUp, CutOut
4. **Model Optimization**: Quantization, pruning for faster inference
5. **Web Interface**: Deploy as web service with API
6. **Mobile App**: Port to mobile devices (CoreML, TFLite)

## 📄 License & Credits

This project demonstrates a complete ML pipeline built from scratch for educational purposes.

**Technologies Used**:
- PyTorch (Deep Learning)
- torchvision (Computer Vision)
- Pillow (Image Processing)
- scikit-learn (Machine Learning)
- NumPy (Numerical Computing)

## 🤝 Contributing

This is a complete, working machine learning system. Feel free to:
- Experiment with different architectures
- Try different augmentation strategies
- Expand the dataset
- Add new features

## 📞 Support

For questions or issues, refer to:
- `TRAINING_GUIDE_HEBREW.md` - Detailed training guide in Hebrew
- `CUSTOM_YOLO_GUIDE.md` - Architecture documentation
- `README.md` - Project overview

---

**Built with ❤️ using Custom PyTorch Implementation**

✨ **The entire YOLO model was built from scratch - no pre-built YOLO libraries used!**
