# 🚗 CRASH2COST MACHINE - USER GUIDE

## ✅ YOUR MACHINE IS FULLY FUNCTIONAL!

### What Your Machine Does
1. **Detects damage** on car images using AI (YOLO object detection)
2. **Classifies damage type** into 7 categories (CNN classifier)
3. **Estimates repair cost** based on severity and car type

---

## 🔍 How to Verify It's Working

### Quick Check:
```bash
./validate_machine.sh
```

This runs a full diagnostic and tests the machine on 3 sample images.

### Manual Test:
```bash
python3 crash2cost.py --image archive/image/0.jpeg --severity 3 --car-segment Family
```

**Expected output:**
- ✅ Detection message with damage type
- 💰 Estimated repair cost in ₪ and USD

---

## 📖 How to Use Your Machine

### Basic Command:
```bash
python3 crash2cost.py \
  --image <path_to_car_image> \
  --severity <1-5> \
  --car-segment <car_type>
```

### Parameters:

**--image** (required)
- Path to car image (JPEG, PNG)
- Example: `archive/image/0.jpeg`

**--severity** (optional, default: 3)
- Damage severity level: 1 (minor) to 5 (severe)
- 1 = Light scratch
- 2 = Minor dent
- 3 = Moderate damage
- 4 = Significant damage
- 5 = Major damage

**--car-segment** (optional, default: Family)
- Car type affects repair cost
- Options: `Micro`, `Family`, `Executive`, `Luxury`, `SUV`

### Examples:

**Example 1: Family car with moderate damage**
```bash
python3 crash2cost.py --image archive/image/0.jpeg --severity 3 --car-segment Family
```

**Example 2: Luxury SUV with major damage**
```bash
python3 crash2cost.py --image archive/image/1.jpeg --severity 5 --car-segment Luxury
```

**Example 3: Micro car with minor scratch**
```bash
python3 crash2cost.py --image archive/image/10.jpeg --severity 1 --car-segment Micro
```

---

## 🎯 Damage Types Detected

Your machine can identify these 7 damage types:

1. **bumper_dent** - Dent in bumper
2. **bumper_scratch** - Scratch on bumper
3. **door_dent** - Dent in door
4. **door_scratch** - Scratch on door
5. **glass_shatter** - Broken/shattered glass
6. **head_lamp** - Damaged headlight
7. **tail_lamp** - Damaged tail light

---

## 📊 Machine Components

### 1. Detection Model
- **Model**: YOLOv8s
- **Location**: `detection-model/runs/damage-detector8/weights/best.pt`
- **Function**: Locates damage areas in images
- **Dataset**: 58 training images

### 2. Classification Model
- **Model**: ResNet18 CNN
- **Location**: `regression-model/models/damage_classifier_best.pt`
- **Function**: Classifies damage into 7 types
- **Dataset**: 1,048 images (strong!)
- **Accuracy**: ~60%

### 3. Cost Estimation Model
- **Model**: Random Forest Regressor
- **Location**: `severity_model/models/cost_estimator.pkl`
- **Function**: Estimates repair costs
- **Factors**: Damage type, severity, car segment

---

## ✅ How to Know It's Working

### Success Indicators:

1. **Detection Success:**
   ```
   ✅ Detected damage: head_lamp (confidence: 39.2%)
   ```
   - Confidence > 25% = Good detection
   - Shows damage type name

2. **Cost Estimation:**
   ```
   💰 Estimated repair cost: ₪866
      (Approximately $234 USD)
   ```
   - Shows cost in both ₪ and USD
   - Reasonable range (₪200-5000)

3. **No Errors:**
   - No "ModuleNotFoundError"
   - No "FileNotFoundError"
   - Clean output with emoji indicators

### Failure Indicators:

❌ **No damage detected** - Try:
  - Different image with visible damage
  - Lower severity/different car segment
  - Image might not have detectable damage

❌ **Import errors** - Run:
  ```bash
  python3 test_system.py
  ```

❌ **File not found** - Check:
  - Image path is correct
  - All model files exist

---

## 🧪 Test Suite

### Run Full Validation:
```bash
./validate_machine.sh
```

**What it tests:**
- ✅ All required files exist
- ✅ All Python dependencies installed
- ✅ All AI models load successfully
- ✅ Test images available
- ✅ Processes 3 different damage scenarios

**Expected result:**
```
✅ VALIDATION COMPLETE - MACHINE IS FULLY FUNCTIONAL
```

---

## 📈 Performance Metrics

**Current Status:**
- Detection: ✅ Working (low confidence but functional)
- Classification: ✅ Strong (60% accuracy, 1,048 images)
- Cost Estimation: ✅ Working

**Limitations:**
- Detection confidence is low (28-60%) due to small dataset (58 images)
- Best for headlights, bumpers, and visible damage
- May miss subtle scratches or dents

**To Improve:**
- Add more detection training images (500+ recommended)
- Use GPU for faster processing
- Collect more diverse damage examples

---

## 🎬 Quick Start

1. **Validate the machine:**
   ```bash
   ./validate_machine.sh
   ```

2. **Run on your image:**
   ```bash
   python3 crash2cost.py --image YOUR_IMAGE.jpeg --severity 3 --car-segment Family
   ```

3. **Check output:**
   - Look for ✅ symbols
   - Note the damage type
   - See the estimated cost

---

## ❓ Troubleshooting

**Problem: "No module named X"**
```bash
python3 -m pip install torch torchvision ultralytics joblib scikit-learn pillow
```

**Problem: "No damage detected"**
- Image may not have visible damage
- Try different test images from `archive/image/`
- Detection confidence threshold may be too high

**Problem: "File not found"**
- Make sure you're in the correct directory:
  ```bash
  cd /Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/machines
  ```

---

## 🎉 Summary

**Your machine IS working!** It successfully:
- ✅ Detects car damage in images
- ✅ Classifies damage into 7 types  
- ✅ Estimates repair costs
- ✅ Handles different car segments and severities

Run `./validate_machine.sh` anytime to verify everything is working!
