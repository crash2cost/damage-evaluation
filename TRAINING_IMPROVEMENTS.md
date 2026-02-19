# Training Results & Improvements

## Current Status (Updated)

### Detection Model (YOLOv8n) - Best Run: roboflow-final
- **mAP@50: 97.12%**
- **mAP@50-95: 94.93%**
- **Precision: 100%**
- **Recall: 94.27%**
- Dataset expanded from 58 → 2,874 images (49.5x improvement)
- Training: 2,299 images | Validation: 575 images
- 8 damage classes: bumper_dent, bumper_scratch, door_dent, door_scratch, glass_shatter, head_lamp, tail_lamp, unknown

### Severity Classification Model — Backbone Comparison
- **Best Model: ResNet50** (Val: 89.22%, Test: 87.50%)
- Dataset: 1,363 train / 269 val / 264 test = 1,896 total images
- 7 classes with augmented data (aug_auto_* files)
- After hyperparameter tuning: overfitting eliminated (train acc ≤ val acc)

### Cost Estimation Model (Random Forest)
- **R²: 0.9778**
- **MAE: ₪26.94**
- **RMSE: ₪38.03**
- **MAPE: 7.51%**
- Synthetic dataset with ~50,000 rows
- Feature importance: Severity (43.2%) > Damage Type (35.6%) > Part (21.2%)

## Literature Review

### Object Detection - YOLOv8
- Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLO* (Version 8.0). GitHub. https://github.com/ultralytics/ultralytics
- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. CVPR 2016.
- Wang, C.Y., Bochkovskiy, A., & Liao, H.Y.M. (2023). *YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors*. CVPR 2023.

**Why YOLOv8**: Single-stage detector optimized for real-time inference. The nano variant (YOLOv8n) balances speed and accuracy — critical for a production pipeline where detection is the first stage.

### Image Classification - ResNet / EfficientNet
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
- Tan, M. & Le, Q.V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML 2019.

**Why ResNet50**: Residual connections solve the vanishing gradient problem. After a controlled comparison of all 3 backbones (see Backbone Comparison section), ResNet50 (23.5M params) achieved the best accuracy (89.22% val, 87.50% test), outperforming both ResNet18 (81.78% val) and EfficientNet_B0 (85.87% val). The deeper architecture extracts richer features for damage severity classification.

### Transfer Learning
- Pan, S.J. & Yang, Q. (2010). *A Survey on Transfer Learning*. IEEE Transactions on Knowledge and Data Engineering, 22(10), 1345-1359.
- Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). *How transferable are features in deep neural networks?*. NeurIPS 2014.

**Application**: All classification backbones use ImageNet-1K pretrained weights. Early layers (generic edge/texture detectors) are frozen; only later layers are fine-tuned for car damage severity — a standard approach for small datasets.

### Cost Estimation - Random Forest & Ensemble Methods
- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
- Friedman, J.H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics, 29(5), 1189-1232.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825-2830.

**Why Random Forest**: Non-parametric ensemble method that handles categorical features (damage type, part, severity) naturally. Outperformed Gradient Boosting in our comparison (R²=0.9778 vs R²=0.9751). Feature importance analysis reveals severity is the dominant cost predictor (43.2%).

### Regularization Techniques Used
- Müller, R., Kornblith, S., & Hinton, G. (2019). *When Does Label Smoothing Help?*. NeurIPS 2019.
- Srivastava, N., Hinton, G., et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. JMLR, 15, 1929-1958.
- Shorten, C. & Khoshgoftaar, T.M. (2019). *A survey on Image Data Augmentation for Deep Learning*. Journal of Big Data, 6(1), 60.

**Application**: Label smoothing (0.1), dropout (0.5), weight decay (L2), heavy data augmentation (rotation, color jitter, perspective, random erasing), and WeightedRandomSampler for class imbalance — all used to combat overfitting on the small severity dataset.

### Vehicle Damage Assessment (Domain)
- Patil, K., Kulkarni, M., Sriraman, A., & Karande, S. (2017). *Deep Learning Based Car Damage Classification*. IEEE International Conference on Big Data.
- Jayawardena, S. (2013). *Image Based Automatic Vehicle Damage Detection*. PhD thesis, Australian National University.

**Relevance**: Our 3-stage pipeline (Detect → Classify Severity → Estimate Cost) follows the industry approach of decomposing vehicle damage assessment into sequential specialized models rather than attempting end-to-end prediction.

---

## Detection Model - Training Runs Comparison

| Run | Epochs | mAP@50 | mAP@50-95 | Precision | Recall |
|-----|--------|--------|-----------|-----------|--------|
| **roboflow-final** | **50** | **97.12%** | **94.93%** | **100%** | **94.27%** |
| trained-detector | 6 | 93.64% | 85.14% | 100% | 87.30% |
| train (default) | 20 | 91.26% | 70.56% | 100% | 82.54% |
| iteration_10 | 20 | 38.33% | 22.01% | 32.27% | 46.37% |
| auto_label_model | 36 | 12.53% | 7.32% | 41.75% | 15.14% |

## Severity Model - Backbone Comparison (ResNet18 vs ResNet50 vs EfficientNet_B0)

All 3 backbones trained with identical hyperparameters, same seed (42), 50 epochs max with early stopping.

| Backbone | Total Params | Val Acc | Test Acc | Epochs Run | Training Time |
|----------|-------------|---------|----------|------------|---------------|
| ResNet18 | 11,180,103 | 81.78% | 81.06% | 28 (early stop) | 138s |
| **ResNet50** | **23,522,375** | **89.22%** | **87.50%** | **50** | **306s** |
| EfficientNet_B0 | 4,016,515 | 85.87% | 84.85% | 37 (early stop) | 359s |

**Conclusion**: ResNet50 is the best backbone for this task — +7.4% val accuracy vs ResNet18, +3.4% vs EfficientNet_B0. The larger model capacity pays off despite the small dataset, thanks to strong regularization (dropout=0.5, weight_decay=0.01, label_smoothing=0.1).

### Per-Class Test Accuracy (by backbone):

| Class | ResNet18 | ResNet50 | EfficientNet_B0 |
|-------|----------|----------|-----------------|
| bumper_dent | 70.97% | **90.32%** | 93.55% |
| bumper_scratch | 92.86% | 92.86% | 89.29% |
| door_dent | 74.55% | **83.64%** | 76.36% |
| door_scratch | 90.70% | **97.67%** | 97.67% |
| glass_shatter | 97.14% | **100.0%** | 97.14% |
| head_lamp | 61.76% | 52.94% | 52.94% |
| tail_lamp | 81.58% | **94.74%** | 89.47% |

**Weak class**: `head_lamp` is the hardest class across all models (53-62%). Likely due to visual similarity with tail_lamp or insufficient diversity in training data (34 test samples).

## Severity Model - Parameter Changes (Overfitting Fix)

Identified overfitting: Train Acc ~91% vs Val Acc ~72-81%

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| Batch Size | 32 | 16 | Better generalization for small dataset (~1,363 images) |
| Learning Rate | 0.0003 | 0.0001 | Smoother convergence, reduces overfitting |
| Weight Decay | 0.001 | 0.01 | 10x stronger L2 regularization |
| Dropout | 0.4 | 0.5 | Stronger regularization in classifier head |
| Freeze Layers | 3 | 4 | Freeze through layer2, only fine-tune layer3+layer4+fc |
| Patience | 10 | 12 | More patience with lower learning rate |

**Result**: Overfitting completely eliminated. With updated parameters, train acc ≤ val acc (strong augmentation + dropout active during training but not validation).

## Dataset Details

### Severity Dataset (per class):
| Class | Train | Val | Test |
|-------|-------|-----|------|
| bumper_dent | 197 | 31 | 31 |
| bumper_scratch | 199 | 27 | 28 |
| door_dent | 172 | 55 | 55 |
| door_scratch | 200 | 42 | 43 |
| glass_shatter | 199 | 38 | 35 |
| head_lamp | 199 | 39 | 34 |
| tail_lamp | 197 | 37 | 38 |
| **Total** | **1,363** | **269** | **264** |

### Detection Dataset (dataset-final):
- Train: 2,299 images
- Validation: 575 images
- Source: Roboflow (https://universe.roboflow.com/crash2cost/car-damage-detector-t1wis)

### Cost Dataset:
- ~50,000 synthetic rows
- 14 parts × 18 damage types × 5 severity levels × 8 car categories

## Next Steps
1. ~~Retrain severity model with updated parameters~~ ✓ Done
2. ~~Try ResNet50 or EfficientNet_B0 as backbone alternatives~~ ✓ Done — ResNet50 wins (89.22% val)
3. Switch production model to ResNet50 backbone (`--backbone resnet50`)
4. Consider collecting more data for underrepresented classes (bumper_scratch val=27)
5. Add per-class metrics (precision/recall/F1 per class) to identify weak classes
