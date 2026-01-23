#!/usr/bin/env python3
"""Quick test of severity model on sample images."""

import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from pathlib import Path

# Load model
weights_path = Path('severity_model/models/best_model.pt')
checkpoint = torch.load(weights_path, map_location='cpu')
classes = checkpoint['classes']
print('Classes:', classes)

# Build model
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.fc.in_features, len(classes)),
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Test each class
print("\n=== Testing samples from each class ===")
for class_name in classes:
    class_dir = Path(f'severity_model/dataset/train/{class_name}')
    if not class_dir.exists():
        continue
    
    samples = list(class_dir.glob('*.jpeg'))[:3]
    correct = 0
    
    print(f"\n{class_name}:")
    for img_path in samples:
        img = Image.open(img_path).convert('RGB')
        input_t = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_t)
            probs = torch.softmax(output, dim=1)
            conf, pred = probs.max(1)
        
        predicted = classes[pred.item()]
        is_correct = "✓" if predicted == class_name else "✗"
        if predicted == class_name:
            correct += 1
        print(f"  {is_correct} {img_path.name}: {predicted} ({conf.item()*100:.1f}%)")
    
    print(f"  Accuracy: {correct}/{len(samples)}")

# Show class distribution confusion
print("\n=== Checking for bias toward specific classes ===")
all_preds = []
for class_name in classes:
    class_dir = Path(f'severity_model/dataset/train/{class_name}')
    if not class_dir.exists():
        continue
    
    samples = list(class_dir.glob('*.jpeg'))[:10]
    for img_path in samples:
        img = Image.open(img_path).convert('RGB')
        input_t = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_t)
            probs = torch.softmax(output, dim=1)
            conf, pred = probs.max(1)
        all_preds.append(classes[pred.item()])

from collections import Counter
pred_counts = Counter(all_preds)
print("\nPrediction distribution across 70 samples:")
for cls, count in sorted(pred_counts.items(), key=lambda x: -x[1]):
    print(f"  {cls}: {count} ({count/len(all_preds)*100:.1f}%)")
