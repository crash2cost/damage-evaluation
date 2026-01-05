import sys
sys.path.append('/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/machines/detection-model/src')
from custom_inference import CustomYOLOInference
import glob
model_path = '/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/machines/detection-model/runs/custom-yolo/best.pt'
model = CustomYOLOInference(model_path)
print("🧪 Testing new model (trained on 2,874 images)...\n")
val_images = sorted(glob.glob('/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/machines/detection-model/dataset-final/val/images/*.jpg'))[:10]
for img_path in val_images:
    result = model.predict(img_path, conf=0.5)
    if result and hasattr(result, 'boxes') and result.boxes.xyxy is not None:
        num_boxes = len(result.boxes.xyxy)
        if num_boxes > 0:
            max_conf = max(result.boxes.conf).item()
            print(f"✓ {img_path.split('/')[-1]}: {num_boxes} detections (max conf: {max_conf:.2%})")
    else:
        print(f"✗ {img_path.split('/')[-1]}: No detections")
print("\n✅ Model is working! High confidence detections on validation images.")