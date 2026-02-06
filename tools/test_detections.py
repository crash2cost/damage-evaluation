"""Test model on sample CarDD images"""
import sys
import glob
sys.path.append('/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/machines/detection-model/src')
from custom_inference import CustomYOLOInference

model_path = '/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/machines/detection-model/runs/custom-yolo/best.pt'
model = CustomYOLOInference(model_path)

# Test on CarDD images
cardd_images = sorted(glob.glob('/Users/idolevi/.cache/huggingface/hub/datasets--harpreetsahota--CarDD/*/data/*.jpg'))[:10]
print('Testing CarDD images with conf=0.05 (lower threshold):')
for img in cardd_images:
    result = model.predict(img, conf=0.05)  # Lowered from 0.25
    if result and hasattr(result, 'boxes') and result.boxes.xyxy is not None:
        num_boxes = len(result.boxes.xyxy)
    else:
        num_boxes = 0
    print(f'  {img.split("/")[-1]}: {num_boxes} detections')

# Also test original trained images
print('\nTesting original training images:')
orig_images = sorted(glob.glob('/Users/idolevi/Library/CloudStorage/OneDrive-Personal/Desktop/crash2cost/machines/detection-model/dataset/train/images/*.jpg'))[:5]
for img in orig_images:
    result = model.predict(img, conf=0.05)  # Lowered from 0.25
    if result and hasattr(result, 'boxes') and result.boxes.xyxy is not None:
        num_boxes = len(result.boxes.xyxy)
    else:
        num_boxes = 0
    print(f'  {img.split("/")[-1]}: {num_boxes} detections')
