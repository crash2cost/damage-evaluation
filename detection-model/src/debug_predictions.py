import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from custom_yolo import CustomYOLO

# Setup
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
checkpoint_path = '../runs/custom-yolo/best.pt'
image_path = 'dataset-final/val/images/auto_000055.jpg'

# Load model
model = CustomYOLO(num_classes=1)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

print(f"Loaded model - Best loss: {checkpoint['best_loss']:.4f}, Epoch: {checkpoint['epoch']+1}")

# Load and transform image
img = Image.open(image_path).convert('RGB')
transform = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
])
img_tensor = transform(img).unsqueeze(0).to(device)

print(f"\nInput shape: {img_tensor.shape}")

# Get predictions
with torch.no_grad():
    predictions = model(img_tensor)

# Debug predictions
print(f"\nPredictions type: {type(predictions)}")
if isinstance(predictions, (list, tuple)):
    print(f"Number of prediction heads: {len(predictions)}")
    for i, pred in enumerate(predictions):
        print(f"\nHead {i}:")
        print(f"  Shape: {pred.shape}")
        print(f"  Min: {pred.min().item():.4f}, Max: {pred.max().item():.4f}")
        print(f"  Mean: {pred.mean().item():.4f}")
        
        # Check objectness scores
        if pred.shape[-1] >= 5:
            objectness = pred[0, :, 4]  # Get objectness for batch 0
            print(f"  Objectness - Min: {objectness.min().item():.4f}, Max: {objectness.max().item():.4f}")
            print(f"  Objectness - Mean: {objectness.mean().item():.4f}")
            print(f"  Num with >0.1: {(objectness > 0.1).sum().item()}")
            print(f"  Num with >0.25: {(objectness > 0.25).sum().item()}")
            print(f"  Num with >0.5: {(objectness > 0.5).sum().item()}")
            
            # Show top 5 objectness scores
            top_vals, top_idx = objectness.topk(5)
            print(f"  Top 5 objectness scores: {top_vals.tolist()}")
            
            # Show corresponding boxes for top scores
            for j in range(min(3, len(top_vals))):
                idx = top_idx[j]
                box = pred[0, idx, :4]
                obj = pred[0, idx, 4]
                print(f"    Box {j}: x={box[0]:.3f}, y={box[1]:.3f}, w={box[2]:.3f}, h={box[3]:.3f}, obj={obj:.3f}")
else:
    print(f"Single prediction shape: {predictions.shape}")
