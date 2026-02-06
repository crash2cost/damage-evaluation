import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
from custom_yolo import CustomYOLO
def load_model(weights_path, num_classes=1):
    model = CustomYOLO(num_classes=num_classes)
    checkpoint = torch.load(weights_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model
def predict_image(model, img_path, conf_threshold=0.25, img_size=640):
    img = Image.open(img_path).convert('RGB')
    original_size = img.size
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        predictions = model(img_tensor)
    return predictions, img, original_size
def visualize_prediction(img, predictions, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    pred_scale_1 = predictions[0][0]
    heatmap = torch.max(pred_scale_1, dim=0)[0].cpu().numpy()
    axes[1].imshow(heatmap, cmap='hot')
    axes[1].set_title('Detection Heatmap (80x80 scale)')
    axes[1].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close()
def main():
    parser = argparse.ArgumentParser(description='Custom YOLO Prediction')
    parser.add_argument('--weights', type=str, default='../runs/custom-yolo/best.pt',
                        help='Path to trained weights')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image or directory')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--num-classes', type=int, default=1,
                        help='Number of classes')
    parser.add_argument('--save-dir', type=str, default='../runs/custom-predict',
                        help='Directory to save results')
    args = parser.parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print("Loading model...")
    model = load_model(args.weights, args.num_classes)
    print(f"Model loaded from {args.weights}")
    source_path = Path(args.source)
    if source_path.is_file():
        img_files = [source_path]
    else:
        img_files = list(source_path.glob('*.jpg')) + \
                   list(source_path.glob('*.jpeg')) + \
                   list(source_path.glob('*.png'))
    print(f"Found {len(img_files)} images")
    for i, img_path in enumerate(img_files):
        print(f"\nProcessing {img_path.name}...")
        predictions, img, original_size = predict_image(
            model, img_path, args.conf
        )
        print(f"Prediction shapes:")
        for j, pred in enumerate(predictions):
            print(f"  Scale {j+1}: {pred.shape}")
        save_path = save_dir / f"{img_path.stem}_prediction.png"
        visualize_prediction(img, predictions, save_path)
        if i >= 4:
            print(f"\nProcessed first 5 images. Remaining: {len(img_files) - 5}")
            break
    print(f"\n Results saved to {save_dir}")
if __name__ == '__main__':
    main()