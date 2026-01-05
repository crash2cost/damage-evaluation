import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import yaml
from custom_yolo import CustomYOLO
from custom_loss import YOLOLoss
class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, augment=True):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        self.img_files = list(self.img_dir.glob('*.jpg')) + \
                        list(self.img_dir.glob('*.jpeg')) + \
                        list(self.img_dir.glob('*.png'))
        print(f"Found {len(self.img_files)} images in {img_dir}")
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        from PIL import Image
        import torchvision.transforms as T
        import random
        img = Image.open(img_path).convert('RGB')
        if self.augment:
            transform = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1
                ),
                T.RandomAffine(
                    degrees=10,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=5
                ),
                T.RandomPerspective(distortion_scale=0.2, p=0.3),
                T.ToTensor(),
                T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
            ])
        else:
            transform = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
            ])
        img_tensor = transform(img)
        label_path = self.label_dir / (img_path.stem + '.txt')
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        coords = [float(x) for x in parts[1:5]]
                        labels.append([cls] + coords)
        if len(labels) == 0:
            labels = torch.zeros((0, 5))
        else:
            labels = torch.tensor(labels)
        return img_tensor, labels
def collate_fn(batch):
    images = []
    targets = []
    for img, label in batch:
        images.append(img)
        targets.append(label)
    images = torch.stack(images, 0)
    return images, targets
class YOLOTrainer:
    def __init__(self, model, train_loader, val_loader, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        self.criterion = YOLOLoss(num_classes=args.num_classes)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
        self.best_loss = float('inf')
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        box_loss = 0
        cls_loss = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs}')
        for batch_idx, batch in enumerate(pbar):
            if batch_idx == 0:
                print(f"Batch type: {type(batch)}")
                print(f"Batch length: {len(batch) if hasattr(batch, '__len__') else 'N/A'}")
                if isinstance(batch, (list, tuple)):
                    for i, item in enumerate(batch):
                        print(f"  Item {i}: type={type(item)}, shape={item.shape if hasattr(item, 'shape') else 'N/A'}")
            images = batch[0]
            targets = batch[1] if len(batch) > 1 else None
            images = images.to(self.device)
            predictions = self.model(images)
            losses = self.criterion(predictions, targets)
            loss = losses['total']
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
            total_loss += loss.item()
            box_loss += losses['box'].item()
            cls_loss += losses['cls'].item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'box': f'{losses["box"].item():.4f}',
                'cls': f'{losses["cls"].item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        avg_loss = total_loss / len(self.train_loader)
        avg_box = box_loss / len(self.train_loader)
        avg_cls = cls_loss / len(self.train_loader)
        return avg_loss, avg_box, avg_cls
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        for images, targets in tqdm(self.val_loader, desc='Validating'):
            images = images.to(self.device)
            predictions = self.model(images)
            losses = self.criterion(predictions, targets)
            total_loss += losses['total'].item()
        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        return avg_loss
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
        }
        torch.save(checkpoint, self.save_dir / 'last.pt')
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pt')
            print(f"✓ Saved best model with loss: {self.best_loss:.4f}")
    def train(self):
        print("="*70)
        print("Starting Custom YOLO Training")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Model: Custom YOLO")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Epochs: {self.args.epochs}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Learning rate: {self.args.lr}")
        print("="*70 + "\n")
        for epoch in range(self.args.epochs):
            train_loss, box_loss, cls_loss = self.train_one_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step()
            print(f"\nEpoch {epoch+1}/{self.args.epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} (box: {box_loss:.4f}, cls: {cls_loss:.4f})")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            self.save_checkpoint(epoch, is_best)
        print("\n" + "="*70)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_loss:.4f}")
        print(f"Models saved to: {self.save_dir}")
        print("="*70)
def parse_args():
    parser = argparse.ArgumentParser(description='Custom YOLO Training')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--num-classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--save-dir', type=str, default='../runs/custom-yolo', help='Save directory')
    parser.add_argument('--pretrained', type=str, default='yolov8n.pt', help='Pretrained weights')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    return parser.parse_args()
def main():
    args = parse_args()
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    data_root = Path(data_config['path'])
    train_img_dir = data_root / data_config['train']
    val_img_dir = data_root / data_config['val']
    train_label_dir = train_img_dir.parent / 'labels'
    val_label_dir = val_img_dir.parent / 'labels'
    train_dataset = YOLODataset(
        train_img_dir, 
        train_label_dir,
        img_size=args.img_size,
        augment=True
    )
    val_dataset = YOLODataset(
        val_img_dir,
        val_label_dir,
        img_size=args.img_size,
        augment=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    model = CustomYOLO(num_classes=args.num_classes)
    if args.pretrained:
        try:
            print(f"Loading pretrained weights from {args.pretrained}...")
            model.load_pretrained_weights(args.pretrained)
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
            print("Starting from random initialization")
    trainer = YOLOTrainer(model, train_loader, val_loader, args)
    trainer.train()
if __name__ == "__main__":
    main()