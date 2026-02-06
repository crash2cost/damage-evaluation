"""
Create multi-class YOLO dataset with car part labels
Maps CSV classes to YOLO class IDs
"""
import pandas as pd
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image

# Define class mapping
CLASS_MAPPING = {
    'unknown': 0,
    'door_dent': 1,
    'bumper_scratch': 2,
    'door_scratch': 3,
    'glass_shatter': 4,
    'tail_lamp': 5,
    'head_lamp': 6,
    'bumper_dent': 7
}

def create_multiclass_dataset(csv_path, output_dir, test_size=0.2):
    """
    Create YOLO dataset with multi-class labels from CSV
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Total images in CSV: {len(df)}")
    print(f"\nClass distribution:")
    print(df['classes'].value_counts())
    
    # Create output directories
    output_path = Path(output_dir)
    for split in ['train', 'val']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Get image directory
    base_dir = Path(csv_path).parent
    
    # Filter valid images and classes
    valid_data = []
    for idx, row in df.iterrows():
        img_path = base_dir / row['image']
        if img_path.exists() and row['classes'] in CLASS_MAPPING:
            try:
                # Verify image can be opened
                with Image.open(img_path) as img:
                    width, height = img.size
                valid_data.append({
                    'image_path': img_path,
                    'class': row['classes'],
                    'class_id': CLASS_MAPPING[row['classes']],
                    'width': width,
                    'height': height
                })
            except Exception as e:
                print(f"Skipping {img_path}: {e}")
    
    print(f"\nValid images: {len(valid_data)}")
    
    # Split into train/val
    train_data, val_data = train_test_split(
        valid_data, 
        test_size=test_size, 
        random_state=42,
        stratify=[d['class'] for d in valid_data]  # Stratified split
    )
    
    print(f"Training images: {len(train_data)}")
    print(f"Validation images: {len(val_data)}")
    
    # Process splits
    for split_name, split_data in [('train', train_data), ('val', val_data)]:
        print(f"\nProcessing {split_name} split...")
        
        for idx, data in enumerate(split_data):
            # Copy image
            img_name = f"auto_{idx:06d}.jpg"
            img_dst = output_path / split_name / 'images' / img_name
            shutil.copy2(data['image_path'], img_dst)
            
            # Create label file with full image bounding box
            # Format: class_id x_center y_center width height (normalized 0-1)
            label_dst = output_path / split_name / 'labels' / f"auto_{idx:06d}.txt"
            with open(label_dst, 'w') as f:
                # Full image bounding box (entire image is damaged area)
                f.write(f"{data['class_id']} 0.5 0.5 1.0 1.0\n")
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(split_data)} images...")
    
    # Create data.yaml
    class_names = [name for name, _ in sorted(CLASS_MAPPING.items(), key=lambda x: x[1])]
    yaml_content = f"""# Car Damage Multi-Class Dataset
path: {output_path.absolute()}
train: train/images
val: val/images

# Classes
nc: {len(CLASS_MAPPING)}
names: {class_names}
"""
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n Dataset created successfully!")
    print(f"  Output: {output_path}")
    print(f"  Config: {yaml_path}")
    print(f"\nClass mapping:")
    for class_name, class_id in sorted(CLASS_MAPPING.items(), key=lambda x: x[1]):
        count_train = sum(1 for d in train_data if d['class'] == class_name)
        count_val = sum(1 for d in val_data if d['class'] == class_name)
        print(f"  {class_id}: {class_name:20s} (train: {count_train:3d}, val: {count_val:3d})")

if __name__ == '__main__':
    csv_path = Path('../archive/data.csv')
    output_dir = Path('dataset-multiclass')
    
    create_multiclass_dataset(csv_path, output_dir, test_size=0.2)
