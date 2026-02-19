import os
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = ROOT / "archive"
OUTPUT_DIR = ROOT / "regression-model" / "dataset"
df = pd.read_csv(ARCHIVE_DIR / "data.csv")
print(f"Total samples: {len(df)}")
print(f"\nClass distribution:\n{df['classes'].value_counts()}")
df = df[df['classes'] != 'unknown'].reset_index(drop=True)
print(f"\nAfter removing 'unknown': {len(df)} samples")
classes = sorted(df['classes'].unique())
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
df['label'] = df['classes'].map(class_to_idx)
print(f"\nClass mapping:")
for cls, idx in class_to_idx.items():
    print(f"  {idx}: {cls}")
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
print(f"\nSplit sizes:")
print(f"  Train: {len(train_df)}")
print(f"  Val:   {len(val_df)}")
print(f"  Test:  {len(test_df)}")
for split in ['train', 'val', 'test']:
    for cls in classes:
        (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)
def copy_files(df_split, split_name):
    print(f"\nCopying {split_name} files...")
    for _, row in tqdm(df_split.iterrows(), total=len(df_split)):
        src = ARCHIVE_DIR / row['image']
        dst = OUTPUT_DIR / split_name / row['classes'] / Path(row['image']).name
        shutil.copy2(src, dst)
copy_files(train_df, 'train')
copy_files(val_df, 'val')
copy_files(test_df, 'test')
train_df.to_csv(OUTPUT_DIR / 'train.csv', index=False)
val_df.to_csv(OUTPUT_DIR / 'val.csv', index=False)
test_df.to_csv(OUTPUT_DIR / 'test.csv', index=False)
with open(OUTPUT_DIR / 'classes.txt', 'w') as f:
    for cls in classes:
        f.write(f"{cls}\n")
print("\n Dataset preparation complete!")
print(f"   Output directory: {OUTPUT_DIR}")