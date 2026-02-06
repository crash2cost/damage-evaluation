from datasets import load_dataset_builder
from huggingface_hub import list_datasets

print("Searching for car damage datasets with labels...\n")

# Search for car damage related datasets
keywords = ["car damage", "vehicle damage", "auto damage", "car dent", "car scratch"]
found_datasets = []

for keyword in keywords:
    print(f"Searching for: {keyword}")
    datasets = list(list_datasets(search=keyword, limit=10))
    for ds in datasets:
        if ds.id not in [d.id for d in found_datasets]:
            found_datasets.append(ds)
            print(f"  ✓ {ds.id}")

print(f"\n\nFound {len(found_datasets)} unique datasets:")
for ds in found_datasets:
    print(f"  - {ds.id}")
