from datasets import load_dataset

dataset = load_dataset("harpreetsahota/CarDD")

print(dataset)
print("\nDataset structure:")
print(f"Number of splits: {len(dataset)}")
for split_name in dataset.keys():
    print(f"\n{split_name} split:")
    print(f"  Number of examples: {len(dataset[split_name])}")
    print(f"  Features: {dataset[split_name].features}")
    if len(dataset[split_name]) > 0:
        print(f"  First example keys: {dataset[split_name][0].keys()}")
