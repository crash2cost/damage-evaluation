from datasets import load_dataset
import traceback

# Promising datasets that might have bounding boxes
candidates = [
    "DrBimmer/comprehensive-car-damage",
    "xlyashuk/car-damage-big",
    "Abijith/car-damage-segmentation-small",
    "chittaranjankhatua/car_damage_pub"
]

for dataset_name in candidates:
    print(f"\n{'='*80}")
    print(f"Checking: {dataset_name}")
    print('='*80)
    try:
        # Load dataset
        ds = load_dataset(dataset_name, split='train', streaming=True)
        
        # Get first example
        first_example = next(iter(ds))
        
        print(f"Features: {list(first_example.keys())}")
        print(f"\nSample data:")
        for key, value in first_example.items():
            if key != 'image':  # Don't print image data
                print(f"  {key}: {type(value)} = {str(value)[:200]}")
        
        # Check if it has bounding boxes
        if 'bbox' in first_example or 'boxes' in first_example or 'objects' in first_example:
            print("\n✅ HAS BOUNDING BOXES!")
        else:
            print("\n❌ No bounding boxes found")
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
