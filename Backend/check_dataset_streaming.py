from datasets import load_dataset
from collections import Counter

print("Loading dataset metadata (streaming mode)...")
dataset = load_dataset("BrandonFors/Plant-Diseases-PlantVillage-Dataset", streaming=True)

for split in ['train']:
    print(f"\nAnalyzing split '{split}'...")
    counter = Counter()
    count = 0
    # Analyze first 5000 samples for a "fair amount" check without downloading everything
    for i, example in enumerate(dataset[split]):
        counter[example['label']] += 1
        count += 1
        if i >= 4999:
            break
    
    print(f"Sampled {count} samples.")
    print(f"Found {len(counter)} distinct classes in sample.")
    print("Class distribution in sample (top 10):")
    for label_idx, freq in counter.most_common(10):
        print(f"Label {label_idx}: {freq}")

# Note: We can't easily get total length without downloading in some datasets, 
# but we can get features.
print("\nDataset Features:")
# To get features in streaming, we can use the info if available or take one sample
# For this dataset, let's just peek at one
# demo_sample = next(iter(dataset['train']))
# print(f"Keys: {demo_sample.keys()}")
