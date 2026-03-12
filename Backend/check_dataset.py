from datasets import load_dataset
import pandas as pd

print("Loading dataset metadata...")
dataset = load_dataset("BrandonFors/Plant-Diseases-PlantVillage-Dataset")

for split in dataset.keys():
    print(f"\nSplit '{split}': {len(dataset[split])} samples")
    df = pd.DataFrame(dataset[split].select_columns(['label']))
    print(f"Number of classes: {df['label'].nunique()}")
    print("Class distribution (top 10):")
    print(df['label'].value_counts().head(10))

# Get class names
labels = dataset['train'].features['label'].names
print(f"\nTotal Class Names: {len(labels)}")
print(f"Sample labels: {labels[:5]}")
