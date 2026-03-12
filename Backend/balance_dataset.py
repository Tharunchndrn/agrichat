from datasets import load_dataset
import pandas as pd
import numpy as np

def get_balanced_indices(dataset_name="BrandonFors/Plant-Diseases-PlantVillage-Dataset", samples_per_class=100, seed=42):
    print(f"Loading dataset {dataset_name} metadata...")
    dataset = load_dataset(dataset_name, split="train")
    
    # Efficiently get labels
    df = pd.DataFrame(dataset.select_columns(['label']))
    
    balanced_indices = []
    
    # Get all unique labels
    labels = df['label'].unique()
    print(f"Found {len(labels)} classes.")
    
    np.random.seed(seed)
    
    for label in labels:
        label_indices = df[df['label'] == label].index.tolist()
        # Sample if we have enough, otherwise take all
        n_to_sample = min(len(label_indices), samples_per_class)
        sampled = np.random.choice(label_indices, n_to_sample, replace=False)
        balanced_indices.extend(sampled.tolist())
        print(f"Class {label}: Selected {len(sampled)}/{len(label_indices)} samples")
    
    print(f"Total balanced samples selected: {len(balanced_indices)}")
    return balanced_indices

if __name__ == "__main__":
    indices = get_balanced_indices(samples_per_class=100)
    # Just a quick check
    print(f"First 10 indices: {indices[:10]}")
