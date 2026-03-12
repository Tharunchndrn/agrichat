from datasets import load_dataset_builder

builder = load_dataset_builder("BrandonFors/Plant-Diseases-PlantVillage-Dataset")
info = builder.info
print(f"Dataset Description: {info.description}")
print(f"Features: {info.features}")
if 'label' in info.features:
    names = info.features['label'].names
    print(f"Number of classes: {len(names)}")
    print(f"Class names: {names}")
else:
    print("Label feature not found in info.")
