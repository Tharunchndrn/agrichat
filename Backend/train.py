import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

# 1. Load Dataset
print("Loading Hugging Face Dataset: BrandonFors/Plant-Diseases-PlantVillage-Dataset...")
dataset = load_dataset("BrandonFors/Plant-Diseases-PlantVillage-Dataset")
# Use a small subset for demonstration if needed, but let's try the full train split
train_ds = dataset["train"].shuffle(seed=42).select(range(100)) # Using 100 samples as requested by user

# 2. Load Model & Processor
MODEL_ID = "mesabo/agri-plant-disease-resnet50"
FALLBACK_ID = "microsoft/resnet-50"

print("Loading Processor...")
try:
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
except Exception:
    print(f"Warning: Could not load processor for {MODEL_ID}. Using fallback: {FALLBACK_ID}")
    processor = AutoImageProcessor.from_pretrained(FALLBACK_ID)

print("Loading Model...")
model = AutoModelForImageClassification.from_pretrained(MODEL_ID)

# 3. Preprocessing
normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
size = (
    processor.size["shortest_edge"]
    if "shortest_edge" in processor.size
    else processor.size["height"]
)

_train_transforms = Compose(
    [
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

def train_transforms(examples):
    examples["pixel_values"] = [_train_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

train_ds.set_transform(train_transforms)

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./plant_disease_model",
    remove_unused_columns=False,
    eval_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    warmup_steps=10,
    logging_steps=10,
    save_strategy="no",
    push_to_hub=False,
)

# 5. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DefaultDataCollator(),
    train_dataset=train_ds,
    processing_class=processor,
)

# 6. Start Training
print("Starting Fine-tuning (demo mode with 100 samples)...")
train_results = trainer.train()
trainer.save_model("./plant_disease_model_final")
print("Training Complete! The updated model is saved in './plant_disease_model_final'")
