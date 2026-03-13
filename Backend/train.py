import torch
import numpy as np
import mlflow
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Custom function to get balanced indices
def create_balanced_splits(ds, num_train_per_class=100, num_val_per_class=20):
    labels = np.array(ds['label'])
    train_indices = []
    val_indices = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        label_idxs = np.where(labels == label)[0]
        # Shuffle indices
        np.random.shuffle(label_idxs)
        
        # Determine available counts
        n_total = len(label_idxs)
        n_train = min(num_train_per_class, int(0.8 * n_total))
        n_val = min(num_val_per_class, n_total - n_train)
        
        train_indices.extend(label_idxs[:n_train].tolist())
        val_indices.extend(label_idxs[n_train:n_train+n_val].tolist())
        
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    return ds.select(train_indices), ds.select(val_indices)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

if __name__ == "__main__":
    np.random.seed(42)
    
    # 1. Load Dataset
    print("Loading Hugging Face Dataset: BrandonFors/Plant-Diseases-PlantVillage-Dataset...")
    dataset = load_dataset("BrandonFors/Plant-Diseases-PlantVillage-Dataset")
    
    print("Creating balanced splits (Train: 100/class, Val: 10/class)")
    train_ds, val_ds = create_balanced_splits(dataset["train"], num_train_per_class=100, num_val_per_class=10)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    # 2. Load Model & Processor
    MODEL_ID = "google/mobilenet_v2_1.0_224"
    FALLBACK_ID = "microsoft/resnet-50"
    
    print("Loading Processor...")
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    except Exception:
        print(f"Warning: Could not load processor for {MODEL_ID}. Using fallback: {FALLBACK_ID}")
        processor = AutoImageProcessor.from_pretrained(FALLBACK_ID)
        
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    size = processor.size.get("shortest_edge", 224) if isinstance(processor.size, dict) else processor.size
    
    _train_transforms = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize])
    _val_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
    
    def train_transforms(examples):
        examples["pixel_values"] = [_train_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples
        
    def val_transforms(examples):
        examples["pixel_values"] = [_val_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples
        
    train_ds.set_transform(train_transforms)
    val_ds.set_transform(val_transforms)

    # 3. Model setup
    labels = dataset['train'].features['label'].names
    id2label = {str(i): l for i, l in enumerate(labels)}
    label2id = {l: str(i) for i, l in enumerate(labels)}
    
    print("Loading Model...")
    try:
        model = AutoModelForImageClassification.from_pretrained(
            MODEL_ID, num_labels=len(labels), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
        )
    except Exception as e:
        print(f"Falling back to {FALLBACK_ID}: {e}")
        model = AutoModelForImageClassification.from_pretrained(
            FALLBACK_ID, num_labels=len(labels), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
        )

    # 4. Training Arguments
    mlflow.set_experiment("AgriPlant-Focused-Training")
    
    training_args = TrainingArguments(
        output_dir="./plant_disease_model",
        remove_unused_columns=False,
        eval_strategy="epoch",  # Evaluate every epoch
        save_strategy="epoch",  # Save every epoch
        learning_rate=5e-5,
        per_device_train_batch_size=16,  # Larger batch = fewer steps = faster
        gradient_accumulation_steps=1,   # No accumulation needed with large batch
        num_train_epochs=1,              # 1 epoch for fast base system
        warmup_steps=20,
        logging_steps=10,
        load_best_model_at_end=True,     # Load best model
        metric_for_best_model="accuracy",
        report_to="mlflow",              # Track in mlflow
        push_to_hub=False,
        save_total_limit=1               # Save disk space
    )

    # 5. Initialize Trainer
    with mlflow.start_run(run_name="MobileNetV2_Focused_100s"):
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=DefaultDataCollator(),
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=processor,
            compute_metrics=compute_metrics
        )
        
        # 6. Start Training
        print(f"Starting Fine-tuning loop on {len(train_ds)} samples...")
        trainer.train()
        
        print("Evaluating best model...")
        final_metrics = trainer.evaluate()
        print(f"Final Validation Metrics: {final_metrics}")
        
        # Save best model to specific path
        best_model_path = r"c:\Users\Taruni\Desktop\My Projects\Agrichatbot\agrichat\Backend\plant_disease_model_final"
        trainer.save_model(best_model_path)
        processor.save_pretrained(best_model_path)
        print(f"Training Complete! The BEST updated model is saved in '{best_model_path}'")
