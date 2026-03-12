import torch
import torch.nn as nn
import time
import mlflow
import mlflow.pytorch
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import argparse
from balance_dataset import get_balanced_indices

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(model_id, dataset, training_args_dict, run_name, train_ds, val_ds):
    # Load Model & Processor
    print(f"\n[{run_name}] Loading Model: {model_id}...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    
    # Get labels from dataset
    labels = dataset['train'].features['label'].names
    id2label = {str(i): label for i, label in enumerate(labels)}
    label2id = {label: str(i) for i, label in enumerate(labels)}
    num_labels = len(labels)

    model = AutoModelForImageClassification.from_pretrained(
        model_id, 
        num_labels=num_labels, 
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    # Preprocessing
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    size = processor.size.get("shortest_edge", 224) if isinstance(processor.size, dict) else processor.size
    
    _transforms = Compose([
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ])

    def apply_transforms(examples):
        examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples

    # Prepare splits
    # train_ds and val_ds are already selected and transformed
    
    # MLflow tracking

    # MLflow tracking
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_id": model_id,
            "architecture": run_name,
            "epochs": training_args_dict['num_train_epochs'],
            "learning_rate": training_args_dict['learning_rate'],
            "batch_size": training_args_dict['per_device_train_batch_size'],
            "train_samples": len(train_ds)
        })

        training_args = TrainingArguments(
            output_dir=f"./results_{run_name}",
            **training_args_dict
        )

        start_time = time.time()
        
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=DefaultDataCollator(),
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )

        print(f"[{run_name}] Starting training...")
        train_results = trainer.train()
        
        training_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", training_time)
        
        print(f"[{run_name}] Evaluating...")
        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)
        
        # Log model as artifact
        # mlflow.pytorch.log_model(model, "model") # This can be very large
        print(f"[{run_name}] Training complete. Metrics: {metrics}")


if __name__ == "__main__":
    mlflow.set_experiment("AgriPlant-Disease-Comparison")
    
    print("Loading full dataset...")
    dataset = load_dataset("BrandonFors/Plant-Diseases-PlantVillage-Dataset")
    
    common_args = {
        "num_train_epochs": 1,
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "logging_steps": 50,
        "eval_strategy": "steps",
        "eval_steps": 100,
        "save_strategy": "no",
        "remove_unused_columns": False,
    }

    models_to_train = {
        "ResNet50": "mesabo/agri-plant-disease-resnet50",
        "MobileNetV2": "google/mobilenet_v2_1.0_224",
        "EfficientNetB0": "google/efficientnet-b0"
    }

    # Get balanced indices
    # We'll take 150 per class and use 100 for training, 50 for validation
    print("Sampling balanced dataset...")
    all_indices = get_balanced_indices(samples_per_class=150)
    
    # Split indices (simple shuffle and split)
    np.random.seed(42)
    np.random.shuffle(all_indices)
    
    num_train = int(len(all_indices) * 0.7)
    train_indices = all_indices[:num_train]
    val_indices = all_indices[num_train:]
    
    train_ds_sampled = dataset["train"].select(train_indices)
    val_ds_sampled = dataset["train"].select(val_indices)

    for name, m_id in models_to_train.items():
        try:
            train_model(m_id, dataset, common_args, name, train_ds_sampled, val_ds_sampled)
        except Exception as e:
            print(f"Error training {name}: {e}")
