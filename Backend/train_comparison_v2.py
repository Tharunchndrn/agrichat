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
import os

# Custom function to get balanced indices
def create_balanced_splits(ds, num_train_per_class=50, num_val_per_class=10):
    labels = np.array(ds['label'])
    train_indices = []
    val_indices = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        label_idxs = np.where(labels == label)[0]
        np.random.shuffle(label_idxs)
        
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

def train_and_log_model(model_name, model_id, train_ds, val_ds, labels, id2label, label2id):
    print(f"\n--- Training {model_name} ({model_id}) ---")
    
    # Load Processor
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"Warning: Falling back to resnet processor for {model_id}: {e}")
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        
    normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
    size = processor.size.get("shortest_edge", 224) if isinstance(processor.size, dict) else processor.size
    
    _train_transforms = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize])
    _val_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
    
    def apply_train_transforms(examples):
        examples["pixel_values"] = [_train_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples
        
    def apply_val_transforms(examples):
        examples["pixel_values"] = [_val_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples
        
    train_ds.set_transform(apply_train_transforms)
    val_ds.set_transform(apply_val_transforms)

    # Load Model
    model = AutoModelForImageClassification.from_pretrained(
        model_id, num_labels=len(labels), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir=f"./results_{model_name}",
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        num_train_epochs=1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="mlflow",
        push_to_hub=False,
        save_total_limit=1
    )

    with mlflow.start_run(run_name=f"Benchmark_50s_1e_{model_name}"):
        mlflow.log_param("architecture", model_name)
        mlflow.log_param("model_id", model_id)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=DefaultDataCollator(),
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=processor,
            compute_metrics=compute_metrics
        )
        
        trainer.train()
        eval_results = trainer.evaluate()
        print(f"Results for {model_name}: {eval_results}")
        
        # Log essential metrics explicitly to make comparison easy
        mlflow.log_metric("final_accuracy", eval_results["eval_accuracy"])
        mlflow.log_metric("final_f1", eval_results["eval_f1"])
        
        return eval_results["eval_accuracy"], mlflow.active_run().info.run_id

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    
    mlflow.set_experiment("AgriPlant-Standardized-Comparison")
    
    # 1. Load Dataset
    print("Loading dataset...")
    dataset = load_dataset("BrandonFors/Plant-Diseases-PlantVillage-Dataset")
    
    print("Creating 50 samples/class splits...")
    train_ds, val_ds = create_balanced_splits(dataset["train"], num_train_per_class=50, num_val_per_class=10)
    
    labels = dataset['train'].features['label'].names
    id2label = {str(i): l for i, l in enumerate(labels)}
    label2id = {l: str(i) for i, l in enumerate(labels)}
    
    models_to_test = [
        ("ResNet50", "microsoft/resnet-50"),
        ("MobileNetV2", "google/mobilenet_v2_1.0_224"),
        ("EfficientNetB0", "google/efficientnet-b0")
    ]
    
    results = []
    
    for name, m_id in models_to_test:
        try:
            acc, run_id = train_and_log_model(name, m_id, train_ds, val_ds, labels, id2label, label2id)
            results.append((name, acc, run_id))
        except Exception as e:
            print(f"Error training {name}: {e}")

    # 2. Find and register the best model
    if results:
        best_model_run = max(results, key=lambda x: x[1])
        best_name, best_acc, best_run_id = best_model_run
        print(f"\n🏆 Best Model: {best_name} with Accuracy: {best_acc:.4f}")
        
        # Determine the checkpoint path for the best model
        best_checkpoint_dir = f"./results_{best_name}"
        # Find the latest/best checkpoint folder in that directory
        checkpoints = [os.path.join(best_checkpoint_dir, d) for d in os.listdir(best_checkpoint_dir) if d.startswith("checkpoint")]
        if checkpoints:
            best_checkpoint = max(checkpoints, key=os.path.getmtime)
            print(f"Deploying best model from {best_checkpoint} to plant_disease_model_final...")
            
            # Use shutil to copy the model files
            import shutil
            target_dir = "plant_disease_model_final"
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            shutil.copytree(best_checkpoint, target_dir)
            print(f"✅ Best model ({best_name}) is now saved in {target_dir}")

        print(f"Registering {best_name} in MLflow Model Registry (AgriPlantDiseaseClassifier)...")
        model_uri = f"runs:/{best_run_id}/model"
        
        try:
            result = mlflow.register_model(model_uri, "AgriPlantDiseaseClassifier")
            print(f"Successfully registered model! Version: {result.version}")
        except Exception as e:
            print(f"Note: MLflow registry might require a backend store (database) or specific config: {e}")
