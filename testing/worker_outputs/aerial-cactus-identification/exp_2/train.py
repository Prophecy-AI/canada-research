#!/usr/bin/env python3
"""
Aerial Cactus Identification - ResNet50 with fastai
Binary classification using fastai vision with fit_one_cycle
"""

# CRITICAL: Import ALL necessary metrics at the TOP
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from fastai.vision.all import *
import warnings
warnings.filterwarnings('ignore')

# Experiment configuration
spec = {
    "model": "resnet50",
    "strategy": "fastai_vision",
    "hyperparameters": {
        "device": "cuda",
        "epochs": 5,
        "size": 128,
        "lr": 0.03,
        "split_pct": 0.01,
        "batch_size": 64
    }
}

print("=" * 60)
print("Aerial Cactus Identification - ResNet50 Training")
print("=" * 60)

# Extract hyperparameters
epochs = spec['hyperparameters']['epochs']
size = spec['hyperparameters']['size']
lr = spec['hyperparameters']['lr']
split_pct = spec['hyperparameters']['split_pct']
batch_size = spec['hyperparameters']['batch_size']

print(f"\nHyperparameters:")
print(f"  Model: {spec['model']}")
print(f"  Epochs: {epochs}")
print(f"  Image size: {size}")
print(f"  Learning rate: {lr}")
print(f"  Validation split: {split_pct}")
print(f"  Batch size: {batch_size}")

# Setup paths
data_path = Path('/home/data')
work_path = Path('.')

# Extract training data
print("\n" + "=" * 60)
print("Extracting training data...")
print("=" * 60)

import zipfile
with zipfile.ZipFile(data_path / 'train.zip', 'r') as zip_ref:
    zip_ref.extractall(work_path)

print("✓ Training images extracted")

# Load training labels
train_df = pd.read_csv(data_path / 'train.csv', dtype={'id': str})
print(f"\nTraining data shape: {train_df.shape}")
print(f"Columns: {train_df.columns.tolist()}")
print(f"\nFirst few rows:")
print(train_df.head())

# Check class distribution
print(f"\nClass distribution:")
print(train_df['has_cactus'].value_counts().sort_index())
print(f"Class balance: {train_df['has_cactus'].value_counts(normalize=True).sort_index()}")

# CRITICAL: Convert labels to strings for binary classification
# This prevents fastai from treating it as regression
train_df['label'] = train_df['has_cactus'].astype(str)
print(f"\nConverted labels to strings: {train_df['label'].unique()}")

# Add full path to images
train_df['filepath'] = train_df['id'].apply(lambda x: str(work_path / 'train' / x))

# Verify images exist
sample_exists = Path(train_df['filepath'].iloc[0]).exists()
print(f"\nSample image exists: {sample_exists}")
if sample_exists:
    print(f"Sample path: {train_df['filepath'].iloc[0]}")

# Create DataLoaders using from_df
print("\n" + "=" * 60)
print("Creating DataLoaders...")
print("=" * 60)

dls = ImageDataLoaders.from_df(
    train_df,
    path=work_path,
    fn_col='filepath',
    label_col='label',
    valid_pct=split_pct,
    seed=42,
    item_tfms=Resize(size),
    batch_tfms=aug_transforms(
        size=size,
        min_scale=0.75,
        do_flip=True,
        flip_vert=True,
        max_rotate=10.0,
        max_lighting=0.2,
        max_warp=0.2
    ),
    bs=batch_size,
    num_workers=0  # Avoid multiprocessing issues
)

print(f"✓ DataLoaders created")
print(f"  Training batches: {len(dls.train)}")
print(f"  Validation batches: {len(dls.valid)}")
print(f"  Classes: {dls.vocab}")

# Show a batch
print("\nSample batch shape:")
xb, yb = dls.one_batch()
print(f"  Images: {xb.shape}")
print(f"  Labels: {yb.shape}")

# Create learner with ResNet50
print("\n" + "=" * 60)
print("Creating ResNet50 learner...")
print("=" * 60)

learn = vision_learner(
    dls,
    resnet50,
    metrics=[accuracy, error_rate]
)

print(f"✓ Learner created with {spec['model']}")
print(f"  Device: {learn.dls.device}")

# Train with fit_one_cycle
print("\n" + "=" * 60)
print(f"Training for {epochs} epochs with fit_one_cycle...")
print("=" * 60)

learn.fit_one_cycle(epochs, lr)

print("\n✓ Training completed!")

# Calculate VALIDATION_SCORE using competition metric (AUC-ROC)
print("\n" + "=" * 60)
print("Calculating validation metrics...")
print("=" * 60)

# Get validation predictions
val_preds, val_targets = learn.get_preds(dl=learn.dls.valid)

# Convert to numpy immediately
val_probs = val_preds.numpy()
y_val = val_targets.numpy()

# Clip probabilities to prevent numerical issues
val_probs = np.clip(val_probs, 1e-7, 1 - 1e-7)

# Calculate competition metric: AUC-ROC (HIGHER is better)
# For binary classification, use probabilities of positive class (class 1)
val_metric = roc_auc_score(y_val, val_probs[:, 1])

print(f"VALIDATION_SCORE: {val_metric:.6f}")

# Also calculate accuracy for reference
val_accuracy = accuracy_score(y_val, val_probs.argmax(axis=1))
print(f"Validation Accuracy: {val_accuracy:.6f}")

# Save the model
print("\n" + "=" * 60)
print("Saving model...")
print("=" * 60)

learn.export('model.pkl')
print("✓ Model saved to model.pkl")

print("\n" + "=" * 60)
print("Training complete!")
print("=" * 60)
