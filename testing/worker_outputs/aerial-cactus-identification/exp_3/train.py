#!/usr/bin/env python3
"""
EfficientNet-B2 training for Aerial Cactus Identification
Strategy: fastai_vision
Model: EfficientNet-B2 with compound scaling
"""

# CRITICAL: Import ALL necessary metrics at the TOP
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import zipfile
import os

# fastai imports
from fastai.vision.all import *

print("=" * 80)
print("AERIAL CACTUS IDENTIFICATION - EfficientNet-B2 Training")
print("=" * 80)

# Hyperparameters from spec
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 7
SIZE = 160
LR = 0.02
SPLIT_PCT = 0.01
BATCH_SIZE = 48
MODEL_NAME = 'efficientnet_b2'

print(f"\nConfiguration:")
print(f"  Device: {DEVICE}")
print(f"  Model: {MODEL_NAME}")
print(f"  Image size: {SIZE}x{SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning rate: {LR}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Validation split: {SPLIT_PCT}")

# Setup paths
data_dir = Path('/home/data')
work_dir = Path('.')

# Extract training data
print("\n" + "=" * 80)
print("EXTRACTING DATA")
print("=" * 80)

train_zip = data_dir / 'train.zip'
if train_zip.exists():
    print(f"Extracting {train_zip}...")
    with zipfile.ZipFile(train_zip, 'r') as zip_ref:
        zip_ref.extractall(work_dir)
    print("✓ Training images extracted")
else:
    print(f"ERROR: {train_zip} not found!")
    exit(1)

# Load training CSV
train_csv = data_dir / 'train.csv'
if not train_csv.exists():
    print(f"ERROR: {train_csv} not found!")
    exit(1)

df = pd.read_csv(train_csv, dtype={'id': str})
print(f"\n✓ Loaded train.csv: {len(df)} samples")
print(f"  Columns: {list(df.columns)}")
print(f"  Class distribution:")
print(df['has_cactus'].value_counts().sort_index())

# CRITICAL: Convert labels to strings for fastai binary classification
# This prevents fastai from treating 0/1 as regression
df['has_cactus'] = df['has_cactus'].astype(str)
print(f"\n✓ Converted labels to strings: {df['has_cactus'].unique()}")

# Verify image directory
train_img_dir = work_dir / 'train'
if not train_img_dir.exists():
    print(f"ERROR: {train_img_dir} not found after extraction!")
    exit(1)

num_images = len(list(train_img_dir.glob('*.jpg')))
print(f"✓ Found {num_images} images in {train_img_dir}")

# Add full path to dataframe
df['path'] = df['id'].apply(lambda x: str(train_img_dir / x))

# Verify a few images exist
missing = 0
for idx in range(min(10, len(df))):
    if not Path(df.iloc[idx]['path']).exists():
        missing += 1
        print(f"  WARNING: Missing {df.iloc[idx]['path']}")

if missing > 0:
    print(f"  WARNING: {missing}/10 sample images missing!")

print("\n" + "=" * 80)
print("CREATING DATALOADERS")
print("=" * 80)

# Create DataBlock with augmentation
# Using larger image size (160) for EfficientNet-B2 to leverage compound scaling
dls = ImageDataLoaders.from_df(
    df,
    path=work_dir,
    fn_col='path',
    label_col='has_cactus',
    valid_pct=SPLIT_PCT,
    seed=42,
    item_tfms=Resize(SIZE),
    batch_tfms=aug_transforms(
        size=SIZE,
        min_scale=0.75,
        do_flip=True,
        flip_vert=True,
        max_rotate=10.0,
        max_lighting=0.2,
        max_warp=0.2
    ),
    bs=BATCH_SIZE,
    num_workers=0  # Avoid multiprocessing issues
)

print(f"✓ DataLoaders created")
print(f"  Training batches: {len(dls.train)}")
print(f"  Validation batches: {len(dls.valid)}")
print(f"  Training samples: {len(dls.train_ds)}")
print(f"  Validation samples: {len(dls.valid_ds)}")
print(f"  Classes: {dls.vocab}")

# Show a batch to verify
print("\n✓ Sample batch shape:", dls.one_batch()[0].shape)

print("\n" + "=" * 80)
print("CREATING LEARNER")
print("=" * 80)

# Create vision learner with EfficientNet-B2
# EfficientNet-B2 uses compound scaling (depth, width, resolution)
# Optimized for efficiency and accuracy tradeoff
learn = vision_learner(
    dls,
    MODEL_NAME,
    metrics=accuracy,
    pretrained=True
)

print(f"✓ Created vision_learner with {MODEL_NAME}")
print(f"  Pretrained: True")
print(f"  Metric: accuracy")

print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

# Train with one-cycle policy
# Using lower LR (0.02) for stability with larger images
print(f"\nStarting training: {EPOCHS} epochs, lr={LR}")
learn.fit_one_cycle(EPOCHS, LR)

print("\n✓ Training completed!")

print("\n" + "=" * 80)
print("VALIDATION EVALUATION")
print("=" * 80)

# Get validation predictions
print("Calculating validation predictions...")
val_preds, val_targets = learn.get_preds(dl=learn.dls.valid)

# Convert to numpy IMMEDIATELY
val_probs = val_preds.numpy()
y_val = val_targets.numpy()

print(f"✓ Validation predictions shape: {val_probs.shape}")
print(f"✓ Validation targets shape: {y_val.shape}")

# Clip probabilities to prevent log(0) errors
val_probs = np.clip(val_probs, 1e-7, 1 - 1e-7)

# CRITICAL: Calculate THE COMPETITION METRIC from EDA
# EDA states: "Evaluation Metric: AUC-ROC (HIGHER is better)"
# For binary classification, use probabilities of positive class (index 1)
val_metric = roc_auc_score(y_val, val_probs[:, 1])

print(f"\n{'=' * 80}")
print(f"VALIDATION_SCORE: {val_metric:.6f}")
print(f"{'=' * 80}")

# Additional metrics for reference (but don't confuse with VALIDATION_SCORE)
val_acc = accuracy_score(y_val, val_probs.argmax(axis=1))
print(f"\nAdditional metrics (for reference):")
print(f"  Validation Accuracy: {val_acc:.6f}")
print(f"  Validation Log Loss: {log_loss(y_val, val_probs):.6f}")

print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)

# Save the learner
learn.export('model.pkl')
print("✓ Model saved to model.pkl")

# Also save just the model weights for flexibility
torch.save(learn.model.state_dict(), 'model_weights.pth')
print("✓ Model weights saved to model_weights.pth")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"\nFinal Results:")
print(f"  Model: {MODEL_NAME}")
print(f"  Image size: {SIZE}x{SIZE}")
print(f"  Epochs trained: {EPOCHS}")
print(f"  Validation AUC-ROC: {val_metric:.6f}")
print(f"  Validation Accuracy: {val_acc:.6f}")
print("\n✓ Ready for inference!")
