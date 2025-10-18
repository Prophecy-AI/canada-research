#!/usr/bin/env python3
"""
Aerial Cactus Identification - DenseNet161 with fastai
Binary classification of 32x32 aerial images
Competition Metric: AUC-ROC (HIGHER is better)
"""

# CRITICAL: Import ALL necessary metrics at the top
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import zipfile

# fastai imports
from fastai.vision.all import *

print("=" * 60)
print("Aerial Cactus Identification - DenseNet161 Training")
print("=" * 60)

# Hyperparameters from spec
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 5
SIZE = 128
LR = 0.03
SPLIT_PCT = 0.01  # 99/1 train/val split to maximize training data
BATCH_SIZE = 64
MODEL_NAME = 'densenet161'

print(f"\nConfiguration:")
print(f"  Device: {DEVICE}")
print(f"  Model: {MODEL_NAME}")
print(f"  Epochs: {EPOCHS}")
print(f"  Image Size: {SIZE}")
print(f"  Learning Rate: {LR}")
print(f"  Train/Val Split: {100*(1-SPLIT_PCT):.0f}/{100*SPLIT_PCT:.0f}")
print(f"  Batch Size: {BATCH_SIZE}")

# Setup paths
data_dir = Path('/home/data')
workspace = Path('.')

# Extract training data
print("\n" + "=" * 60)
print("Extracting data...")
print("=" * 60)

train_zip = data_dir / 'train.zip'
test_zip = data_dir / 'test.zip'

if train_zip.exists():
    print(f"Extracting {train_zip}...")
    with zipfile.ZipFile(train_zip, 'r') as zip_ref:
        zip_ref.extractall(workspace)
    print(f"  ✓ Extracted training images")
else:
    print(f"  ⚠️  {train_zip} not found, assuming data already extracted")

if test_zip.exists():
    print(f"Extracting {test_zip}...")
    with zipfile.ZipFile(test_zip, 'r') as zip_ref:
        zip_ref.extractall(workspace)
    print(f"  ✓ Extracted test images")
else:
    print(f"  ⚠️  {test_zip} not found, assuming data already extracted")

# Load CSV files
train_csv_path = data_dir / 'train.csv'
test_csv_path = data_dir / 'sample_submission.csv'

print(f"\nLoading CSV files...")
train_df = pd.read_csv(train_csv_path, dtype={'id': str})
test_df = pd.read_csv(test_csv_path, dtype={'id': str})

print(f"  Training samples: {len(train_df)}")
print(f"  Test samples: {len(test_df)}")
print(f"  Class distribution:")
print(f"    Class 0 (no cactus): {(train_df['has_cactus'] == 0).sum()} ({100*(train_df['has_cactus'] == 0).sum()/len(train_df):.1f}%)")
print(f"    Class 1 (has cactus): {(train_df['has_cactus'] == 1).sum()} ({100*(train_df['has_cactus'] == 1).sum()/len(train_df):.1f}%)")

# CRITICAL: Convert labels to strings for fastai binary classification
train_df['label'] = train_df['has_cactus'].astype(str)

# Verify image paths
train_img_dir = workspace / 'train'
test_img_dir = workspace / 'test'

print(f"\nVerifying image directories...")
print(f"  Train images: {train_img_dir} - exists: {train_img_dir.exists()}")
print(f"  Test images: {test_img_dir} - exists: {test_img_dir.exists()}")

if train_img_dir.exists():
    sample_images = list(train_img_dir.glob('*.jpg'))[:3]
    print(f"  Sample train images: {[img.name for img in sample_images]}")

# Create DataLoaders with heavy augmentation
print("\n" + "=" * 60)
print("Creating DataLoaders...")
print("=" * 60)

# Define function to get image path from dataframe row
def get_image_path(row):
    return train_img_dir / row['id']

# Create DataBlock with heavy augmentation
dls = ImageDataLoaders.from_df(
    train_df,
    path=workspace,
    fn_col='id',
    label_col='label',
    folder='train',
    valid_pct=SPLIT_PCT,
    seed=42,
    item_tfms=Resize(SIZE),
    batch_tfms=aug_transforms(
        size=SIZE,
        min_scale=0.75,
        do_flip=True,
        flip_vert=True,
        max_rotate=20.0,
        max_lighting=0.3,
        max_warp=0.2,
        p_affine=0.75,
        p_lighting=0.75
    ),
    bs=BATCH_SIZE
)

print(f"  ✓ DataLoaders created")
print(f"  Training batches: {len(dls.train)}")
print(f"  Validation batches: {len(dls.valid)}")
print(f"  Classes: {dls.vocab}")

# Show a batch to verify
print("\nShowing sample batch...")
dls.show_batch(max_n=4)

# Create learner with DenseNet161
print("\n" + "=" * 60)
print("Creating learner...")
print("=" * 60)

learn = vision_learner(
    dls,
    densenet161,
    metrics=[accuracy, error_rate],
    pretrained=True
)

print(f"  ✓ Learner created with {MODEL_NAME}")
print(f"  Model architecture: DenseNet161 (pretrained on ImageNet)")

# Train with fit_one_cycle
print("\n" + "=" * 60)
print("Training...")
print("=" * 60)

learn.fit_one_cycle(EPOCHS, LR)

print("\n" + "=" * 60)
print("Training completed!")
print("=" * 60)

# Calculate VALIDATION_SCORE using competition metric (AUC-ROC)
print("\n" + "=" * 60)
print("Calculating validation score...")
print("=" * 60)

# Get validation predictions
val_preds, val_targets = learn.get_preds(dl=learn.dls.valid)

# Convert to numpy immediately
val_probs = val_preds.numpy()
y_val = val_targets.numpy()

# Clip probabilities to prevent numerical issues
val_probs = np.clip(val_probs, 1e-7, 1 - 1e-7)

# Calculate AUC-ROC (competition metric - HIGHER is better)
# For binary classification, use probabilities of positive class (class 1)
val_metric = roc_auc_score(y_val, val_probs[:, 1])

print(f"  Competition Metric: AUC-ROC")
print(f"  Validation AUC-ROC: {val_metric:.6f}")
print(f"\nVALIDATION_SCORE: {val_metric:.6f}")

# Save model
print("\n" + "=" * 60)
print("Saving model...")
print("=" * 60)

learn.export('model.pkl')
print(f"  ✓ Model saved to model.pkl")

# Generate predictions on test set
print("\n" + "=" * 60)
print("Generating test predictions...")
print("=" * 60)

# Get test image files
test_files = sorted(list(test_img_dir.glob('*.jpg')))
print(f"  Found {len(test_files)} test images")

# Create test dataloader
test_dl = learn.dls.test_dl(test_files)
print(f"  Test batches: {len(test_dl)}")

# Get predictions
test_preds, _ = learn.get_preds(dl=test_dl)
test_probs = test_preds.numpy()

# Clip probabilities
test_probs = np.clip(test_probs, 1e-7, 1 - 1e-7)

# Create submission dataframe
# Extract just the filename from full path
test_ids = [f.name for f in test_files]

# For binary classification, use probability of positive class (class 1)
submission_df = pd.DataFrame({
    'id': test_ids,
    'has_cactus': test_probs[:, 1]  # Probability of class 1 (has cactus)
})

# Sort by id to match sample submission format
submission_df = submission_df.sort_values('id').reset_index(drop=True)

# Save submission
submission_path = workspace / 'submission.csv'
submission_df.to_csv(submission_path, index=False)

print(f"  ✓ Submission saved to {submission_path}")
print(f"  Submission shape: {submission_df.shape}")
print(f"\nFirst few predictions:")
print(submission_df.head(10))
print(f"\nPrediction statistics:")
print(f"  Min probability: {submission_df['has_cactus'].min():.6f}")
print(f"  Max probability: {submission_df['has_cactus'].max():.6f}")
print(f"  Mean probability: {submission_df['has_cactus'].mean():.6f}")
print(f"  Median probability: {submission_df['has_cactus'].median():.6f}")

print("\n" + "=" * 60)
print("Training pipeline completed successfully!")
print("=" * 60)
