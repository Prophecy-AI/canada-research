# Common Kaggle Competition Pitfalls

## Data Leakage

### What It Is
Using information from test set during training, causing artificially high CV scores that don't generalize.

### Common Causes

**Statistics on full dataset:**
```python
# WRONG: Computes mean on train+test
scaler = StandardScaler()
scaler.fit(pd.concat([train, test]))

# RIGHT: Fit only on train
scaler = StandardScaler()
scaler.fit(train)
```

**Target encoding leakage:**
```python
# WRONG: Uses test targets for encoding
df['category_encoded'] = df.groupby('category')['target'].transform('mean')

# RIGHT: Fit encoder on train only, transform test
encoder = TargetEncoder()
train['category_encoded'] = encoder.fit_transform(train['category'], train['target'])
test['category_encoded'] = encoder.transform(test['category'])
```

**Time-series leakage:**
```python
# WRONG: Random CV split for time-series
KFold(shuffle=True)

# RIGHT: Time-based split
TimeSeriesSplit()
```

## Label Encoding Bugs

### Column Order Mismatch

**Issue**: Predictions align with wrong rows due to index mismatch.

```python
# WRONG: Sorted differently
train.sort_values('id')
test.sort_values('date')

# RIGHT: Consistent sorting
test = test.sort_values('id').reset_index(drop=True)
predictions = model.predict(test[features])
submission = pd.DataFrame({'id': test['id'], 'target': predictions})
```

### Categorical Encoding Mismatch

**Issue**: Categories encoded differently in train vs test.

```python
# WRONG: Separate encoding
train['cat'] = LabelEncoder().fit_transform(train['cat'])
test['cat'] = LabelEncoder().fit_transform(test['cat'])

# RIGHT: Fit on train, transform test
le = LabelEncoder()
train['cat'] = le.fit_transform(train['cat'])
test['cat'] = le.transform(test['cat'])
```

## Cross-Validation Issues

### Wrong Split Strategy

**Classification**: Use stratified splits to maintain class balance
```python
StratifiedKFold(n_splits=5)  # Not KFold()
```

**Time-series**: Use time-based splits
```python
TimeSeriesSplit(n_splits=5)  # Not KFold()
```

**Grouped data**: Prevent group leakage
```python
GroupKFold(n_splits=5)  # When samples from same user/entity
```

### Too Few Folds

**Issue**: Noisy CV estimates with high variance.

**Solution**: Use 5-10 folds for reliable estimates.

## Resource Underutilization

### CPU Not Maxed

```python
# WRONG: Uses 1 core
model = RandomForestClassifier()

# RIGHT: Uses all cores
model = RandomForestClassifier(n_jobs=-1)
```

### GPU Not Used

```python
# WRONG: CPU training (100x slower)
model = XGBClassifier()

# RIGHT: GPU training
model = XGBClassifier(tree_method='gpu_hist', gpu_id=0)
```

### Batch Size Too Small

```python
# WRONG: 32 batch size on A10 GPU with 24GB RAM
DataLoader(dataset, batch_size=32)

# RIGHT: Max out GPU memory
DataLoader(dataset, batch_size=2048)  # Reduce if OOM
```

## Evaluation Metric Mismatch

### Optimizing Wrong Metric

**Issue**: Using default metric instead of competition metric.

```python
# Competition uses log_loss
# WRONG: Optimizes accuracy
model = LogisticRegression()

# RIGHT: Optimizes log_loss (default for LogisticRegression)
# Or use custom objective for XGBoost/LightGBM
```

### Not Using Competition Metric for CV

```python
# WRONG: Using different metric
cross_val_score(model, X, y, scoring='accuracy')

# RIGHT: Match competition metric
cross_val_score(model, X, y, scoring='neg_log_loss')
```

## Progress Monitoring Issues

### No Progress Logs

**Issue**: Training runs for hours with no output, then fails.

```python
# WRONG: Silent training
model.fit(X, y, epochs=100)

# RIGHT: Progress logs
model.fit(X, y, epochs=100, verbose=1)

# For custom training loops
for epoch in range(100):
    loss = train_epoch()
    print(f"Epoch {epoch}: loss={loss:.4f}")  # Every epoch
```

### Not Killing Broken Runs

**Issue**: Waiting 2 hours for training with NaN loss.

**Solution**: Monitor output, kill immediately on anomalies:
- Errors or exceptions
- NaN/Inf values
- Zero accuracy
- Suspiciously bad metrics

## Reproducibility Issues

### No Random Seeds

**Issue**: Results change between runs, can't debug.

```python
# Set all seeds
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

### Non-Deterministic Operations

**Issue**: GPU operations are non-deterministic by default.

```python
# For PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
