# GPU Optimization Guide

All training and inference must use GPU. CPU training is 10-100x slower and wastes compute time.

## Verify GPU Availability

```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## Framework-Specific GPU Usage

### PyTorch

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to GPU
model = MyModel()
model.to(device)

# Move data to GPU
inputs = inputs.to(device)
targets = targets.to(device)

# Or use .cuda() shorthand
model.cuda()
inputs.cuda()
targets.cuda()
```

### XGBoost

```python
params = {
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    # ... other params
}

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)
```

### LightGBM

```python
params = {
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    # ... other params
}

model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train)
```

### CatBoost

```python
model = CatBoostClassifier(
    task_type='GPU',
    devices='0',
    # ... other params
)
model.fit(X_train, y_train)
```

### TensorFlow/Keras

```python
# TensorFlow automatically uses GPU if available
import tensorflow as tf

# Verify GPU
print(tf.config.list_physical_devices('GPU'))

# Model automatically runs on GPU
model = tf.keras.Sequential([...])
model.fit(X_train, y_train)
```

## Maximizing GPU Utilization

### Batch Size Optimization

**Goal**: Use largest batch size that fits in GPU memory.

```python
# Start large, reduce if OOM
batch_sizes_to_try = [4096, 2048, 1024, 512, 256]

for batch_size in batch_sizes_to_try:
    try:
        train_loader = DataLoader(dataset, batch_size=batch_size)
        model.train()
        # Try one batch
        batch = next(iter(train_loader))
        output = model(batch.to(device))
        loss = criterion(output, targets)
        loss.backward()
        print(f"Batch size {batch_size} fits!")
        break
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"Batch size {batch_size} too large, trying smaller...")
            torch.cuda.empty_cache()
        else:
            raise e
```

### Mixed Precision Training

**Benefit**: 2-3x speedup with minimal accuracy impact.

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in train_loader:
    inputs = inputs.cuda()
    targets = targets.cuda()

    optimizer.zero_grad()

    # Use mixed precision
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Efficient Data Loading

```python
import os

# Maximize parallel data loading
num_workers = os.cpu_count()  # Use all CPU cores

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,  # Parallel loading
    pin_memory=True,          # Faster GPU transfer
    persistent_workers=True   # Keep workers alive
)
```

## Monitoring GPU Usage

### During Training

```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

**What to check:**
- GPU Utilization: Should be 80-100%
- GPU Memory: Should be nearly full (but not OOM)
- Temperature: Should be <85°C

### Low GPU Utilization?

**Possible causes:**
1. **CPU bottleneck**: Increase `num_workers` in DataLoader
2. **Small batch size**: Increase batch size
3. **Not using GPU**: Check code actually moves tensors to GPU
4. **I/O bottleneck**: Use faster storage or preload data to RAM

## Common Mistakes

### Forgetting to Move Data

```python
# WRONG: Model on GPU, data on CPU
model.cuda()
output = model(inputs)  # Error or silent CPU fallback

# RIGHT: Move both
model.cuda()
inputs = inputs.cuda()
output = model(inputs)
```

### Moving Data in Loop

```python
# WRONG: Moves data every iteration (slow)
for epoch in range(100):
    model.cuda()
    inputs.cuda()

# RIGHT: Move once before loop
model.cuda()
for epoch in range(100):
    inputs = next_batch().cuda()  # Only move new batches
```

### Not Clearing Cache

```python
# After OOM error, clear cache before retry
torch.cuda.empty_cache()
```

## Prediction on GPU

**Important**: predict.py must also use GPU for fast inference.

```python
# Load model to GPU
model = torch.load('model.pt')
model.cuda()
model.eval()

# Batch prediction on GPU
predictions = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.cuda()
        pred = model(batch)
        predictions.append(pred.cpu().numpy())

predictions = np.concatenate(predictions)
```
