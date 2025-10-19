# GPU Validation Quick Reference

## For the Agent

### When to Use GPUValidate Tool

1. **First turn (system resource check):** ALWAYS validate GPU before any training
2. **Before long training jobs:** If suspicious about GPU setup
3. **After debugging GPU issues:** Confirm fix worked

### How to Use

```python
# Validate PyTorch (most common)
GPUValidate(framework='pytorch', model_size='small', batch_size=256)

# Validate LightGBM
GPUValidate(framework='lightgbm', rows=100000)

# Validate XGBoost
GPUValidate(framework='xgboost', rows=50000)
```

### Expected Results

**Success:**
```
✅ GPU validation PASSED for pytorch
GPU training is working correctly. Proceed with full training.
```

**Failure:**
```
❌ GPU validation FAILED for pytorch
CPU fallback detected. Check your training code:
- PyTorch: Ensure model and data use .to(device) or .cuda()
Fix the issue before proceeding with full training.
```

## How to Detect GPU vs CPU Training

### Method 1: Epoch Timing (Most Reliable)

**GPU Speed (A100 40GB):**
- EfficientNet-B3: 0.5-1 min/epoch (~30-60s)
- EfficientNet-B4: 1-1.5 min/epoch (~60-90s)
- ResNet-50: 0.3-0.5 min/epoch (~20-30s)

**CPU Speed (36 cores):**
- EfficientNet-B3: 10-20 min/epoch (~600-1200s)
- EfficientNet-B4: 15-30 min/epoch (~900-1800s)
- ResNet-50: 5-10 min/epoch (~300-600s)

**Decision Rule:**
- Epoch <2 min → GPU ✅
- Epoch 2-5 min → Suspicious, check nvidia-smi
- Epoch >5 min → CPU ❌ (KILL IMMEDIATELY)

### Method 2: nvidia-smi (Secondary Check)

```bash
# Check if training process is using GPU
nvidia-smi

# Expected output (GPU in use):
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     12345      C   python                          8192MiB |
+-----------------------------------------------------------------------------+

# Bad output (CPU training):
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### Method 3: GPU Memory Usage (Least Reliable)

**DON'T USE** memory checks after first batch - too early, gives false positives.

**DO USE** memory checks after first epoch (more stable):
- GPU training: Usually 8-20GB allocated for medium models
- CPU training: <1GB allocated (model weights only)

But **epoch timing is more reliable** than memory.

## Common Mistakes to Avoid

### ❌ DON'T: Check GPU memory after first batch
```python
# BAD - gives false positives (too early)
if epoch == 0 and batch == 0:
    mem = torch.cuda.memory_allocated()
    if mem < 0.1:
        raise RuntimeError("GPU not used!")  # WRONG!
```

**Why bad:** PyTorch allocates GPU memory gradually during first epoch. After first batch, you'll only see model weights (~200MB), not full allocation.

### ✅ DO: Check epoch timing after first epoch
```python
# GOOD - reliable indicator
epoch_start = time.time()
# ... training loop ...
epoch_time = time.time() - epoch_start
print(f"Epoch {epoch} completed in {epoch_time:.1f}s")

if epoch == 0 and epoch_time > 300:  # >5 min for first epoch
    print("⚠️  CPU training detected (epoch too slow)")
    raise RuntimeError("GPU not being used - check .to(device) calls")
```

### ❌ DON'T: Aim for 70-90% GPU memory usage
```python
# BAD - over-optimization, wastes time tuning batch size
if gpu_memory < 0.7:
    print("Increase batch size to 70%+ memory!")
```

**Why bad:** Focusing on maxing out GPU memory is premature optimization. Better to train 2-3 models in parallel than spend time tuning one model's batch size.

### ✅ DO: Use reasonable batch sizes and focus on speed
```python
# GOOD - efficient defaults
batch_size = 256  # For 224x224 images on A100
# This will use ~30-50% GPU memory, which is fine!
# Focus on fast training and parallel models, not max utilization
```

## Timing Expectations (A100 40GB)

### Image Classification (1000 samples, 3 folds, 8 epochs)

| Model          | Time/Epoch | Total Time | GPU? |
|----------------|-----------|------------|------|
| EfficientNet-B3| 0.5-1 min | 12-24 min  | ✅   |
| EfficientNet-B3| 10-20 min | 240-480 min| ❌ CPU|
| EfficientNet-B4| 1-1.5 min | 24-36 min  | ✅   |
| ResNet-50      | 0.3-0.5 min| 7-12 min  | ✅   |
| ResNet-50      | 5-10 min  | 120-240 min| ❌ CPU|

### Tabular (LightGBM, 100K rows, 50 features)

| Setup         | Time/100 rounds | GPU? |
|---------------|----------------|------|
| GPU (CUDA)    | 1-3s           | ✅   |
| CPU (36 cores)| 10-20s         | ❌   |

## Debugging Checklist

If GPUValidate fails or epoch timing suggests CPU:

### PyTorch
- [ ] Model moved to device: `model = model.to(device)`
- [ ] Data moved to device: `x = x.to(device)`, `y = y.to(device)`
- [ ] Device defined correctly: `device = torch.device('cuda')`
- [ ] Check torch.cuda.is_available() returns True
- [ ] Run nvidia-smi to confirm GPU visible

### LightGBM
- [ ] Built from source with `-DUSE_CUDA=ON` (PyPI version is CPU-only)
- [ ] Use `device_type='cuda'` (NOT `device='gpu'`)
- [ ] Check lightgbm.__version__ >= 4.0.0
- [ ] Run validate_gpu.py to confirm CUDA support

### XGBoost
- [ ] Use `tree_method='gpu_hist'` (NOT `tree_method='hist'`)
- [ ] Check xgboost.__version__ >= 1.7.0
- [ ] GPU version installed (not CPU-only build)

## Summary

**Key Principle:** Epoch timing is the most reliable GPU indicator. Don't over-rely on memory checks.

**Quick Test:** Run GPUValidate tool before any long training job.

**During Training:** Monitor epoch timing. If >5 min/epoch for medium model → CPU fallback → KILL IMMEDIATELY.

**Goal:** Efficient training with reasonable batch sizes and parallel models, not extreme single-model GPU optimization.
