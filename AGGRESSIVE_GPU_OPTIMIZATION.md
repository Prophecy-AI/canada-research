# Aggressive GPU Optimization Strategy - Final Fix

## The Core Problem

The agent used **batch_size=32** on an A100-40GB GPU (or A10-24GB in production), which resulted in:
- **Only 0.4 GB / 39.5 GB GPU memory used** (~1% utilization)
- **GPU sitting idle 80-90% of the time** waiting for CPU to prepare tiny batches
- **25+ minutes of training time** for something that should take 5-10 minutes

**Root cause:** The prompt gave ranges like "32-64" and the LLM chose the conservative minimum (32) instead of the maximum.

---

## Solution: Aggressive Defaults + Mandatory Monitoring

### 1. **Explicit Aggressive Defaults**

**Before (ambiguous):**
```
- CNNs: batch_size=64-128 (EfficientNet-B4/B5), 32-64 (EfficientNet-B6/B7)
```
→ LLM picks 32 (conservative)

**After (explicit):**
```
- CRITICAL: batch_size=32 is TOO SMALL and wastes 80% of GPU! ALWAYS start with 128+ for images
- Image Classification (224x224): batch_size=128 (start here, increase to 192 if no OOM)
- Image Classification (384x384): batch_size=64 (start here, increase to 96 if no OOM)
- EfficientNet-B4/B5: batch_size=128 (DEFAULT for most competitions)
- EfficientNet-B6/B7: batch_size=64 (only for very large models)
- RULE: If GPU util <60% after 1 minute, DOUBLE the batch size immediately
```
→ LLM will use 128

### 2. **Mandatory GPU Checkpoint (60 seconds after launch)**

New workflow enforces GPU monitoring:

```
CRITICAL WORKFLOW:
1. Write train.py and validate with Oracle
2. Launch training: Bash(command="python -u train.py", background=true)
3. **MANDATORY GPU CHECK (60 seconds after launch):**
   - Read training output with ReadBashOutput
   - Look for GPU memory usage print (should show XX.X GB / YY.Y GB)
   - **If GPU memory <50% → KILL TRAINING IMMEDIATELY, increase batch_size by 2x, relaunch**
   - **If no GPU memory print found → KILL TRAINING, add GPU monitoring code, relaunch**
   - Only proceed if GPU memory >50% and batch processing speed looks good
4. IMMEDIATELY write predict.py (don't wait for training)
...
```

**This forces the agent to:**
- Check GPU usage after 60 seconds
- Kill and restart if underutilized
- Never waste 25 minutes on slow training

### 3. **Mandatory GPU Memory Prints**

Every training script MUST print:

```python
# At start
print(f"RESOURCES: {os.cpu_count()} CPU cores, batch={BATCH_SIZE}, GPU={torch.cuda.get_device_name(0)}, Mixed Precision={'ON' if USE_AMP else 'OFF'}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"DataLoader: num_workers={NUM_WORKERS}, prefetch_factor={PREFETCH_FACTOR}, persistent_workers={PERSISTENT_WORKERS}")

# After first forward pass in training loop
print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB ({torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100:.1f}%)")
print(f"VALIDATION: If <50% memory, batch_size={BATCH_SIZE} is TOO SMALL - should be {BATCH_SIZE*2}+")
```

**Output should look like:**
```
RESOURCES: 32 CPU cores, batch=128, GPU=NVIDIA A10, Mixed Precision=ON
GPU Memory: 24.0 GB
DataLoader: num_workers=10, prefetch_factor=4, persistent_workers=True
GPU Memory Used: 18.3 GB / 24.0 GB (76.3%)
VALIDATION: If <50% memory, batch_size=128 is TOO SMALL - should be 256+
```

### 4. **Concrete Example in Prompt**

Added explicit code example:

```python
# Example for EfficientNet-B4 on 224x224 images:
BATCH_SIZE = 128  # Start here for A10 24GB
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          num_workers=10, pin_memory=True,
                          prefetch_factor=4, persistent_workers=True)
```

This gives the LLM a concrete pattern to copy.

---

## Expected Performance Improvement

### Before (batch_size=32):
- **GPU memory:** 0.4 GB / 39.5 GB (1% usage)
- **GPU utilization:** 10-20% (idle most of the time)
- **Training speed:** ~1.5 min/epoch → 25+ min total
- **Bottleneck:** CPU data loading

### After (batch_size=128):
- **GPU memory:** 15-20 GB / 24 GB (70-80% usage)
- **GPU utilization:** 80-95% (constantly busy)
- **Training speed:** ~0.3 min/epoch → 5-7 min total
- **Bottleneck:** GPU compute (optimal)

**Expected speedup: 4-5x faster training**

---

## Why Batch Size Matters So Much

### Technical Explanation:

**Small batch (32):**
```
GPU: [Process 32 images] ←─ 0.05s (5% time)
GPU: [Wait for CPU]     ←─ 0.95s (95% time) 🐢
CPU: [Load 32 images, augment, transfer to GPU] ←─ 0.95s
→ Total: 1.0s per batch
→ GPU idle 95% of the time
```

**Large batch (128):**
```
GPU: [Process 128 images] ←─ 0.15s (60% time)
CPU: [Load 128 images, augment, transfer to GPU] ←─ 0.10s (40% time)
→ Total: 0.25s per batch
→ GPU busy 60% of the time
→ 4x more images processed per second
```

**With num_workers=10 + prefetch_factor=4:**
```
GPU: [Process batch N]     ←─ 0.15s (GPU busy)
CPU: [Preparing batch N+4] ←─ 0.15s (parallel, ready when GPU needs it)
→ GPU never waits
→ GPU busy 90%+ of the time
```

### Memory Targets:

| GPU | Safe Batch (224px) | Aggressive Batch | Memory Target |
|-----|-------------------|------------------|---------------|
| A10 24GB | 128 | 192 | 17-20 GB (70-80%) |
| A100 40GB | 192 | 256 | 28-34 GB (70-85%) |

**Rule:** If memory <50% → batch too small → double it

---

## Implementation in Agent Prompt

### Key Changes Made:

1. **Line 201:** "CRITICAL: batch_size=32 is TOO SMALL and wastes 80% of GPU!"
2. **Line 202-212:** Specific batch sizes by model and resolution (no ranges, just explicit numbers)
3. **Line 212:** "If GPU util <60% after 1 minute, DOUBLE the batch size"
4. **Line 108-118:** Explicit example code with BATCH_SIZE=128
5. **Line 121-126:** Mandatory GPU checkpoint 60 seconds after launch
6. **Line 252-267:** Mandatory GPU monitoring prints with validation logic

### Philosophy Shift:

**Before:** "Here's a range, you decide"
→ LLM picks conservative minimum

**After:** "Start with this specific value, increase if no OOM, monitor and kill if underutilized"
→ LLM uses aggressive default, forced to verify it's working

---

## Validation Checklist

When reviewing agent logs, verify:

- [ ] Training script has `BATCH_SIZE = 128` (or higher) for images
- [ ] Training script prints GPU memory usage after first batch
- [ ] GPU memory usage shows >50% (ideally 70-80%)
- [ ] Agent checks GPU usage after 60 seconds
- [ ] If GPU memory <50%, agent kills and relaunches with 2x batch size
- [ ] DataLoader has num_workers=8-12, prefetch_factor=3-4, persistent_workers=True
- [ ] Mixed precision enabled (autocast/GradScaler)

---

## Testing Strategy

### Quick Test (Local):

```python
# test_gpu_saturation.py
import torch
import time
from torch.utils.data import DataLoader, TensorDataset

# Create dummy dataset
data = torch.randn(10000, 3, 224, 224)
labels = torch.randint(0, 10, (10000,))
dataset = TensorDataset(data, labels)

# Test different batch sizes
for batch_size in [32, 64, 128, 192]:
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=10,
                       pin_memory=True, prefetch_factor=4, persistent_workers=True)

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
        torch.nn.Linear(64, 10)
    ).cuda()

    start = time.time()
    for batch, target in loader:
        batch, target = batch.cuda(), target.cuda()
        output = model(batch)
    elapsed = time.time() - start

    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"batch_size={batch_size}: {elapsed:.2f}s, {mem_gb:.2f} GB GPU memory")
```

**Expected output:**
```
batch_size=32: 8.5s, 2.1 GB GPU memory  ← Too small!
batch_size=64: 5.2s, 4.3 GB GPU memory  ← Better
batch_size=128: 3.1s, 8.7 GB GPU memory ← Good
batch_size=192: 2.3s, 13.1 GB GPU memory ← Optimal
```

---

## Summary

**Problem:** Agent used batch_size=32 → 1% GPU utilization → 25 min training

**Root Cause:** Ambiguous prompt with ranges → LLM picked conservative minimum

**Solution:**
1. Explicit defaults (128 for images, 4096 for tabular)
2. Mandatory GPU checkpoint at 60 seconds (kill if <50% memory)
3. Mandatory GPU monitoring prints in training scripts
4. Concrete code examples to copy

**Expected Result:** 4-5x faster training with 70-80% GPU utilization

**Files Changed:** [kaggle_agent.py](mle-bench/agents/agent_v5_kaggle/kaggle_agent.py)
- Lines 108-118: Explicit batch size defaults + example code
- Lines 121-126: Mandatory GPU checkpoint workflow
- Lines 200-212: Aggressive batch size guidelines
- Lines 252-267: Mandatory GPU monitoring code

---

**Status:** ✅ Ready for testing. Next agent run should use batch_size=128 and monitor GPU usage.
