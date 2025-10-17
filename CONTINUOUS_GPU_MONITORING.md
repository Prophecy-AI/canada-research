# Continuous GPU Monitoring During Training

## Overview

Enhanced the agent to continuously monitor GPU usage throughout training (not just at 60 seconds), with context-aware expectations based on model size.

---

## Problem

Previous implementation only checked GPU at 60 seconds after launch. This didn't:
1. Track GPU usage throughout training (could degrade over time)
2. Distinguish between small models (acceptable 50-60% GPU) vs large models (should be 70%+)
3. Allow agent to learn from GPU trends during training

---

## Solution: Multi-Stage GPU Monitoring

### Stage 1: Initial Check (60 seconds)

**Location:** [kaggle_agent.py:144-150](mle-bench/agents/agent_v5_kaggle/kaggle_agent.py#L144-L150)

```
5. MANDATORY GPU CHECK (60 seconds after launch):
   - Read training output with ReadBashOutput
   - Look for GPU memory usage print (should show XX.X GB / YY.Y GB)
   - If GPU memory <50% → KILL TRAINING IMMEDIATELY, increase batch_size by 2x, relaunch
   - If GPU memory 50-70% → OPTIONAL: Can increase batch_size by 1.5x for better utilization
   - If no GPU memory print found → KILL TRAINING, add GPU monitoring code, relaunch
   - Only proceed if GPU memory >50% and batch processing speed looks good
```

**Key Addition:**
- New middle threshold: 50-70% allows optional 1.5x batch increase
- Distinguishes between "must fix" (<50%) and "could optimize" (50-70%)

### Stage 2: Continuous Monitoring (Every 120-180s)

**Location:** [kaggle_agent.py:153-158](mle-bench/agents/agent_v5_kaggle/kaggle_agent.py#L153-L158)

```
8. Monitor GPU usage during training (every 120-180s):
   - Check GPU memory in training logs (should print every epoch)
   - If model small (e.g., ResNet-18, tabular NN) and GPU <60%: This is acceptable, note it
   - If model large (e.g., EfficientNet-B4+, ViT) and GPU <60%: Consider increasing batch size
   - Goal: Maximize GPU without OOM. Small underutilization is okay for small models.
   - If consistently <50% GPU for large model: Plan to increase batch_size in next iteration
```

**Key Features:**
- **Context-aware:** Different expectations for small vs large models
- **Continuous:** Monitors every 2-3 minutes throughout training
- **Adaptive:** Plans improvements for next iteration if underutilized
- **Pragmatic:** Accepts lower GPU for small models (50-60% is fine)

### Stage 3: Training Script Requirements

**Location:** [kaggle_agent.py:289-310](mle-bench/agents/agent_v5_kaggle/kaggle_agent.py#L289-L310)

```python
• MANDATORY GPU monitoring during training:
  - Print GPU memory usage EVERY EPOCH: torch.cuda.memory_allocated() / 1024**3
  - Track GPU usage throughout training to ensure maximization
  - If GPU memory <50% after first epoch → batch size is TOO SMALL → STOP and rewrite with 2x batch size
  - If GPU memory 50-70% consistently: Note model size. Small models (ResNet-18, tabular)
    may not fully utilize GPU - this is acceptable. Large models (EfficientNet-B4+, ViT)
    should be 70%+ - consider increasing batch size.
  - Target: 70-90% GPU memory usage, 80-95% GPU utilization (for large models)
  - Acceptable lower utilization for small/simple models (e.g., 50-60% for ResNet-18 is fine)

# EVERY EPOCH (inside training loop):
# Print at end of each epoch to monitor GPU usage throughout training
print(f"Epoch {epoch}: Loss={train_loss:.4f}, GPU={torch.cuda.memory_allocated() / 1024**3:.2f}GB ({torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100:.1f}%)")
```

---

## Context-Aware GPU Targets

### Small Models (Acceptable 50-60% GPU)

**Examples:**
- ResNet-18, ResNet-34
- EfficientNet-B0, B1, B2
- MobileNet variants
- Tabular neural networks (small MLPs)
- Linear models (logistic regression, SVM)

**Why lower GPU is okay:**
- Model too small to saturate GPU compute
- Memory footprint naturally smaller
- Increasing batch size has diminishing returns
- Time spent on data loading becomes bottleneck

**Expected output:**
```
GPU Memory Used: 12.3 GB / 24.0 GB (51.3%)
Model: ResNet-18
Agent: "GPU at 51% is acceptable for small model (ResNet-18).
        Proceeding without adjustment."
```

### Large Models (Should be 70%+ GPU)

**Examples:**
- EfficientNet-B4, B5, B6, B7
- ResNet-101, ResNet-152
- Vision Transformers (ViT-Base, ViT-Large)
- BERT, RoBERTa, GPT variants
- Large CNNs (DenseNet-161, EfficientNetV2-L)

**Why higher GPU expected:**
- Model large enough to saturate GPU
- Memory footprint substantial
- Compute-intensive forward/backward passes
- Batch size can be increased significantly

**Expected output:**
```
GPU Memory Used: 19.2 GB / 24.0 GB (80.0%)
Model: EfficientNet-B4
Agent: "GPU at 80% is optimal for large model (EfficientNet-B4).
        Proceeding as planned."
```

**If underutilized:**
```
GPU Memory Used: 12.3 GB / 24.0 GB (51.3%)
Model: EfficientNet-B4
Agent: "GPU at 51% is LOW for large model (EfficientNet-B4).
        Should be 70%+. Increasing batch_size from 128 to 192."
```

---

## GPU Monitoring Timeline

### Example: Dog Breed Classification (EfficientNet-B4)

**Timeline:**
```
[00:00] Agent writes train.py with batch_size=128
[00:30] Training launches
[01:00] 60-second GPU check:
        GPU: 18.3 GB / 24.0 GB (76.3%) ✓
        Model: EfficientNet-B4 (large model)
        Agent: "GPU at 76% is optimal for large model. Proceeding."

[01:30] Writes predict.py

[03:00] Continuous monitoring check:
        Epoch 3 logs: "Epoch 3: Loss=0.542, GPU=18.1GB (75.4%)"
        Agent: "GPU stable at 75%, training efficiently."

[05:00] Continuous monitoring check:
        Epoch 6 logs: "Epoch 6: Loss=0.321, GPU=17.9GB (74.6%)"
        Agent: "GPU consistent ~75%, no adjustment needed."

[07:00] Continuous monitoring check:
        Epoch 9 logs: "Epoch 9: Loss=0.198, GPU=18.0GB (75.0%)"
        Agent: "Training completing, GPU usage stable."

[08:30] Training completes
        Agent: "GPU maintained 74-76% throughout training. Optimal."
```

### Example: MNIST (Small CNN)

**Timeline:**
```
[00:00] Agent writes train.py with batch_size=256
[00:30] Training launches
[01:00] 60-second GPU check:
        GPU: 8.2 GB / 24.0 GB (34.2%) ⚠️
        Model: Small CNN (3 conv layers)
        Agent: "GPU at 34% but model is small (simple CNN).
                Could increase to batch_size=512, but may not help much.
                Trying batch_size=384 for slight improvement."
        → Restarts with batch_size=384

[01:30] 60-second GPU check (retry):
        GPU: 11.7 GB / 24.0 GB (48.8%) ⚠️ (borderline)
        Agent: "GPU at 49% - borderline. Trying batch_size=512."
        → Restarts with batch_size=512

[02:00] 60-second GPU check (retry 2):
        GPU: 14.1 GB / 24.0 GB (58.8%) ✓
        Agent: "GPU at 59% is acceptable for small model.
                Further increases likely won't help (data loading bottleneck).
                Proceeding."

[02:30] Continues normally...
```

---

## Agent Decision Matrix

| GPU % | Model Size | Action | Rationale |
|-------|------------|--------|-----------|
| <50% | Any | Kill & 2x batch | Severe underutilization |
| 50-60% | Small | Proceed | Acceptable for small models |
| 50-60% | Large | Optional: 1.5x batch | Could optimize, but not critical |
| 60-70% | Small | Proceed | Good for small models |
| 60-70% | Large | Proceed (note suboptimal) | Acceptable but not ideal |
| 70-90% | Any | Proceed | Optimal range |
| >90% | Any | Proceed (near limit) | Excellent, pushing limits |

---

## Training Script Template (Updated)

**With continuous GPU monitoring:**

```python
# ... setup code ...

for fold in range(N_FOLDS):
    print(f"\n=== Fold {fold+1}/{N_FOLDS} ===")

    # Create model
    model = create_model().cuda()

    # Print initial GPU info
    print(f"RESOURCES: {os.cpu_count()} CPU cores, batch={BATCH_SIZE}, "
          f"GPU={torch.cuda.get_device_name(0)}, Mixed Precision=ON")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    first_batch = True

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for i, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()

            with autocast():
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()

            # First batch GPU check
            if first_batch:
                mem = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"GPU Memory Used: {mem:.2f} GB / {total:.1f} GB ({mem/total*100:.1f}%)")
                print(f"VALIDATION: If <50% memory, batch_size={BATCH_SIZE} is TOO SMALL")
                first_batch = False

        # Validation
        model.eval()
        val_loss, val_acc = validate(model, val_loader)

        # GPU monitoring EVERY EPOCH ← NEW
        mem = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Epoch {epoch}: Loss={val_loss:.4f}, Acc={val_acc:.4f}, "
              f"GPU={mem:.2f}GB ({mem/total*100:.1f}%)")  # ← Agent monitors this

        # Checkpointing, early stopping, etc.
        ...
```

**Output example:**
```
=== Fold 1/3 ===
RESOURCES: 32 CPU cores, batch=128, GPU=NVIDIA A10, Mixed Precision=ON
GPU Memory: 24.0 GB
GPU Memory Used: 18.3 GB / 24.0 GB (76.3%)
VALIDATION: If <50% memory, batch_size=128 is TOO SMALL

Epoch 0: Loss=0.891, Acc=0.723, GPU=18.2GB (75.8%)  ← Agent sees this
Epoch 1: Loss=0.654, Acc=0.801, GPU=18.1GB (75.4%)  ← Agent sees this
Epoch 2: Loss=0.542, Acc=0.847, GPU=18.3GB (76.3%)  ← Agent sees this
...

Agent monitoring (every 120s): "GPU stable at 75-76%, training efficiently."
```

---

## Benefits

### 1. Continuous Awareness
**Before:** Only checked at 60 seconds, no ongoing monitoring
**After:** Checks every 120-180s throughout training, tracks trends

### 2. Context-Aware Decisions
**Before:** Same 70-90% target for all models
**After:** Different expectations for small (50-60% ok) vs large (70%+ expected) models

### 3. Adaptive Learning
**Before:** Fixed batch size for entire training
**After:** Can adjust strategy for next iteration if GPU underutilized

### 4. Better Diagnostics
**Before:** Single GPU snapshot
**After:** Full GPU usage history (every epoch)

**Example agent reasoning:**
```
Agent: "GPU history shows gradual decline from 76% → 72% over 10 epochs.
        This is normal (optimizer states accumulate). No concern."
```

vs

```
Agent: "GPU jumped from 76% → 45% at epoch 5.
        Potential memory leak or batch size change in code? Investigating..."
```

---

## Files Modified

### 1. Main Agent Prompt
**File:** [mle-bench/agents/agent_v5_kaggle/kaggle_agent.py](mle-bench/agents/agent_v5_kaggle/kaggle_agent.py)

**Changes:**
- Lines 144-150: Added 50-70% middle threshold, optional 1.5x batch increase
- Lines 153-158: NEW continuous monitoring workflow (every 120-180s)
- Lines 289-310: Enhanced GPU monitoring requirements with context awareness
- Line 310: NEW epoch-level GPU printing requirement

### 2. Training Hints Template
**File:** [mle-bench/environment/training_hints.txt](mle-bench/environment/training_hints.txt)

**Changes:**
- Lines 390-393: Added GPU monitoring print every epoch in template code

---

## Validation Checklist

When reviewing agent logs:

### ✅ Initial GPU Check (60s)
- [ ] Agent checks GPU at 60 seconds after launch
- [ ] If GPU <50%: Agent kills and relaunches with 2x batch
- [ ] If GPU 50-70%: Agent notes and optionally increases by 1.5x
- [ ] If GPU 70%+: Agent proceeds without adjustment

### ✅ Continuous Monitoring
- [ ] Agent checks GPU every 120-180 seconds during training
- [ ] Agent reads epoch logs showing GPU memory usage
- [ ] Agent distinguishes small models (50-60% ok) vs large models (70%+ expected)
- [ ] Agent tracks GPU trends (stable, increasing, decreasing)

### ✅ Training Script Output
- [ ] Script prints GPU memory after first batch
- [ ] Script prints GPU memory EVERY EPOCH
- [ ] Format: "Epoch X: Loss=Y, GPU=ZGB (W%)"
- [ ] Agent can parse and monitor these prints

### ✅ Context-Aware Decisions
- [ ] For small models: Agent accepts 50-60% GPU
- [ ] For large models: Agent targets 70-90% GPU
- [ ] Agent notes model architecture when assessing GPU usage

---

## Summary

✅ **Added continuous GPU monitoring** (every 120-180s, not just 60s)
✅ **Context-aware targets** (50-60% ok for small models, 70%+ for large)
✅ **Middle threshold** (50-70% → optional 1.5x batch increase)
✅ **Epoch-level printing** (GPU memory printed every epoch for tracking)
✅ **Adaptive learning** (agent learns from GPU trends, plans improvements)

**Expected impact:**
- Better GPU utilization throughout training (not just initially)
- Smarter decisions based on model size
- Early detection of GPU degradation or memory leaks
- More data for debugging and optimization

**Status:** ✅ Complete, ready for testing

---

**Date:** 2025-10-16
**Related:** GPU optimization, time constraints, training hints
