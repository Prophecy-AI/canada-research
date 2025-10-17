# Time Constraint & GPU Maximization - Final Additions

## Overview

Added explicit time constraints and GPU maximization mandates to ensure the agent:
1. **Solves competitions in 20±10 minutes (10-30 min target)**
2. **Fully maximizes A10 GPU (70-90% memory, 80-95% utilization)**

---

## Changes Made

### 1. Environment Section - Time & GPU Mandates

**Location:** [kaggle_agent.py:27-28](mle-bench/agents/agent_v5_kaggle/kaggle_agent.py#L27-L28)

**Added:**
```
- **TARGET SOLVE TIME: 20±10 minutes (10-30 min range).** This is a HARD CONSTRAINT for planning.
  You may exceed this if the competition is exceptionally difficult or the dataset is extremely
  large (>100GB), but always aim for efficiency. Plan cv_folds, epochs, and batch_size to fit
  this time budget.

- **GPU UTILIZATION MANDATE: MAXIMIZE A10 GPU usage at all times (target: 70-90% GPU memory,
  80-95% GPU utilization). Underutilizing the GPU is wasteful and slow.**
```

**Why:** Sets clear expectations upfront before agent starts planning.

---

### 2. Time Management Section - Detailed Planning

**Location:** [kaggle_agent.py:121-130](mle-bench/agents/agent_v5_kaggle/kaggle_agent.py#L121-L130)

**Changed FROM:**
```
• TIME MANAGEMENT (CRITICAL):
  - Estimate total training time before starting
  - If estimated time > 80% of total budget: Reduce cv_folds from 5→3
  - Monitor training speed continuously
  - Always reserve 15% of time for inference
  - Use early stopping aggressively
```

**Changed TO:**
```
• TIME MANAGEMENT (CRITICAL - TARGET: 20±10 minutes total solve time):
  - **HARD CONSTRAINT: Aim for 10-30 minute total solve time.** Plan everything around this budget.
  - Estimate total training time before starting (epochs × steps_per_epoch × seconds_per_step)
  - **Default strategy for 20-min target:** 3 CV folds × 10 epochs × early stopping = ~15 min training + 5 min inference
  - **If estimated time >25 minutes:** Reduce to 3 folds, 8 epochs, or smaller model
  - **If dataset extremely large (>100GB) or exceptionally complex:** You may exceed 30 min, but justify the decision
  - Monitor training speed continuously - if first fold takes >5 min, adjust strategy immediately (reduce epochs or folds)
  - **Always reserve 15-20% of time for inference/submission generation** - kill training at 25 min if needed
  - Use early stopping aggressively (patience=3 epochs) to avoid wasting time on plateaued models
  - **Efficiency is key:** Don't waste time on marginal improvements. Get a good baseline fast, then iterate if time permits.
```

**Why:** Provides concrete default strategy (3 folds × 10 epochs) and specific thresholds (first fold >5 min → adjust).

---

### 3. Resource Maximization - GPU Targets

**Location:** [kaggle_agent.py:227-232](mle-bench/agents/agent_v5_kaggle/kaggle_agent.py#L227-L232)

**Changed FROM:**
```
**Resource Maximization Rules (MANDATORY - Assume A10 24GB VRAM):**
• CPU: Always n_jobs=-1 (all cores)
• GPU (Assume A10 24GB VRAM): ALWAYS START WITH MAXIMUM BATCH SIZE. Never be conservative!
  - CRITICAL: batch_size=32 is TOO SMALL and wastes 80% of GPU!
```

**Changed TO:**
```
**Resource Maximization Rules (MANDATORY - MAXIMIZE A10 24GB VRAM):**
• CPU: Always n_jobs=-1 (all cores)
• GPU MAXIMIZATION (A10 24GB VRAM - TARGET: 70-90% memory, 80-95% utilization):
  - **ALWAYS START WITH MAXIMUM BATCH SIZE.** Never be conservative! Push the A10 to its limits.
  - **CRITICAL: batch_size=32 is TOO SMALL and wastes 80% of GPU! ALWAYS start with 128+ for images, 4096+ for tabular**
  - **Goal: Use 17-22 GB out of 24 GB GPU memory (70-90% utilization)**
```

**Why:** Makes GPU targets explicit and quantifiable (17-22 GB out of 24 GB).

---

## Expected Agent Behavior

### Time Planning

**Agent should now:**

1. **Estimate time upfront:**
   ```
   Dataset: 10,000 images (224x224)
   Model: EfficientNet-B4
   Batch size: 128
   Steps per epoch: 10000/(128*3) ≈ 26 steps per fold
   Time per step: ~0.5s (with A10)

   Total training estimate:
   3 folds × 10 epochs × 26 steps × 0.5s = ~6.5 min
   + Inference: ~2 min
   = ~8.5 min total ✓ (within 10-30 min range)
   ```

2. **Adjust if over budget:**
   ```
   Dataset: 100,000 images (384x384)
   Estimated time: 45 minutes

   → TOO LONG! Adjust:
   - Reduce from 5 folds → 3 folds
   - Reduce from 15 epochs → 8 epochs
   - New estimate: ~18 minutes ✓
   ```

3. **Justify if exceeding 30 min:**
   ```
   Dataset: 1,000,000 images (512x512)
   Estimated time: 60 minutes

   Agent: "Dataset is exceptionally large (200GB).
   Estimated 60 min for 3 folds. Proceeding with extended
   time budget due to data scale."
   ```

### GPU Maximization

**Agent should now:**

1. **Target 70-90% GPU memory:**
   ```
   GPU Memory Used: 18.3 GB / 24.0 GB (76.3%) ✓
   VALIDATION: Memory usage optimal
   ```

2. **Increase batch size if underutilized:**
   ```
   GPU Memory Used: 10.2 GB / 24.0 GB (42.5%) ✗
   VALIDATION: Memory <50%, batch_size=64 is TOO SMALL - should be 128+
   → Agent kills training, increases batch_size to 128, relaunches
   ```

3. **Push limits aggressively:**
   ```
   Starting with batch_size=128
   GPU Memory: 18.3 GB (76%) - Good

   Agent: "GPU memory at 76%, let me try batch_size=160 to maximize"
   → Tests batch_size=160
   → GPU Memory: 22.1 GB (92%) - Excellent! Using this.
   ```

---

## Default Strategy for Different Competitions

### Small Dataset (<10K samples)
```
Time target: ~10 minutes
Strategy: 5 folds × 15 epochs × early stopping
Batch size: 128 (images) or 4096 (tabular)
Expected: 7-12 min total
```

### Medium Dataset (10K-100K samples)
```
Time target: ~20 minutes
Strategy: 3 folds × 10 epochs × early stopping
Batch size: 128 (images) or 4096 (tabular)
Expected: 15-25 min total
```

### Large Dataset (100K-1M samples)
```
Time target: ~30 minutes
Strategy: 3 folds × 8 epochs × early stopping
Batch size: 128-192 (images) or 8192 (tabular)
Expected: 25-35 min total
```

### Very Large Dataset (>1M samples or >100GB)
```
Time target: ~40-60 minutes (exception allowed)
Strategy: 3 folds × 5 epochs × early stopping
Batch size: 192+ (images) or 8192 (tabular)
Justification: "Dataset is 500GB, extended time needed"
Expected: 40-60 min total
```

---

## Validation Checklist

When reviewing agent logs, verify:

### ✅ Time Planning
- [ ] Agent estimates total time before training starts
- [ ] Estimate follows format: `folds × epochs × steps × time_per_step = X minutes`
- [ ] If estimate >25 min, agent adjusts strategy (fewer folds/epochs)
- [ ] Agent reserves 15-20% time for inference
- [ ] If first fold takes >5 min, agent adjusts immediately

### ✅ GPU Maximization
- [ ] Agent targets 70-90% GPU memory (17-22 GB out of 24 GB)
- [ ] GPU monitoring prints show >70% memory usage
- [ ] If GPU memory <50%, agent kills and relaunches with larger batch
- [ ] Agent attempts to push batch size higher if memory allows
- [ ] Final GPU utilization is 80-95%

### ✅ Time Adherence
- [ ] Total solve time is 10-30 minutes (or justified if longer)
- [ ] Training killed at 25 min if running over
- [ ] Submission generated within time budget
- [ ] Agent doesn't waste time on marginal improvements (<0.001 gain)

---

## Example Agent Reasoning (Expected)

**Before training:**
```
Agent: "Let me estimate training time:
- Dataset: 25,000 images (224x224)
- Model: EfficientNet-B4
- Batch size: 128 (starting point for A10)
- Steps per fold: 25000/(128*3) ≈ 65 steps
- Time per step: ~0.4s (with mixed precision on A10)

Estimated time:
3 folds × 10 epochs × 65 steps × 0.4s = ~780s = ~13 min
+ Inference: ~2-3 min
= ~15-16 min total ✓

This fits within the 20±10 min target. Proceeding with:
- 3 CV folds
- 10 epochs with early stopping (patience=3)
- batch_size=128
- Target GPU memory: 70-90% (17-22 GB)
"
```

**After 60 seconds:**
```
Agent: "GPU check after 60 seconds:
GPU Memory Used: 19.2 GB / 24.0 GB (80.0%)
GPU Utilization: 87% (from nvidia-smi)
First epoch timing: 26 steps in 12s = 0.46s per step

✓ GPU memory at 80% - excellent
✓ GPU utilization at 87% - optimal
✓ Time per step 0.46s - matches estimate

Projected total time: 3 × 10 × 65 × 0.46s = ~900s = ~15 min
+ inference ~2 min = ~17 min total ✓

Proceeding as planned. No adjustments needed."
```

**If underutilized:**
```
Agent: "GPU check after 60 seconds:
GPU Memory Used: 10.8 GB / 24.0 GB (45.0%)
GPU Utilization: 52%

✗ GPU memory at 45% - TOO LOW (target: 70-90%)
✗ GPU utilization at 52% - underutilized

DECISION: Killing training, doubling batch_size from 64 to 128.
Expected new GPU memory: ~80-90%
Relaunching training now..."
```

---

## Impact Summary

| Aspect | Before | After | Benefit |
|--------|--------|-------|---------|
| **Time Planning** | Vague "finish ASAP" | Explicit 20±10 min target | Clear planning constraint |
| **Default Strategy** | Agent chooses arbitrarily | 3 folds × 10 epochs default | Consistent efficient baseline |
| **Time Monitoring** | Passive monitoring | Active 5-min checkpoints | Early detection of overruns |
| **GPU Target** | "Use GPU" (vague) | 70-90% memory, 80-95% util | Quantifiable, measurable |
| **Efficiency Mindset** | "Get best score" | "Fast baseline, iterate if time" | Speed-quality balance |

---

## Files Changed

**Single file updated:** [mle-bench/agents/agent_v5_kaggle/kaggle_agent.py](mle-bench/agents/agent_v5_kaggle/kaggle_agent.py)

**Three sections modified:**
1. **Lines 27-28:** Environment section - Added time & GPU mandates
2. **Lines 121-130:** Time management - Detailed 20±10 min planning strategy
3. **Lines 227-232:** Resource maximization - GPU targets (70-90% memory)

---

## Testing Recommendations

### Test 1: Time Estimation
Check agent logs for:
```
[Agent] Estimating training time: 3 folds × 10 epochs × 65 steps × 0.4s = ~13 min
[Agent] + Inference: ~2 min = ~15 min total ✓ (within 20±10 min target)
```

### Test 2: Time Adherence
Verify total solve time:
```bash
# Extract start and end times from logs
START_TIME=$(grep "Starting agent run" logs.txt | head -1)
END_TIME=$(grep "Submission generated" logs.txt | head -1)
# Calculate duration - should be 10-30 min for most competitions
```

### Test 3: GPU Maximization
Check GPU monitoring output:
```
[Train] GPU Memory Used: 19.2 GB / 24.0 GB (80.0%)  ← Target: 70-90%
[Train] GPU Utilization: 87%  ← Target: 80-95%
```

### Test 4: Adjustment Logic
If first fold slow, verify agent adjusts:
```
[Agent] First fold took 7 minutes (>5 min threshold)
[Agent] Reducing from 10 epochs to 6 epochs to fit time budget
[Agent] New estimate: ~18 minutes ✓
```

---

## Summary

✅ **Added explicit 20±10 minute time constraint** (10-30 min range)
✅ **Default strategy:** 3 folds × 10 epochs for typical competitions
✅ **GPU maximization targets:** 70-90% memory, 80-95% utilization, 17-22 GB usage
✅ **Efficiency mindset:** Fast baseline first, iterate only if time permits
✅ **Exception handling:** May exceed 30 min for extremely large datasets (>100GB)

**Expected impact:**
- More consistent solve times (less variance)
- Better GPU utilization (avoid 1% usage scenarios)
- Faster iteration (don't waste time on marginal gains)
- Clear planning (agent estimates time upfront)

**Status:** ✅ Complete, ready for testing

---

**Date:** 2025-10-16
**Related:** GPU optimization, training hints, Oracle upgrade
