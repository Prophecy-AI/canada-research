# Final GPU Optimization Fix - Complete Summary

## Problem Statement

**Observed:** Agent used `batch_size=32` on A100-40GB GPU → Only 1% GPU memory utilized → 25+ minute training time → Timeout → No submission generated

**Root Cause:** Ambiguous prompt gave ranges (e.g., "32-64") and LLM conservatively chose minimum value instead of maximum

---

## Complete Solution Applied

### 1. Explicit Batch Size Defaults (No Ranges!)

**Location:** [kaggle_agent.py:200-212](mle-bench/agents/agent_v5_kaggle/kaggle_agent.py#L200-L212)

**Changed FROM:**
```
• GPU: Max batch sizes for A10. Start large, reduce by 2x if OOM.
  - CNNs: batch_size=64-128 (EfficientNet-B4/B5), 32-64 (B6/B7)
```

**Changed TO:**
```
• GPU: ALWAYS START WITH MAXIMUM BATCH SIZE. Never be conservative!
  - CRITICAL: batch_size=32 is TOO SMALL and wastes 80% of GPU! ALWAYS start with 128+ for images
  - Image Classification (224x224): batch_size=128 (start here, increase to 192 if no OOM)
  - Image Classification (384x384): batch_size=64 (start here, increase to 96 if no OOM)
  - EfficientNet-B4/B5: batch_size=128 (DEFAULT for most competitions)
  - EfficientNet-B6/B7: batch_size=64 (only for very large models)
  - Tabular NNs: batch_size=4096-8192
  - RULE: If GPU util <60% after 1 minute, DOUBLE the batch size immediately
```

**Why:** Clear defaults eliminate ambiguity. LLM will now default to 128, not 32.

### 2. Concrete Code Example

**Location:** [kaggle_agent.py:108-118](mle-bench/agents/agent_v5_kaggle/kaggle_agent.py#L108-L118)

**Added:**
```python
- IMAGES: Start with batch_size=128 (NOT 32!). For A10 24GB, 128 is safe for most models
- TABULAR: Start with batch_size=4096 minimum (tabular models are tiny)
- Example for EfficientNet-B4 on 224x224 images:
  ```python
  BATCH_SIZE = 128  # Start here for A10 24GB
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                            num_workers=10, pin_memory=True,
                            prefetch_factor=4, persistent_workers=True)
  ```
```

**Why:** Gives LLM a concrete pattern to copy-paste.

### 3. Mandatory GPU Checkpoint (60 seconds)

**Location:** [kaggle_agent.py:129-134](mle-bench/agents/agent_v5_kaggle/kaggle_agent.py#L129-L134)

**Added to workflow:**
```
3. **MANDATORY GPU CHECK (60 seconds after launch):**
   - Read training output with ReadBashOutput
   - Look for GPU memory usage print (should show XX.X GB / YY.Y GB)
   - **If GPU memory <50% → KILL TRAINING IMMEDIATELY, increase batch_size by 2x, relaunch**
   - **If no GPU memory print found → KILL TRAINING, add GPU monitoring code, relaunch**
   - Only proceed if GPU memory >50% and batch processing speed looks good
```

**Why:** Forces agent to verify GPU usage after 60 seconds. Prevents wasting 25 minutes on underutilized training.

### 4. Mandatory GPU Monitoring Code

**Location:** [kaggle_agent.py:252-267](mle-bench/agents/agent_v5_kaggle/kaggle_agent.py#L252-L267)

**Added requirements:**
```python
• MANDATORY prints at start AND after first batch:
  # At start
  print(f"RESOURCES: {os.cpu_count()} CPU cores, batch={BATCH_SIZE}, GPU={torch.cuda.get_device_name(0)}, Mixed Precision={'ON' if USE_AMP else 'OFF'}")
  print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
  print(f"DataLoader: num_workers={NUM_WORKERS}, prefetch_factor={PREFETCH_FACTOR}, persistent_workers={PERSISTENT_WORKERS}")

  # After first forward pass in training loop
  print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB ({torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100:.1f}%)")
  print(f"VALIDATION: If <50% memory, batch_size={BATCH_SIZE} is TOO SMALL - should be {BATCH_SIZE*2}+")
```

**Why:** Ensures agent can verify GPU usage. Self-documenting for debugging.

### 5. Enhanced DataLoader Configuration

**Location:** [kaggle_agent.py:213-222](mle-bench/agents/agent_v5_kaggle/kaggle_agent.py#L213-L222)

**Enhanced:**
```python
• DataLoader (CRITICAL for GPU saturation):
  - num_workers=8-12 (high for async loading while GPU computes)
  - pin_memory=True (mandatory)
  - prefetch_factor=3-4 (preload more batches)
  - persistent_workers=True (avoid worker respawn overhead)
```

**Why:** Better data pipeline = GPU never waits for CPU = higher throughput.

### 6. Oracle Review Explicitly Checks Batch Size

**Location:** [kaggle_agent.py:125](mle-bench/agents/agent_v5_kaggle/kaggle_agent.py#L125)

**Changed FROM:**
```
Ask: "Review for: GPU usage, resource utilization, data leakage, label encoding bugs, parameter issues, or any logic errors."
```

**Changed TO:**
```
Ask: "Review for: **batch_size (should be 128+ for images, 4096+ for tabular, NOT 32!)**, GPU usage, resource utilization, DataLoader config (num_workers=10+), mixed precision enabled, data leakage, label encoding bugs, parameter issues, or any logic errors."
```

**Why:** Oracle (O3) will explicitly check batch size and flag if it's too small.

---

## Expected Performance Improvements

| Metric | Before (batch=32) | After (batch=128) | Improvement |
|--------|-------------------|-------------------|-------------|
| GPU Memory Used | 0.4 GB (1%) | 17-20 GB (70-80%) | **70x more** |
| GPU Utilization | 10-20% | 80-95% | **4-8x more** |
| Training Speed | 1.5 min/epoch | 0.3 min/epoch | **5x faster** |
| Total Training Time | 25+ min (timeout) | 5-7 min (complete) | **4x faster** |
| Submission Generated | ❌ No | ✅ Yes | **Success** |

---

## Safety Mechanisms

1. **Mandatory 60-second checkpoint** → Detects underutilization early
2. **Automatic kill and relaunch** → Prevents wasting compute on slow training
3. **GPU memory validation prints** → Self-documenting, easy to debug
4. **Oracle review includes batch size** → Catches issues before launch
5. **Graceful degradation for predict.py** → Generates submission even with partial models

---

## Testing Checklist

When reviewing agent logs, verify:

- [ ] `BATCH_SIZE = 128` (or higher) in training script for images
- [ ] `BATCH_SIZE = 4096` (or higher) in training script for tabular
- [ ] Training script prints `GPU Memory Used: XX.X GB / YY.Y GB (ZZ%)`
- [ ] GPU memory usage shows >50% (ideally 70-80%)
- [ ] Agent checks GPU usage 60 seconds after training starts
- [ ] If GPU memory <50%, agent kills training and relaunches with 2x batch size
- [ ] DataLoader has `num_workers=8-12, prefetch_factor=3-4, persistent_workers=True`
- [ ] Mixed precision enabled (`from torch.cuda.amp import autocast, GradScaler`)
- [ ] Oracle review mentions batch size validation

---

## Files Changed

**Primary file:** [mle-bench/agents/agent_v5_kaggle/kaggle_agent.py](mle-bench/agents/agent_v5_kaggle/kaggle_agent.py)

**Key line ranges:**
- **Lines 108-118:** Explicit batch size defaults + concrete example code
- **Lines 125:** Oracle review explicitly checks batch size
- **Lines 129-134:** Mandatory GPU checkpoint at 60 seconds (kill if underutilized)
- **Lines 200-212:** Aggressive batch size guidelines (no ranges, explicit values)
- **Lines 213-222:** Enhanced DataLoader configuration
- **Lines 252-267:** Mandatory GPU monitoring code requirements

---

## Related Documents

1. [DEEPSEEK_API_KEY_FIX.md](DEEPSEEK_API_KEY_FIX.md) - Fixed DeepSeek API key passing to container
2. [GPU_OPTIMIZATION_IMPROVEMENTS.md](GPU_OPTIMIZATION_IMPROVEMENTS.md) - Initial GPU optimization improvements
3. [AGGRESSIVE_GPU_OPTIMIZATION.md](AGGRESSIVE_GPU_OPTIMIZATION.md) - Detailed explanation of batch size strategy
4. [ORACLE_UPGRADE_SUMMARY.md](ORACLE_UPGRADE_SUMMARY.md) - Multi-model Oracle upgrade (O3 + DeepSeek-R1)

---

## Summary

**Problem:** batch_size=32 → 1% GPU usage → 25 min timeout → no submission

**Root Cause:** Ambiguous prompt ranges → LLM picked conservative minimum

**Solution:**
1. ✅ Explicit defaults (128 for images, 4096 for tabular)
2. ✅ Concrete code example to copy
3. ✅ Mandatory GPU checkpoint at 60 seconds
4. ✅ Mandatory GPU monitoring prints
5. ✅ Enhanced DataLoader config
6. ✅ Oracle review checks batch size

**Expected Result:** 4-5x faster training, 70-80% GPU utilization, submission always generated

**Status:** ✅ **COMPLETE** - Ready for production testing

---

## Next Steps

1. **Commit changes:**
   ```bash
   git add mle-bench/agents/agent_v5_kaggle/kaggle_agent.py
   git commit -m "Fix: Aggressive GPU optimization - enforce batch_size=128+ with mandatory monitoring"
   git push
   ```

2. **Test on real competition:**
   - Run agent on dog-breed-identification or similar
   - Verify logs show `BATCH_SIZE = 128`
   - Verify GPU memory shows 70-80% usage
   - Verify training completes in 5-10 minutes (not 25+)
   - Verify submission.csv is generated

3. **Monitor first agent run:**
   - Check that 60-second GPU checkpoint triggers
   - Verify agent doesn't kill and relaunch (batch size should be correct first time)
   - Confirm training speed matches expectations

---

**Date:** 2025-10-16
**Author:** Claude (Sonnet 4.5)
**Tested:** Pending production validation
