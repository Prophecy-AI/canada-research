# What to Expect in the Next Agent Run

## Before This Fix

**Typical agent execution with dog-breed-identification:**

```
[00:00] Agent starts, explores data
[02:00] Consults Oracle for strategy
[05:00] Writes train.py with batch_size=32  ← PROBLEM
[07:00] Launches training in background
[08:00] Writes predict.py
[10:00] Monitors training...
[15:00] Still training... (only 10% GPU util)
[20:00] Still training... (still slow)
[25:00] Still training... (realizes it's slow)
[30:00] Training finally finishes fold 1/5
[35:00] ⏱️ TIMEOUT - no submission generated
```

**Result:** ❌ No submission, wasted 35 minutes

---

## After This Fix

**Expected execution with aggressive GPU optimization:**

```
[00:00] Agent starts, explores data
[02:00] Consults Oracle for strategy
[05:00] Writes train.py with batch_size=128  ← FIXED: Explicit default
[05:30] Oracle reviews code: "batch_size looks good (128), proceed"
[06:00] Launches training: python -u train.py (background)
[06:01] Training prints:
        RESOURCES: 32 CPU cores, batch=128, GPU=NVIDIA A10, Mixed Precision=ON
        GPU Memory: 24.0 GB
        DataLoader: num_workers=10, prefetch_factor=4, persistent_workers=True
[06:15] Training prints:
        GPU Memory Used: 18.3 GB / 24.0 GB (76.3%)  ← GOOD
        VALIDATION: If <50% memory, batch_size=128 is TOO SMALL - should be 256+
[07:00] Agent checks GPU usage (60s checkpoint):  ← NEW CHECKPOINT
        ✅ GPU memory 76% → Good, proceed
[07:01] Agent writes predict.py immediately
[08:00] Training progressing: Fold 1/5, Epoch 5/10
[09:00] Training progressing: Fold 2/5, Epoch 3/10
[10:00] Training progressing: Fold 3/5, Epoch 8/10
[11:00] Training complete (all 5 folds done)
[11:30] Running predict.py
[12:00] ✅ Submission generated: submission.csv
```

**Result:** ✅ Submission generated in 12 minutes (vs 35+ min timeout)

---

## Alternative Scenario: GPU Still Underutilized

**If agent somehow still uses small batch:**

```
[06:00] Launches training with batch_size=64  ← Still too small
[06:01] Training prints:
        GPU Memory Used: 8.2 GB / 24.0 GB (34%)  ← PROBLEM DETECTED
[07:00] Agent checks GPU usage (60s checkpoint):
        ❌ GPU memory 34% < 50% threshold
        → Agent kills training immediately
        → Agent edits train.py: BATCH_SIZE = 128 (doubled)
        → Agent relaunches training
[07:30] Training prints:
        GPU Memory Used: 18.3 GB / 24.0 GB (76.3%)  ← FIXED
[08:00] Agent checks GPU usage again:
        ✅ GPU memory 76% → Good, proceed
[continues normally from here...]
```

**Result:** ✅ Self-corrects within 60 seconds, doesn't waste 25 minutes

---

## Key Differences to Watch For

### 1. Initial Code Writing

**Before:**
```python
BATCH_SIZE = 32  # Conservative choice
```

**After:**
```python
BATCH_SIZE = 128  # Start here for A10 24GB (from prompt example)
```

### 2. Oracle Review

**Before:**
```
Oracle review focuses on: data leakage, model architecture, metrics
```

**After:**
```
Oracle explicitly checks: "batch_size is 128 ✓ (not 32), GPU usage will be good"
```

### 3. GPU Monitoring Prints

**Before (missing):**
```
Fold 1/5, Epoch 1/10, Loss: 0.542
```

**After (comprehensive):**
```
RESOURCES: 32 CPU cores, batch=128, GPU=NVIDIA A10, Mixed Precision=ON
GPU Memory: 24.0 GB
DataLoader: num_workers=10, prefetch_factor=4, persistent_workers=True
GPU Memory Used: 18.3 GB / 24.0 GB (76.3%)
VALIDATION: If <50% memory, batch_size=128 is TOO SMALL - should be 256+
Fold 1/5, Epoch 1/10, Loss: 0.542
```

### 4. 60-Second Checkpoint

**Before (missing):**
```
[07:00] Writes predict.py
[08:00] Continues monitoring training passively
```

**After (active verification):**
```
[07:00] Reads training output with ReadBashOutput
[07:00] Parses GPU memory usage: 76.3%
[07:00] ✓ GPU memory >50%, batch size good, proceed
[07:01] Writes predict.py
```

### 5. Training Speed

**Before:**
```
Fold 1/5: 30 min (estimated 150 min total)
Fold 2/5: [times out before completion]
```

**After:**
```
Fold 1/5: 2.5 min (estimated 12.5 min total)
Fold 2/5: 2.4 min
Fold 3/5: 2.5 min
Fold 4/5: 2.6 min
Fold 5/5: 2.5 min
Total: 12.5 min ✓
```

---

## Validation Checklist

When reviewing next agent run, check:

### ✅ Code Writing Phase
- [ ] train.py has `BATCH_SIZE = 128` (or higher) for images
- [ ] train.py has `BATCH_SIZE = 4096+` for tabular data
- [ ] train.py has `num_workers=8-12` in DataLoader
- [ ] train.py has `prefetch_factor=3-4` in DataLoader
- [ ] train.py has `persistent_workers=True` in DataLoader
- [ ] train.py has mixed precision (autocast/GradScaler)

### ✅ Oracle Review
- [ ] Oracle review mentions checking batch size
- [ ] Oracle doesn't flag batch size as too small
- [ ] Oracle approves GPU configuration

### ✅ Training Launch
- [ ] Training prints "RESOURCES: ... batch=128 ..."
- [ ] Training prints "GPU Memory: X.X GB"
- [ ] Training prints "DataLoader: num_workers=10, prefetch_factor=4, persistent_workers=True"

### ✅ GPU Monitoring (First Batch)
- [ ] Training prints "GPU Memory Used: XX.X GB / YY.Y GB (ZZ%)"
- [ ] GPU usage shows >50% (ideally 70-80%)
- [ ] Training prints validation message about batch size

### ✅ 60-Second Checkpoint
- [ ] Agent uses ReadBashOutput after ~60 seconds
- [ ] Agent parses GPU memory usage from output
- [ ] Agent confirms GPU memory >50% before proceeding
- [ ] (If <50%: Agent kills training, doubles batch size, relaunches)

### ✅ Training Progress
- [ ] Training completes in 5-15 minutes (not 25+)
- [ ] All CV folds complete (or gracefully stopped with partial models)
- [ ] predict.py executes successfully
- [ ] submission.csv generated

---

## Success Metrics

| Metric | Before | Target After | How to Measure |
|--------|--------|--------------|----------------|
| Batch Size | 32 | 128+ | Check train.py source code |
| GPU Memory | 1-5% | 70-80% | Check GPU monitoring prints |
| GPU Utilization | 10-20% | 80-95% | Run `nvidia-smi` during training |
| Training Time | 25+ min | 5-12 min | Time from launch to completion |
| Submission Generated | ❌ | ✅ | Check for submission.csv file |
| Agent Restarts | 0 | 0-1 | Count training kills/relaunches |

---

## Common Issues and Resolutions

### Issue 1: Agent still uses batch_size=32

**Symptom:** train.py has `BATCH_SIZE = 32`

**Cause:** Prompt not explicit enough or LLM ignored guidance

**Resolution:**
1. Check that kaggle_agent.py line 108 says "Start with batch_size=128 (NOT 32!)"
2. Check that Oracle review (line 125) mentions batch size validation
3. May need to make example code even more explicit

### Issue 2: No GPU monitoring prints

**Symptom:** Training output doesn't show "GPU Memory Used: ..."

**Cause:** Agent didn't add monitoring code to train.py

**Resolution:**
1. 60-second checkpoint should catch this
2. Agent should kill training and add monitoring code
3. Check prompt lines 257-267 are clear enough

### Issue 3: Agent doesn't kill slow training

**Symptom:** GPU memory <50%, but agent doesn't restart

**Cause:** Agent isn't correctly parsing GPU memory prints

**Resolution:**
1. Check agent logs for ReadBashOutput at ~60 seconds
2. Verify agent code can parse "GPU Memory Used: X.X GB / Y.Y GB (Z%)" format
3. May need to make parsing more robust

---

## Next Steps After Successful Run

1. **Document actual performance:**
   - Record actual batch size used
   - Record actual GPU memory %
   - Record actual training time
   - Compare to these predictions

2. **Fine-tune if needed:**
   - If GPU memory 50-60%: Increase batch size recommendation to 160-192
   - If GPU memory >90%: Might be pushing OOM risk, keep at 128
   - If training still slow: Check DataLoader workers, disk I/O

3. **Test on multiple competitions:**
   - Image classification (current)
   - Tabular data (batch_size=4096 guidance)
   - NLP (batch_size=64-128 guidance)
   - Verify all work as expected

---

**Summary:** Next run should show `batch_size=128`, 70-80% GPU memory usage, 5-12 minute training time, and successful submission generation. The 60-second checkpoint will catch any issues early and force a restart with corrected batch size.
