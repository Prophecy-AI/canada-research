# Agent Failure Analysis: dog-breed-identification

## 🔴 What Went Wrong

### The Problem
```
[2025-10-16 22:20:23,989] Invalid submission file: None. Please check that the file exists and it is a CSV.
```

**Result:**
- ❌ No submission created
- ❌ Training incomplete (killed at epoch 13/30, fold 1/5)
- ❌ Agent scored 0 points

---

## 🔍 Root Cause Analysis

### Timeline of Events

```
22:09:05 - Agent starts
22:11:34 - Training launches (background process bash_5865a2cf)
22:11:34 to 22:20:13 - Agent monitors training repeatedly
22:20:13 - Container TIMEOUT (667 seconds = 11 minutes)
22:20:23 - Grading: No submission found
```

### What the Agent Did Wrong

**1. Inefficient Monitoring Pattern**
```
Turn 26: sleep 15  → ReadBashOutput  (waited 15s)
Turn 27: sleep 60  → ReadBashOutput  (waited 60s)
Turn 28: sleep 120 → TIMEOUT!        (tried to wait 120s but killed)
Turn 29: sleep 120 → TIMEOUT!        (tried again)
Turn 30-35: More polling...
```

The agent spent **10+ minutes just waiting and checking logs** instead of working!

**2. Never Created predict.py**

The agent should have:
1. ✅ Write train.py (DONE)
2. ✅ Launch training (DONE)
3. ❌ **Immediately write predict.py** (NEVER HAPPENED!)
4. ❌ Wait for training to finish
5. ❌ Run predict.py

Instead, it got stuck in a loop:
```
while training_running:
    sleep_and_check_logs()
    sleep_and_check_logs()
    sleep_and_check_logs()
    # ... container killed before predict.py created
```

**3. Training Too Long for Container Timeout**

Training plan: **5-fold CV × 30 epochs**
- Estimated time: 75-90 minutes
- Container timeout: **11 minutes**
- Math doesn't work! ❌

---

## 📊 Evidence from Logs

### Training Progress When Killed
```
Epoch 13/30 for Fold 1/5
Train Acc: 99.33%
Val Acc: 85.60%
Val Loss: 0.5884 (best)
```

Training was going well but nowhere near done:
- Completed: 13/30 epochs × 1/5 folds = **8.7% progress**
- Remaining: ~70 minutes
- Time left: 0 minutes (container killed)

### Agent Behavior Pattern
```
[22:12:57] → Bash(sleep 15)           # Wasted 15s
[22:14:05] → Bash(sleep 60)           # Wasted 60s
[22:16:15] → Bash(sleep 120) TIMEOUT  # Wasted 120s (failed)
[22:18:03] → Bash(sleep 120) TIMEOUT  # Wasted 120s (failed)
[22:20:13] Container killed
```

The agent treated this like a patient monitoring task, not realizing:
- **It has other work to do (predict.py!)**
- **Time is limited**
- **Training takes forever**

---

## ✅ Fix Applied

### Updated System Prompt (kaggle_agent.py:111-117)

**Before:**
```markdown
8) **Execute**
   • Launch training in background
   • Monitor via ReadBashOutput every ≤30s
   • Keep training in train.py, inference in predict.py
```

**After:**
```markdown
8) **Execute**
   • **CRITICAL WORKFLOW - PARALLEL EXECUTION:**
     1. Write train.py and validate with Oracle
     2. Launch training: Bash(command="python -u train.py", background=true)
     3. **IMMEDIATELY (same turn) write predict.py** - DO NOT wait for training
     4. Validate predict.py with Oracle if needed
     5. Monitor training progress occasionally (every 60-120s, not more frequent)
     6. When training completes, run predict.py to generate submission
```

### Key Changes

**1. Explicit Parallel Workflow**
- ✅ Makes it clear: **write predict.py IMMEDIATELY**
- ✅ Don't wait for training
- ✅ Work in parallel

**2. Reduced Monitoring Frequency**
- Before: "every ≤30s" → Agent did 15s, 60s, 120s (wasteful)
- After: "every 60-120s" → Less frequent polling
- Why: Training takes 30-60s per epoch, checking every 15s is overkill

**3. Prioritization**
- Before: Implicit workflow (agent guessed wrong)
- After: Numbered steps (crystal clear)

---

## 🎯 Expected Behavior After Fix

### Ideal Timeline (Future Runs)

```
Turn 1-10: Data exploration, Oracle consultation, planning
Turn 11: Write train.py
Turn 12: Launch training (background)
Turn 12: IMMEDIATELY write predict.py (same turn or next)
Turn 13-15: Validate predict.py, make sure it works
Turn 16: Check training (1st check after 90s)
Turn 17: Check training (2nd check after 120s)
...
Turn N: Training complete
Turn N+1: Run predict.py → submission.csv
Turn N+2: Done!
```

**Key difference:** predict.py exists **BEFORE** training finishes!

### Why This Matters

**Scenario: Training fails at 90% completion**
- ❌ Old behavior: No submission (wasted everything)
- ✅ New behavior: Can still use partially trained models if predict.py exists

**Scenario: Container timeout**
- ❌ Old behavior: No submission (total failure)
- ✅ New behavior: predict.py exists, can use best checkpoint saved

---

## 💡 Additional Recommendations

### 1. **Reduce Training Scope for First Run**

Instead of 5-fold × 30 epochs, start with:
- **1-fold × 10 epochs** (quick baseline)
- Get submission working first
- Then scale up if time permits

**Update train.py generation prompt:**
```python
# Quick baseline config
N_FOLDS = 1      # Start with single fold
EPOCHS = 10      # Quick iteration
# Can increase later if time permits
```

### 2. **Early Stopping**

Add to training script:
```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,     # Stop if no improvement for 3 epochs
    restore_best_weights=True
)
```

This prevents wasting time on training that already converged.

### 3. **Checkpoint Strategy**

Save models more aggressively:
```python
# Save every epoch, not just best
checkpoint_callback = ModelCheckpoint(
    filepath='model_epoch_{epoch:02d}.pth',
    save_freq='epoch'
)
```

If container dies, can resume or use last checkpoint.

### 4. **predict.py Template Ready**

Have agent generate predict.py template IMMEDIATELY after train.py:
```python
# predict.py (template generated before training starts)
import torch
import pandas as pd

# Load best model
model = torch.load('best_model.pth')
model.eval()

# Load test data
test_data = ...

# Generate predictions
predictions = model(test_data)

# Save submission
submission.to_csv('/home/submission/submission.csv', index=False)
print("✅ Submission saved!")
```

This can be refined later if needed, but **exists immediately**.

---

## 🔬 Testing the Fix

### Test Plan

**1. Run same competition again:**
```bash
cd /Users/Yifan/canada-research/mle-bench
python run_agent.py agent_v5_kaggle dog-breed-identification
```

**2. Check critical turns:**
```
Turn where train.py launched: X
Turn where predict.py created: X+0 or X+1 (NOT X+20!)
Turn where training complete: Y
Turn where submission created: Y+1
```

**3. Validate timeline:**
- predict.py created: Within 2 minutes of train.py launch
- Monitoring frequency: 60-120s intervals
- Total runtime: <15 minutes (should complete or fail gracefully)

### Success Criteria

✅ predict.py exists before training finishes
✅ Submission.csv created (even if partial/bad quality)
✅ Agent doesn't waste >2 minutes just sleeping
✅ Grading report shows "valid_submission": true

---

## 📝 Summary

### What Broke
1. ❌ Agent monitored training obsessively (wasted time)
2. ❌ Never created predict.py (critical omission)
3. ❌ Container timeout killed incomplete work

### How Fixed
1. ✅ Explicit parallel workflow in prompt
2. ✅ "IMMEDIATELY write predict.py" instruction
3. ✅ Reduced monitoring frequency (60-120s)

### Expected Outcome
- ✅ predict.py created early
- ✅ More efficient use of time
- ✅ Submission generated even if training incomplete
- ✅ Better success rate overall

---

**Fix applied to:** `/Users/Yifan/canada-research/mle-bench/agents/agent_v5_kaggle/kaggle_agent.py:111-117`

**Status:** ✅ Ready for testing

**Next action:** Re-run competition to validate fix
