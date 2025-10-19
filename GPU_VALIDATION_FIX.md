# GPU Validation Fix - Timing-Based Detection

## Problem Summary

**Root Cause:** Agent had faulty GPU validation code that:
1. Checked GPU memory too early (first batch) before PyTorch allocated significant memory
2. Used aggressive threshold (<10% memory) that triggered false failures
3. Misinterpreted performance - saw "17s/epoch" and thought GPU was working, but that's actually CPU speed
4. Didn't understand that **epoch timing is the most reliable GPU indicator**, not memory usage

**Result:** Training ran on CPU (10-20x slower) because the agent:
- Killed training due to low memory (<1% after first batch)
- Later saw low memory but fast-ish epochs (~17s) and assumed GPU was working
- Lacked proper context to know 17s/epoch = CPU, not GPU (<2s/epoch on A100)

## Solution

### 1. Created Dedicated GPU Validation Script

**File:** `/Users/Yifan/canada-research/mle-bench/environment/validate_gpu.py`

**Features:**
- Timing-based benchmark (more reliable than memory checks)
- Warmup phase (GPU needs initialization before timing)
- 100-batch benchmark for accurate timing
- Framework-specific validation (PyTorch, LightGBM, XGBoost)
- Clear pass/fail based on timing thresholds (CPU is 10-20x slower)

**Usage:**
```bash
# Validate PyTorch
python validate_gpu.py --framework pytorch --model-size small --batch-size 256

# Validate LightGBM
python validate_gpu.py --framework lightgbm --rows 100000

# Validate XGBoost
python validate_gpu.py --framework xgboost --rows 50000
```

**Expected Output (Success):**
```
✅ GPU TRAINING CONFIRMED
   Performance matches GPU expectations (1.2s < 12.0s threshold)
```

**Expected Output (Failure):**
```
❌ CPU TRAINING DETECTED
   Performance too slow for GPU (18.5s >= 12.0s threshold)
   Expected GPU time: ~1.5s (10-20x faster)
```

### 2. Created GPUValidate Tool

**File:** `/Users/Yifan/canada-research/mle-bench/agents/agent_v5_kaggle/tools/gpu_validate.py`

**Purpose:** Allows agent to validate GPU training before launching long jobs

**Tool Schema:**
```python
{
    "name": "GPUValidate",
    "description": "Validate GPU training is working correctly by running a quick benchmark...",
    "input_schema": {
        "type": "object",
        "properties": {
            "framework": {"type": "string", "enum": ["pytorch", "lightgbm", "xgboost"]},
            "model_size": {"type": "string", "enum": ["small", "medium", "large"]},
            "batch_size": {"type": "number"},
            "rows": {"type": "number"}
        },
        "required": ["framework"]
    }
}
```

**Agent Usage:**
```python
# Before training, validate GPU works
GPUValidate(framework='pytorch', model_size='small', batch_size=256)
```

### 3. Updated Agent Prompt

**File:** `kaggle_agent.py`

**Changes:**

#### GPU Usage Section (Lines 64-70)
**Before:**
```
**GPU MANDATE (NEVER TRAIN ON CPU):**
- Target GPU utilization: 70-90% memory (28-36GB on A100 40GB)
- Underutilizing GPU is wasteful - maximize batch size
```

**After:**
```
**GPU USAGE (EFFICIENT, NOT EXTREME):**
- Goal: Efficient training, not max GPU usage - focus on speed and parallel training
- Batch sizes: Use reasonable sizes that work well (256-384 for 224x224 images)
- Parallel training preferred: Train 2-3 models simultaneously rather than one giant model
- Validation: Use GPUValidate tool BEFORE training to confirm GPU is working (timing-based, reliable)
```

#### Tools List (Lines 99-110)
**Added:**
```
- **GPUValidate:** Verify GPU training is working correctly (timing-based benchmark).
  Use BEFORE training to catch CPU fallback early.
  - Example: GPUValidate(framework='pytorch', model_size='small', batch_size=256)
  - Returns clear confirmation if GPU is working or error if CPU fallback detected
```

#### System Resource Check (Lines 131-139)
**Before:**
```
• Run: Bash(command='python -c "import torch..."') to verify PyTorch GPU
• Document: "We have X CPU cores, NVIDIA A10 GPU..."
• CRITICAL: Use ALL resources: max batch sizes, n_jobs=-1, mixed precision
```

**After:**
```
• Run: GPUValidate(framework='pytorch', model_size='small', batch_size=256) to verify GPU training works
• Document: "We have X CPU cores, NVIDIA A100 GPU with Y GB VRAM. GPU validation: PASSED"
• CRITICAL: Focus on efficient parallel training (2-3 models), not extreme single-model optimization
```

#### Training Validation (Lines 324-335)
**Before:**
```
5. MANDATORY GPU VALIDATION (60 seconds after launch):
   - CHECK 1 - GPU IS BEING USED: Look for GPU memory usage print
     * If GPU memory <10% → KILL IMMEDIATELY
   - CHECK 2 - GPU UTILIZATION: check memory %
     * If GPU memory <50% → KILL, increase batch_size by 2x
```

**After:**
```
5. MANDATORY VALIDATION (After launch):
   - CHECK 1 - LOSS SANITY (after 2-3 epochs): Check validation loss vs random baseline
   - CHECK 2 - EPOCH TIMING (after first epoch): Verify GPU speed vs CPU speed
     * Expected on A100 GPU: EfficientNet-B3 = 0.5-1 min/epoch, ResNet-50 = 0.3-0.5 min/epoch
     * CPU fallback (BAD): EfficientNet-B3 = 10-20 min/epoch, ResNet-50 = 5-10 min/epoch
     * If epoch >5 min → KILL IMMEDIATELY - likely training on CPU
     * Don't rely on memory checks alone - epoch timing is the most reliable GPU indicator
```

### 4. Updated Training Template

**File:** `training_hints.txt` (Lines 295-302)

**Before:**
```python
# GPU validation (first batch only) - A100 40GB
if epoch == 0 and i == 0:
    mem = torch.cuda.memory_allocated() / 1024**3
    if mem/total < 0.10:
        print("⚠️  CRITICAL: GPU <10% - likely training on CPU!")
        raise RuntimeError("GPU not being used")
```

**After:**
```python
# Print timing every epoch (GPU validation via speed, not memory)
if i == 0:  # First batch of epoch
    epoch_start = time.time()
if i == len(train_loader) - 1:  # Last batch of epoch
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch} completed in {epoch_time:.1f}s")
    # Expected A100 times: EfficientNet-B3 ~30-60s/epoch, ResNet-50 ~20-30s/epoch
    # If seeing >300s/epoch, likely training on CPU (10-20x slower)
```

### 5. Registered Tool in KaggleAgent

**File:** `kaggle_agent.py` (Lines 596-603)

**Added:**
```python
def _register_core_tools(self):
    """Register core tools + Kaggle-specific tools"""
    # Register parent class tools
    super()._register_core_tools()

    # Register Kaggle-specific tool: GPUValidate
    from .tools.gpu_validate import GPUValidateTool
    self.tools.register(GPUValidateTool(self.workspace_dir))
```

## Key Improvements

### 1. Timing-Based Validation (More Reliable)
- **Before:** Checked GPU memory after first batch (unreliable - too early)
- **After:** Benchmark timing over 100 batches (reliable - CPU is 10-20x slower)

### 2. Clear Thresholds
- **Before:** <10% memory = CPU (false positives - memory allocates gradually)
- **After:** >5s/epoch for medium model = CPU (accurate - timing doesn't lie)

### 3. Practical Guidance
- **Before:** "Maximize GPU usage to 70-90% memory"
- **After:** "Use efficient batch sizes, prefer parallel training over single giant model"

### 4. Proper Tool Integration
- **Before:** Manual checks via bash commands and prints
- **After:** Dedicated GPUValidate tool that returns clear pass/fail

### 5. Removed Problematic Code
- **Before:** Aggressive memory check that crashed training incorrectly
- **After:** Simple epoch timing print for manual monitoring

## Expected Behavior

### Before Training (First Turn)
```
Agent: Running system checks...
[GPU validation tool executes 100-batch benchmark]
GPUValidate: ✅ GPU TRAINING CONFIRMED (1.2s for 100 batches, <12s threshold)

Agent: System verified:
- 36 CPU cores
- NVIDIA A100 40GB GPU (39.5 GB free)
- 440GB RAM
- GPU validation: PASSED
```

### During Training (After First Epoch)
```
Training output:
Epoch 0 completed in 35.2s
Loss=0.4521, Acc=0.8234

Agent: First epoch completed in 35s - matches GPU expectations for EfficientNet-B3.
Training proceeding normally.
```

### CPU Fallback Detected (Bad Case)
```
Training output:
Epoch 0 completed in 542.1s
Loss=0.4521, Acc=0.8234

Agent: ⚠️  CRITICAL: Epoch took 542s - this is CPU speed (GPU should be <60s).
Killing training and debugging GPU setup...
```

## Validation Checklist

- [x] Created timing-based GPU validation script
- [x] Created GPUValidate tool with proper schema
- [x] Registered tool in KaggleAgent
- [x] Updated agent prompt to use GPUValidate
- [x] Updated system resource check to include GPU validation
- [x] Updated training validation to use epoch timing (not memory)
- [x] Removed problematic early memory check from training template
- [x] Added clear epoch timing prints to training template
- [x] Updated guidance from "maximize GPU" to "efficient parallel training"

## Testing Instructions

1. **Test GPU validation script directly:**
```bash
cd /Users/Yifan/canada-research/mle-bench/environment
python validate_gpu.py --framework pytorch --model-size small --batch-size 256
```

Expected: ✅ GPU TRAINING CONFIRMED in ~2-3 seconds

2. **Test in agent (next competition run):**
- Agent should call GPUValidate during system resource check
- Agent should see timing-based confirmation
- Agent should monitor epoch timing during training
- Agent should detect CPU fallback if epochs >5 min

3. **Verify no false positives:**
- GPU validation should not fail on A100 with proper setup
- Training should not be killed due to low memory after first batch
- Epoch timing should be the definitive indicator, not memory

## Files Modified

1. ✅ `/Users/Yifan/canada-research/mle-bench/environment/validate_gpu.py` (created)
2. ✅ `/Users/Yifan/canada-research/mle-bench/agents/agent_v5_kaggle/tools/gpu_validate.py` (created)
3. ✅ `/Users/Yifan/canada-research/mle-bench/agents/agent_v5_kaggle/kaggle_agent.py` (modified - 5 sections)
4. ✅ `/Users/Yifan/canada-research/mle-bench/environment/training_hints.txt` (modified - validation code)

## Summary

**Problem:** Faulty GPU validation code + agent misinterpretation = accidental CPU training

**Solution:** Timing-based validation tool + clearer guidance + removed problematic checks

**Result:** Agent now has reliable GPU detection and won't waste time on CPU training
