# A100 40GB Upgrade - Changes Summary

## Completed: Agent Prompts Updated for A100 40GB GPU

**Date:** 2025-10-17
**Hardware Change:** A10 24GB → A100 40GB (1.67x more VRAM, 2x faster training)

---

## Changes Made (4 Files, 15 Strategic Edits)

### 1. **kaggle_agent.py** - Main System Prompt

**Hardware Specs (Lines 49-54):**
```
BEFORE: A10 24GB, target 17-22GB (70-90%)
AFTER:  A100 40GB, target 28-36GB (70-90%)

Added note: "A100 is 2x faster than A10 - enables larger models (B4/B5 vs B2/B3) or 3-4 parallel medium models"
```

**GPU Mandate (Lines 64-68):**
```
Batch sizes updated:
- 256-384 for 224x224 images (was 128-192)
- 128-192 for 384x384 (was 64-96)
- 8192+ for tabular (was 4096+)
- Target: 28-36GB of 40GB
```

**Domain Architectures (Lines 76-78):**
```
Image Classification: EfficientNet-B3/B4 (20-30 min on A100), B5/B6 (40-60 min)
  Note: "A100 trains B4 as fast as A10 trained B2"
Segmentation: U-Net + EfficientNet-B3/B4 backbone, 512x512 tiles
```

**A100 Strategy Guidance (Lines 189-193):**
```
Added principles for Oracle and Agent:
- "A100 enables larger models (B4/B5) or 3-4 parallel medium models"
- "Prefer quality over speed: use B4/B5 if they bring >2-3% score boost"
- "For exploration: run 3-4 smaller models in parallel for faster feedback"
- "Don't use larger models just because you can - validate cost-effectiveness"
```

---

### 2. **kaggle_competition_strategy.txt** - Strategy Playbook

**Image Classification Section (Lines 161-175):**
```
BEFORE: EfficientNet-B0 or ResNet-34
AFTER:  EfficientNet-B3 or ResNet-50

Strategy updates:
- Batch size: 256-384 for 224x224 (A100 can use 2x larger batches)
- Train: 6-8 epochs with early stopping
- Note: "A100 40GB trains B3 as fast as A10 trained B0"
```

**Image Segmentation Section (Lines 177-190):**
```
BEFORE: U-Net + EfficientNet-B0/ResNet-34 backbone
AFTER:  U-Net + EfficientNet-B3/ResNet-50 backbone

Strategy:
- Tiles: 256x256 or 512x512 (larger tiles possible)
- Batch size: 32-64 for 256x256, 16-32 for 512x512
- Train: 8-12 epochs
```

**Parallel Training Section (Lines 285-309):**
```
Resource allocation updated for A100 40GB:

Option 1 (Large models):
  - Model 1 (CPU): LightGBM, 12 cores, 0% GPU
  - Model 2 (GPU): EfficientNet-B4, batch=192, ~18GB GPU
  - Model 3 (GPU): ResNet-50, batch=256, ~16GB GPU

Option 2 (More parallelism):
  - 3x EfficientNet-B3, batch=256 each, ~12GB GPU each

Example updated:
"3 models in parallel on A100 complete in ~10-12 min (vs 15-20 min on A10)"
"A100 enables B4 models at B2 speed"
```

---

### 3. **training_hints.txt** - Training Failure Prevention

**Model Sizing Guide (Lines 17-36):**
```
Time estimates updated for A100:

Time Budget | Recommended Model (A100)  | Expected Time
20-30 min   | EfficientNet-B4, B5       | ~20-25 min  (was B2/B3)
30-40 min   | EfficientNet-B5, B6       | ~30-35 min  (was B3/B4)
40-60 min   | EfficientNet-B6, B7       | ~45-55 min  (was B4/B5)

Time formula updated:
- B3 = 0.5 min/epoch (was 1 min on A10)
- B4 = 1 min/epoch (was 2-3 min on A10)
- B5 = 2 min/epoch (was 4-5 min on A10)
```

**Batch Size Section (Lines 94-102):**
```
BEFORE:
- Images 224x224: batch_size=128-192 (A10)
- Target: 17-22GB of 24GB

AFTER:
- Images 224x224: batch_size=256-384 (A100)
- Images 384x384: batch_size=128-192
- Images 512x512: batch_size=64-96
- Tabular: batch_size=8192+
- Target: 28-36GB of 40GB (70-90%)
```

**Training Template (Lines 254-300):**
```
Config updated:
- BATCH_SIZE = 256 (was 128)
- EPOCHS = 8-10 for 20-30 min (was 6-8)
- GPU validation: "Expected on A100 40GB: 28-36GB (70-90%)"
```

---

### 4. **memory/competition_memory.py** - Memory Patterns

**Image Classification Patterns:**
```
small_dataset:
  - best_models: ["EfficientNet-B3", "ResNet-50", "EfficientNet-B2"]
  - typical_time_min: "6-10" (was "8-12")
  - batch_size: "256-384 for 224x224 (A100 40GB)"
  - note: "A100 trains B3 as fast as A10 trained B0 (2x speedup)"

medium_dataset:
  - best_models: ["EfficientNet-B4", "EfficientNet-B5", "ResNet-50"]
  - typical_time_min: "12-18" (was "15-25")
  - batch_size: "256-384 for 224x224, 128-192 for 256x256"
  - advanced_techniques: "Parallel training: 3x B3 or 2x B4 simultaneously"
  - note: "A100 enables B4/B5 at same speed as A10 trained B2/B3"
```

---

## Impact Summary

### What A100 40GB Enables:

**1. Larger Models in Same Time**
- 20-30 min budget: Can use B4/B5 (was B2/B3)
- 30-40 min budget: Can use B5/B6 (was B3/B4)
- 2x faster training than A10

**2. More Parallel Training**
- A10: 2-3 small models (B0/B2)
- A100: 3-4 medium models (B3/B4) or 2 large models (B4/B5)

**3. Larger Batch Sizes → Better Convergence**
- Images 224x224: 256-384 (was 128-192) = 2x larger
- Images 384x384: 128-192 (was 64-96) = 2x larger
- Tabular: 8192+ (was 4096+) = 2x larger
- Larger batches = more stable gradients, faster convergence

**4. Faster Feedback Loops**
- Run 3-4 experiments in parallel
- B4 completes in time B2 used to take
- More iterations per hour = better exploration

---

## Strategy Principles (Oracle & Agent)

### Balanced Approach:
1. **Use larger models when cost-effective**
   - Prefer B4/B5 if they bring >2-3% score improvement
   - Don't use B6+ unless competition clearly benefits from it

2. **Parallel training for exploration**
   - Run 3-4 smaller models (B3) in parallel for diverse ensemble
   - OR 2 larger models (B4) for quality + diversity balance

3. **Cost-effectiveness first**
   - Don't use larger models just because you can
   - Validate with Oracle that model choice makes sense
   - Consider: will B5 really beat B4 by enough to justify extra complexity?

4. **Encourage experimentation**
   - A100 enables more experiments in same time
   - Run multiple approaches in parallel
   - Faster feedback = better learning

---

## Validation After Changes

**Test checklist:**
- [ ] GPU memory usage reaches 28-36GB (70-90%)
- [ ] Batch sizes are 2x larger than before (256-384 for 224x224)
- [ ] Training B4 completes in ~20-25 min (3 folds, 8 epochs)
- [ ] Can run 3x B3 models in parallel (~36GB total)
- [ ] Agent recommends B4/B5 for 20-30 min budget (not B2/B3)

**Expected improvements:**
- ✅ Better model quality (B4 vs B2)
- ✅ Faster iteration (2x speedup)
- ✅ More parallelism (3-4 models)
- ✅ Better convergence (2x batch sizes)
- ✅ More exploration (parallel experiments)

---

## Files Modified:
1. `/Users/Yifan/canada-research/mle-bench/agents/agent_v5_kaggle/kaggle_agent.py` - Main prompt
2. `/Users/Yifan/canada-research/mle-bench/environment/kaggle_competition_strategy.txt` - Strategy playbook
3. `/Users/Yifan/canada-research/mle-bench/environment/training_hints.txt` - Training tips
4. `/Users/Yifan/canada-research/mle-bench/agents/agent_v5_kaggle/memory/competition_memory.py` - Memory patterns

**Total edits:** 15 strategic changes
**Approach:** Focused and minimal - only essential updates for A100 capabilities

---

**Status:** ✅ COMPLETE
**Ready for testing:** YES
**Next step:** Run a test competition to validate GPU usage and model sizing
