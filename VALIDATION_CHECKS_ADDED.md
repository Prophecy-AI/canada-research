# Critical Validation Checks Added

**Date:** 2025-10-17
**Commit:** `0eb5e5f`
**Status:** ✅ Complete

---

## Problem Analysis

### Failure Case 1: CPU Training (51+ minutes)
**What happened:**
- Library conflict → switched from albumentations to torchvision
- Subtle bug: tensors never moved to GPU (silent failure)
- Agent saw 1% GPU usage but misinterpreted as "batch size too small"
- Trained on CPU for 51 minutes
- Validation loss stayed at random-guess baseline (4.7 for 120 classes ≈ ln(120) = 4.79)
- Agent only discovered issue after submission when consulting Oracle

**Root cause:** No validation that GPU was actually being used

### Failure Case 2: Wrong Model (9 minutes)
**What happened:**
- Oracle recommended `efficientnetv2_s` (good choice)
- Model not available offline → crash
- Agent switched to `efficientnetv2_rw_s.ra2_in1k` (weaker model) without consulting Oracle
- Training succeeded but with inferior model → poor score

**Root cause:** No validation that chosen model exists offline

---

## Solution: 4 Critical Validation Checks

### 1. GPU Validation Check ✅

**Location:** Agent prompt, step 5 "MANDATORY GPU VALIDATION"

**What it does:**
- **CHECK 1:** Verify GPU is actually being used (>10% GPU memory)
  - If <10% → KILL immediately (training on CPU, 10-100x slower)
  - This is the #1 failure mode

- **CHECK 2:** Verify GPU utilization (>50% GPU memory)
  - If <50% → KILL, increase batch_size by 2x

- **CHECK 3:** Verify loss sanity after 1-2 epochs
  - Calculate random baseline: `ln(num_classes)`
  - If validation loss ≈ baseline (within 0.1) → model not learning, KILL and debug

**Example:**
```
120-class problem: random baseline = ln(120) ≈ 4.79
If validation loss = 4.7 after 2 epochs → model guessing randomly → KILL
```

**Impact:** Catches CPU training in <1 min (vs 51 min), catches non-learning models early

---

### 2. Model Availability Check ✅

**Location:** Agent prompt, step 4 "VALIDATE MODEL AVAILABILITY"

**What it does:**
- Before committing to strategy with pretrained model, verify it exists offline
- Command: `python -c "import timm; print('MODEL_NAME' in timm.list_models())"`
- If model not available → ask Oracle for alternative (don't silently substitute)

**Impact:** Prevents switching to weaker models without Oracle consultation

---

### 3. Loss Sanity Check in Monitoring ✅

**Location:** Agent prompt, step 9 "Oracle consultation format"

**What it does:**
- During Oracle consultations, agent MUST include:
  - **GPU validation:** "XX.X GB / 24.0 GB (ZZ%) - verify >10%"
  - **Loss validation:** "Validation loss = A.BC, random baseline = ln(num_classes) = D.EF"
- Oracle can immediately spot issues:
  - If GPU <10% → tell agent to KILL (running on CPU)
  - If loss ≈ baseline → tell agent to KILL (model not learning)

**Impact:** Oracle catches issues during monitoring, not just post-mortem

---

### 4. Oracle Monitoring Guidance ✅

**Location:** Oracle prompt, "MONITORING GUIDANCE (CRITICAL CHECKS)"

**What Oracle now checks:**
- **GPU validation:** If GPU <10% → IMMEDIATELY tell agent to KILL training
- **Loss sanity:** If validation loss ≈ ln(num_classes) after 2+ epochs → tell agent to KILL and debug
- **Time management:** If fold 1 took 12+ min → suggest reducing to 2 folds or 6 epochs

**Impact:** Oracle provides immediate actionable guidance when issues detected

---

## Changes Summary

### Files Modified

1. **`mle-bench/agents/agent_v5_kaggle/kaggle_agent.py`**
   - Step 4: Added model availability check (~3 lines)
   - Step 5: Enhanced GPU validation with 3 checks (~10 lines)
   - Step 9: Updated Oracle consultation format with mandatory metrics (~5 lines)

2. **`agent_v5/tools/oracle.py`**
   - Added monitoring guidance with critical checks (~3 lines)

**Total additions: ~21 lines (minimal, targeted)**

---

## How It Prevents Failures

### Failure Case 1: CPU Training
**Before:**
- Agent trains on CPU for 51 min
- Sees 1% GPU usage → misinterprets as batch size issue
- Only discovers after submission

**After:**
- 60 seconds after launch → CHECK 1 detects <10% GPU usage
- Agent KILLS immediately
- Debugs and fixes tensor.cuda() issue
- Relaunches with proper GPU usage
- **Time saved: 50+ minutes**

### Failure Case 2: Wrong Model
**Before:**
- Model crashes → agent silently switches to weaker model
- Continues without consulting Oracle

**After:**
- During planning → model availability check
- Detects `efficientnetv2_s` not available
- Agent asks Oracle for alternative
- Oracle recommends available model (e.g., `efficientnetv2_rw_t`)
- **Result: Correct model choice, better score**

---

## Validation Checklist

After these changes, agent will validate:

**Planning Phase:**
- [ ] Pretrained model exists offline (if applicable)
- [ ] Training time estimated and within budget

**Training Launch (60 sec):**
- [ ] GPU memory usage >10% (not training on CPU)
- [ ] GPU memory usage >50% (proper utilization)
- [ ] Validation loss improving beyond random baseline

**During Training (every 5-10 min):**
- [ ] GPU still being used (>10%)
- [ ] Loss still improving (not stuck at baseline)
- [ ] On track to finish within time budget

**Oracle Consultation:**
- [ ] Agent provides GPU validation metrics
- [ ] Agent provides loss vs baseline comparison
- [ ] Oracle checks for critical issues

---

## Expected Behavior Changes

### Training Launch
**Old:**
```
[09:31:37] Launching training...
[09:31:38] Training started
[09:32:00] Checking GPU... 1% memory used
[09:32:00] Note: GPU underutilized, but continuing
[... 51 minutes of CPU training ...]
```

**New:**
```
[09:31:37] Launching training...
[09:31:38] Training started
[09:32:00] CHECK 1 - GPU usage: 0.2 GB / 24.0 GB (0.8%)
[09:32:00] ⚠️  CRITICAL: GPU <10% - training on CPU!
[09:32:00] KILLING training immediately
[09:32:05] Analyzing code... found bug: tensors not moved to GPU
[09:32:30] Fixed: added data.cuda() in training loop
[09:32:35] Relaunching with fix...
[09:32:45] CHECK 1 - GPU usage: 18.2 GB / 24.0 GB (75%) ✓
```

### Model Selection
**Old:**
```
[09:29:17] Using model: efficientnetv2_s
[09:29:32] RuntimeError: Model weights not found
[09:29:51] Switching to: efficientnetv2_rw_s.ra2_in1k
[09:29:51] Training successfully started
```

**New:**
```
[09:28:00] Validating model availability...
[09:28:05] Checking: efficientnetv2_s... NOT FOUND offline
[09:28:10] Consulting Oracle for alternative...
[09:28:45] Oracle recommends: efficientnetv2_rw_t (similar to efficientnetv2_s)
[09:29:00] Validating: efficientnetv2_rw_t... FOUND ✓
[09:29:15] Using model: efficientnetv2_rw_t
[09:29:17] Training successfully started
```

---

## Testing Recommendations

### Test 1: GPU Validation
Create training script that doesn't move data to GPU, verify agent catches it:
```python
# Intentionally broken (no .cuda())
for batch in train_loader:
    data, target = batch  # Missing .cuda()
    output = model(data)
```
**Expected:** Agent kills training within 60 seconds

### Test 2: Loss Sanity
Create training script with wrong loss function (e.g., MSE for classification):
```python
criterion = nn.MSELoss()  # Wrong! Should be CrossEntropyLoss
```
**Expected:** Agent detects loss stuck at baseline, kills after 1-2 epochs

### Test 3: Model Availability
Specify non-existent model in strategy:
```
Model: efficientnetv2_fake
```
**Expected:** Agent detects model not available, asks Oracle for alternative

---

## Summary

**Added:** 4 critical validation checks (~21 lines)
**Impact:**
- Catches CPU training in <1 min (vs 51 min)
- Prevents wrong model substitutions
- Detects non-learning models early
- Enables Oracle to provide real-time guidance

**Key principle:** Validate early, fail fast, fix immediately

The agent will now catch these failure modes within minutes, not after wasting an hour! ✅

---

# Goal Refinement Changes (2025-10-17)

**Commit:** TBD
**Status:** ✅ Complete

## Objective

Shifted agent goal from "always chase gold medal" to "maximize ranking within time constraints" - emphasizing realistic outcomes (gold/silver/bronze) based on competition difficulty and time budget (20±10 min).

## Changes Made (6 targeted edits to kaggle_agent.py)

### 1. **Line 19 - Primary Objective** ✓
**Before:**
```
Your explicit objective is to deliver **gold-medal (top-1%) leaderboard performance** within the resource and time limits.
```

**After:**
```
Your explicit objective is to **maximize your ranking within time constraints (typically 20±10 min)** - achieving the best medal tier possible given the competition difficulty and time budget.
```

**Rationale:** Removes pressure to always achieve gold, emphasizes time as primary constraint.

---

### 2. **Lines 21-23 - Goal Setting Header** ✓
**Before:**
```
**REALISTIC GOAL SETTING (CRITICAL):**
- **Gold medal is the GOAL, but NOT always achievable** - some competitions are too hard for this setup
```

**After:**
```
**REALISTIC GOAL SETTING (CRITICAL):**
- **Maximize ranking within time budget** - gold medal if achievable, otherwise best possible medal (silver/bronze)
- **Gold medal is NOT guaranteed** - some competitions are too hard for this setup
```

**Rationale:** Reframes gold as one of several valid outcomes, not the default expectation.

---

### 3. **Lines 40-41 - Efficiency Mindset** ✓
**Before:**
```
- **Efficiency mindset:** Aim for best score within time/compute budget, not perfect score at any cost
```

**After:**
```
- **Efficiency mindset:** Aim for best ranking within time/compute budget, not perfect score at any cost
- **Success = maximizing ranking given constraints** - gold is ideal but silver/bronze in 20 min can be better than gold in 100+ min
```

**Rationale:** Explicitly values time efficiency, legitimizes non-gold outcomes.

---

### 4. **Line 148 - Oracle Consultation Prompt** ✓
**Before:**
```
Validate my strategy and recommend optimizations for gold-medal performance in 20±10 min."
```

**After:**
```
Validate my strategy and recommend optimizations for best possible ranking in 20±10 min (gold if feasible, otherwise maximize medal tier)."
```

**Rationale:** Oracle guidance should optimize for best achievable, not assume gold is possible.

---

### 5. **Line 165 - Strategic Planning Goal** ✓
**Before:**
```
• Goal: Strong performance in reasonable time (balance quality vs speed)
```

**After:**
```
• Goal: Best achievable ranking within time budget (balance quality vs speed)
```

**Rationale:** Clarifies "strong performance" means best ranking given constraints.

---

### 6. **Line 194 - Execution Guidance** ✓
**Before:**
```
• Oracle has already provided a gold-medal strategy - execute that plan, not generic baselines
```

**After:**
```
• Oracle has already provided a strategy optimized for best ranking within time - execute that plan, not generic baselines
```

**Rationale:** Removes "gold-medal" assumption from execution phase.

---

## Philosophy

- **Time-first mindset:** 20±10 min is the primary constraint (allow 40 min for edge cases)
- **Realistic outcomes:** Gold if feasible, otherwise silver/bronze is success
- **Efficiency valued:** Faster completion with good ranking > slow perfection
- **Minimal changes:** Only 6 edits, preserving existing realistic guidance (lines 24-39)

## Unchanged (Intentionally Preserved)

- **Time constraints:** Still target 20±10 min, allow up to 40 min for extreme cases
- **"When to push for gold" section:** Still encourages gold when gap is small/strategy clear
- **Resource optimization:** Still maximize GPU/CPU usage
- **Quality standards:** Still use Oracle, validation checks, best practices

## Expected Behavior Changes

**Before:** Agent felt pressure to always chase gold, potentially wasting time on impossible improvements.

**After:** Agent will:
1. Assess gold feasibility early (with Oracle)
2. Optimize for best medal tier achievable in time
3. Stop improving when returns diminish (silver in 20 min > gold in 100+ min)
4. Feel legitimized settling for silver/bronze if competition is too hard

## Validation

Run this command to verify all changes:
```bash
grep -n "explicit objective\|REALISTIC GOAL\|Efficiency mindset\|Validate my strategy\|Goal: Best\|Oracle has already" /Users/Yifan/canada-research/mle-bench/agents/agent_v5_kaggle/kaggle_agent.py
```

Expected output showing all 6 changes:
```
19:Your explicit objective is to **maximize your ranking within time constraints (typically 20±10 min)**
21:**REALISTIC GOAL SETTING (CRITICAL):**
40:- **Efficiency mindset:** Aim for best ranking within time/compute budget
148:Validate my strategy and recommend optimizations for best possible ranking in 20±10 min (gold if feasible, otherwise maximize medal tier)."
165:• Goal: Best achievable ranking within time budget (balance quality vs speed)
194:• Oracle has already provided a strategy optimized for best ranking within time
```

✅ All changes verified successfully!

---

# Efficient Training Strategies Added (2025-10-17)

**Commit:** TBD
**Status:** ✅ Complete

## Objective

Added domain-specific model sizing guidance optimized for 20-30 minute time budget to:
1. Kaggle Competition Strategy playbook
2. Kaggle Agent prompt
3. Oracle system prompt

## Changes Made

### 1. **kaggle_competition_strategy.txt** - Added Part V ✓

**Location:** `/Users/Yifan/canada-research/mle-bench/environment/kaggle_competition_strategy.txt`

**Added Section:** "PART V: EFFICIENT TRAINING STRATEGIES (20-30 MINUTE BUDGET)"

**Contents (~90 lines):**
- **Image Classification:** EfficientNet-B0/ResNet-34, 3-5 epochs, 224x224
- **Image Segmentation:** U-Net + EfficientNet-B0/ResNet-34, 256x256 tiles, 5-10 epochs
- **Object Detection:** YOLOv5s/v8n, PointPillars (3D), 5-10 epochs fine-tuning
- **Tabular & Time Series:** LightGBM, minimal features, 3-fold CV
- **NLP:** distilbert/small DeBERTa, 1-2 epochs, max_length=128/256
- **Audio:** Mel-spectrogram → EfficientNet-B0/ResNet

**Updated:** Quick Decision Tree to reference Part V and include specific model choices

**Rationale:** Provides concrete model sizing guidance for each domain based on time constraints

---

### 2. **kaggle_agent.py** - Condensed Model Sizing ✓

**Location:** `/Users/Yifan/canada-research/mle-bench/agents/agent_v5_kaggle/kaggle_agent.py`

**Modified:** Lines 75-82 (Domain-specific architectures section)

**Before:**
```
- Tabular: GBDTs (LightGBM/XGBoost/CatBoost), heavy feature engineering, GBDT+NN ensembles
- Computer Vision: EfficientNet/ResNeXt/ViT, advanced augmentation (MixUp/CutMix), TTA
- NLP: Transformer models (BERT/RoBERTa/DeBERTa), fine-tuning strategies, knowledge distillation
- Time Series: Transform to tabular + GBDTs, lag/window features, TimeSeriesSplit CV
```

**After:**
```
- Tabular: LightGBM (fastest), XGBoost, CatBoost. Minimal feature engineering for speed.
- Image Classification: EfficientNet-B0/B2 (20-30 min), B3/B4 (40-60 min), ResNet-34 baseline. MixUp/CutMix.
- Image Segmentation: U-Net + EfficientNet-B0/ResNet-34 backbone, 256x256 tiles, 5-10 epochs
- Object Detection: YOLOv5s/v8n (fast), PointPillars (3D). Fine-tune 5-10 epochs.
- NLP: distilbert (fastest), DeBERTa (stronger). Train 1-2 epochs only. max_length=128/256.
- Time Series: Transform to tabular + LightGBM. Lag/rolling features. TimeSeriesSplit CV.
- Audio: Mel-spectrogram → EfficientNet-B0/ResNet (treat as image classification)
```

**Rationale:** Agent sees specific model choices for each domain with time budget in parentheses

---

### 3. **oracle.py** - Model Sizing Guide + Goal Alignment ✓

**Location:** `/Users/Yifan/canada-research/agent_v5/tools/oracle.py`

**Changes:**

#### A. Added MODEL SIZING GUIDE section (lines 192-200) ✓
```
**MODEL SIZING GUIDE (20-30 MIN BUDGET):**
• **Tabular:** LightGBM (fastest), 3-fold CV, default params + early stopping
• **Image Classification:** EfficientNet-B0/B2 or ResNet-34. 3-fold CV, 3-5 epochs, 224x224 images.
• **Image Segmentation:** U-Net + EfficientNet-B0/ResNet-34 backbone. 256x256 tiles, 3-fold CV, 5-10 epochs.
• **Object Detection:** YOLOv5s/v8n (images), PointPillars (3D). Fine-tune 5-10 epochs, 512x512 images.
• **NLP:** distilbert-base-uncased (fastest) or small DeBERTa. 1-2 epochs only, max_length=128/256.
• **Time Series:** Transform to tabular + LightGBM. Lag/rolling features, TimeSeriesSplit CV.
• **Audio:** Mel-spectrogram → EfficientNet-B0/ResNet. Treat as image classification.
• **AVOID THESE FOR SPEED:** EfficientNet-B4+ (too slow for 30-min), 5-fold CV (use 3), >8 epochs, >300x300 images
```

#### B. Updated REALISTIC GOAL SETTING (lines 220-226) ✓
**Before:**
```
• **Gold medal is the GOAL, but NOT always achievable** - some competitions are too hard for this setup
• **Time/EV Tradeoff:** Consider expected value of additional training time
  - Silver medal in 20 min > gold medal in 120 min (if improvement uncertain)
  - Quick iteration > perfect solution (we can try multiple approaches)
```

**After:**
```
• **Maximize ranking within time budget** - gold medal if achievable, otherwise best possible medal (silver/bronze)
• **Gold medal is NOT guaranteed** - some competitions are too hard for this setup
• **Time/EV Tradeoff:** Consider expected value of additional training time
  - Silver medal in 20 min > gold medal in 120 min (if improvement uncertain)
  - Quick iteration > perfect solution (we can try multiple approaches)
• **Success = maximizing ranking given constraints** - gold is ideal but silver/bronze in 20 min can be better than gold in 100+ min
```

**Rationale:** Oracle now has concrete model sizing recommendations and aligned goal philosophy

---

## Expected Impact

### For Agent:
1. **Clearer model choices:** Knows exactly which models to use for each domain
2. **Time-aware defaults:** Sees time budgets next to model names (B0/B2 for 20-30 min)
3. **Faster decisions:** Less time exploring model options, more execution

### For Oracle:
1. **Consistent recommendations:** Has model sizing reference for 20-30 min budget
2. **Avoids slow models:** Knows to avoid B4+ for 30-min budget
3. **Aligned goals:** Won't push for gold if time/complexity makes it infeasible

### For Strategy:
1. **Complete reference:** Part V provides detailed strategies for 6 competition types
2. **Concrete examples:** Lists actual competition names for each type
3. **Actionable templates:** Exact configurations (epochs, folds, image sizes)

---

## Philosophy

- **Domain-specific:** Each competition type has tailored model recommendations
- **Time-first:** Model choices explicitly tied to time budgets (20-30 min vs 40-60 min)
- **Battle-tested:** Based on winning Kaggle solutions and practical experience
- **Actionable:** Concrete configurations, not just high-level advice

---

## Validation

### Test 1: Check kaggle_competition_strategy.txt
```bash
grep -A 5 "PART V: EFFICIENT TRAINING STRATEGIES" /Users/Yifan/canada-research/mle-bench/environment/kaggle_competition_strategy.txt
```
Should show Part V section with 6 domain types.

### Test 2: Check kaggle_agent.py
```bash
grep -A 7 "Domain-specific architectures" /Users/Yifan/canada-research/mle-bench/agents/agent_v5_kaggle/kaggle_agent.py
```
Should show condensed model sizing with time budgets.

### Test 3: Check oracle.py
```bash
grep -A 10 "MODEL SIZING GUIDE" /Users/Yifan/canada-research/agent_v5/tools/oracle.py
```
Should show Oracle's model sizing reference.

---

## Summary

Added efficient training strategies to 3 key locations:
1. **Strategy playbook:** Detailed Part V with 6 competition types
2. **Agent prompt:** Condensed model sizing with time budgets
3. **Oracle prompt:** Model sizing guide + aligned goal philosophy

**Total additions:** ~120 lines of actionable domain-specific guidance

**Key principle:** Maximize ranking within time budget, not perfect score at any cost

✅ All changes complete and validated!

---

# Parallel Training Strategy Added (2025-10-17)

**Commit:** TBD
**Status:** ✅ Complete

## Objective

Added parallel training strategy to maximize hardware utilization (36 CPUs + A10 24GB GPU) by training multiple smaller models simultaneously instead of one large sequential model.

**Key Insight:** 3 small models in parallel (10-12 min) + ensemble > 1 large model (25-30 min)

## Changes Made

### 1. **kaggle_agent.py** - Detailed Parallel Training Section ✓

**Location:** Lines 221-269 (48 lines added to TIME MANAGEMENT section)

**Contents:**
- **Concept:** Train 2-3 smaller/diverse models simultaneously → ensemble results
- **Why faster:** Concrete comparison (3 small parallel 10 min vs 1 large 30 min)
- **Hardware utilization:** 36 CPUs + A10 GPU → run 2-3 models concurrently
- **When to use:** Single model >25 min, benefits from diversity, early exploration
- **Implementation pattern:** Code template showing 3 parallel jobs
- **Resource allocation:**
  * Model 1 (CPU-only): LightGBM, 12 cores, 0% GPU
  * Model 2 (GPU): ResNet-34/B0, 12 cores, batch_size=64, ~8-10GB GPU
  * Model 3 (GPU): ResNet-34/B0, 12 cores, batch_size=64, ~8-10GB GPU
- **Practical examples:**
  * Image Classification: LightGBM features + ResNet-34 + EfficientNet-B0
  * Tabular: LightGBM + XGBoost + CatBoost (all CPU)
  * Mixed: LightGBM (18 cores CPU) + Tabular NN (18 cores + full GPU)
- **Ensemble strategy:** Weighted average by CV score
- **Monitoring:** GPU memory split, CPU usage, OOM handling
- **When NOT to use:** Single model <20 min, needs full GPU, I/O bottleneck

**Rationale:** Agent has complete blueprint for implementing parallel training with concrete examples

---

### 2. **kaggle_competition_strategy.txt** - Part VI Added ✓

**Location:** Lines 270-346 (76 lines, new Part VI section)

**Contents:**
- **Concept & speed benefit** (clear comparison)
- **When to use parallel training** (4 criteria)
- **Resource allocation pattern** (specific CPU/GPU splits)
- **Practical implementation:**
  * Example 1: Image Classification (3 models)
  * Example 2: Tabular (3 GBDTs)
  * Example 3: Mixed (GBDT + Neural)
- **Ensemble strategy** (weighted average formula)
- **Monitoring parallel jobs** (GPU/CPU checks)
- **When NOT to use** (4 cases)
- **Key principle:** Diversity + Speed

**Rationale:** Complete reference guide with 3 concrete examples for different competition types

---

### 3. **oracle.py** - Condensed Parallel Training Guide ✓

**Location:** Lines 201-211 (11 lines added before HARDWARE section)

**Contents:**
- **Concept:** Multiple small models simultaneously → ensemble
- **When to use:** Single model >25 min, diversity helps, hardware supports
- **Resource split:** Specific allocation (12 cores each, GPU sharing)
- **Speed benefit:** Concrete comparison (3 parallel 10-12 min vs 1 large 25-30 min)
- **Diversity bonus:** +1-3% from different models
- **Example:** LightGBM + ResNet-34 + EfficientNet-B0 parallel
- **When NOT to use:** 3 clear cases

**Rationale:** Oracle can recommend parallel training when single model too slow

---

## Key Benefits

### Speed Optimization:
- **Faster completion:** 10-12 min parallel vs 25-30 min sequential
- **Better time utilization:** Uses all 36 CPUs + GPU simultaneously
- **Fits 20±10 min budget:** Parallel approach more likely to complete in target time

### Performance Optimization:
- **Diversity bonus:** Different architectures ensemble → +1-3% boost
- **Risk mitigation:** If one model fails/underperforms, still have 2 others
- **Better exploration:** Try multiple approaches simultaneously

### Resource Optimization:
- **CPU utilization:** 36 cores split across models (not wasted)
- **GPU sharing:** PyTorch naturally shares GPU between processes
- **Memory efficiency:** Smaller models (batch_size=64) use less VRAM

---

## Resource Allocation Patterns

### Pattern 1: Image Classification (3 models)
```
CPU (36 cores):  [12] LightGBM  |  [12] ResNet-34  |  [12] EfficientNet-B0
GPU (24GB):      [ 0%  ]         |  [40%  8-10GB ]  |  [40%  8-10GB      ]
Time: ~10-12 min all models → ensemble → submit
```

### Pattern 2: Tabular (3 GBDTs)
```
CPU (36 cores):  [12] LightGBM  |  [12] XGBoost  |  [12] CatBoost
GPU (24GB):      [ 0%  unused   ]
Time: ~8-10 min all models → ensemble → submit
```

### Pattern 3: Mixed (2 models)
```
CPU (36 cores):  [18] LightGBM              |  [18] Tabular NN
GPU (24GB):      [ 0%  unused   ]           |  [100%  24GB full]
Time: ~10 min both models → ensemble → submit
```

---

## Practical Implementation Example

**Scenario:** Image classification, single EfficientNet-B3 estimated 28 min (too slow)

**Parallel Alternative:**
```python
# Step 1: Write 3 training scripts (lighter models)
# train_lgbm_features.py - Extract features, train LightGBM
# train_resnet34.py - ResNet-34, batch_size=64, num_workers=12
# train_effnet_b0.py - EfficientNet-B0, batch_size=64, num_workers=12

# Step 2: Launch all 3 in parallel (background=true)
Bash(command="python -u train_lgbm_features.py --n_jobs=12", background=true)
Bash(command="python -u train_resnet34.py --batch_size=64 --num_workers=12", background=true)
Bash(command="python -u train_effnet_b0.py --batch_size=64 --num_workers=12", background=true)

# Step 3: Monitor all 3 jobs every 2-3 min with ReadBashOutput
# Check GPU: nvidia-smi should show 2 processes, ~60-80% total memory
# Check progress: Each model should complete in ~10-12 min

# Step 4: After all complete, ensemble predictions
# weights = [cv_lgbm/sum, cv_resnet/sum, cv_effnet/sum]
# final = w1*pred_lgbm + w2*pred_resnet + w3*pred_effnet

# Result: 10-12 min total + diversity bonus (+1-3%) vs 28 min single model
```

---

## Expected Agent Behavior Changes

### Before (Sequential):
```
Agent: Single EfficientNet-B3 estimated 28 min
Agent: Launching training...
[28 min later]
Agent: Training complete, generating submission
Total: 28 min + 5 min inference = 33 min (exceeds 30 min budget)
```

### After (Parallel):
```
Agent: Single EfficientNet-B3 estimated 28 min (too slow)
Agent: Using parallel training strategy instead
Agent: Launching 3 models in parallel (LightGBM + ResNet-34 + EfficientNet-B0)
[10-12 min later]
Agent: All 3 models complete
Agent: Ensembling predictions (weighted average)
Agent: Generating submission
Total: 12 min + 3 min ensemble + 5 min inference = 20 min (fits budget!)
```

---

## Oracle Recommendations

Oracle can now suggest parallel training:

**Example consultation:**
```
Agent: "Oracle, single model estimated 28 min. What should I do?"

Oracle: "28 min exceeds our 20±10 min budget. Use parallel training strategy instead:

1. Launch 3 smaller models simultaneously (each 10-12 min):
   - LightGBM on extracted features (12 cores CPU)
   - ResNet-34 (12 cores + 8GB GPU, batch_size=64)
   - EfficientNet-B0 (12 cores + 8GB GPU, batch_size=64)

2. After all complete (~12 min), ensemble with weighted average

Benefits:
- Completes in 20 min (fits budget) vs 28+ min single model
- Diversity bonus: +1-3% from different architectures
- Risk mitigation: 3 models more robust than 1

Resource allocation handles GPU sharing automatically. Monitor with nvidia-smi."
```

---

## Validation

### Test 1: Check kaggle_agent.py has parallel section
```bash
grep -A 5 "PARALLEL TRAINING STRATEGY" /Users/Yifan/canada-research/mle-bench/agents/agent_v5_kaggle/kaggle_agent.py
```
Should show 48-line detailed section with examples.

### Test 2: Check strategy playbook has Part VI
```bash
grep -A 10 "PART VI: PARALLEL TRAINING" /Users/Yifan/canada-research/mle-bench/environment/kaggle_competition_strategy.txt
```
Should show 76-line section with 3 practical examples.

### Test 3: Check Oracle has parallel guidance
```bash
grep -A 8 "PARALLEL TRAINING" /Users/Yifan/canada-research/agent_v5/tools/oracle.py
```
Should show 11-line condensed guide.

---

## Summary

**Added:** Parallel training strategy to 3 key files
- **Agent:** 48 lines (detailed implementation guide)
- **Strategy:** 76 lines (Part VI with 3 examples)
- **Oracle:** 11 lines (condensed recommendations)

**Total additions:** ~135 lines of parallel training guidance

**Key principle:** Train multiple small models in parallel → ensemble → faster & better than single large sequential model

**Speed benefit:** 10-12 min parallel vs 25-30 min sequential (fits 20±10 min budget)

**Performance benefit:** Diversity bonus +1-3% from ensembling different architectures

✅ All changes complete and validated!

---

# Memory System Added (2025-10-17)

**Commit:** TBD
**Status:** ✅ Complete

## Objective

Added memory management system to Kaggle agent to learn from past competitions and make better decisions faster, similar to agent_v6's competition memory.

## Key Components

### 1. **CompetitionMemory Class** ✓

**Location:** `/Users/Yifan/canada-research/mle-bench/agents/agent_v5_kaggle/memory/competition_memory.py`

**Features:**
- **Pattern Storage:** Learned patterns for 7 competition types (image classification, segmentation, detection, tabular, NLP, time series, audio)
- **Size Categories:** Small/medium/large dataset strategies for image tasks
- **Model Recommendations:** Best models for each domain optimized for 20-30 min budget
- **Time Estimates:** Expected completion times per domain/size
- **Medal Expectations:** Realistic medal targets (bronze/silver/gold)
- **Parallel Training Patterns:** Resource allocation for parallel model training
- **Competition History:** JSONL log of all past competitions with results

**Key Methods:**
```python
# Get strategy for new competition
strategy = memory.get_strategy_for_competition(
    data_type="image",  # or tabular, nlp, etc.
    dataset_size=50000,
    time_budget_min=30
)
# Returns: {recommended_models, strategies, estimated_time, expected_medal, use_parallel_training}

# Find similar past competitions
similar = memory.get_similar_competitions(
    data_type="image",
    dataset_size=50000,
    limit=5
)
# Returns: Top 5 similar competitions sorted by medal (best first) then time (fastest)

# Record competition result (for learning)
memory.record_competition_result(
    competition_id="cassava-leaf-disease",
    data_type="image_classification",
    dataset_size=21397,
    strategy="parallel_training",
    models_used=["LightGBM", "ResNet-34", "EfficientNet-B0"],
    final_score=0.897,
    time_minutes=18.5,
    medal="silver",
    notes="Parallel training saved 10 min, ensemble gave +2% boost"
)
# Updates patterns if result shows better approach than stored

# Get memory summary
summary = memory.get_memory_summary()
# Returns formatted text of all learned patterns
```

---

### 2. **Default Patterns** ✓

Memory comes pre-seeded with battle-tested patterns from Kaggle competition strategy playbook:

#### Image Classification:
- **Small (<10K):** EfficientNet-B0/ResNet-34, 3-fold CV, 3-5 epochs, 8-12 min, bronze-silver
- **Medium (10-100K):** EfficientNet-B2/ResNet-50, 3-fold CV, 6-8 epochs, MixUp, 15-25 min, silver-gold
- **Large (>100K):** EfficientNet-B3/B4, 2-fold CV, 6-8 epochs, 20-30 min, silver-gold

#### Tabular:
- LightGBM/XGBoost/CatBoost, 3-fold CV, minimal features, early stopping, 5-15 min, silver-gold

#### NLP:
- distilbert/DeBERTa-small, 1-2 epochs, max_length=128/256, 3-fold CV, 10-20 min, bronze-silver

#### Image Segmentation:
- U-Net + EfficientNet-B0/ResNet-34, 256x256 tiles, 3-fold CV, 5-10 epochs, 15-25 min, bronze-silver

#### Object Detection:
- YOLOv5s/v8n, 512x512, 5-10 epochs fine-tune, 10-20 min, bronze-silver

#### Time Series:
- LightGBM with lag features, TimeSeriesSplit CV, 8-15 min, bronze-silver

#### Audio:
- EfficientNet-B0 on mel-spectrograms, 10-20 min, bronze-silver

---

### 3. **Parallel Training Patterns** ✓

Memory stores resource allocation patterns for parallel training:

**Image Classification:**
- Models: LightGBM (features) + ResNet-34 + EfficientNet-B0
- Resources: 12 cores CPU + (12 cores + 8GB GPU) + (12 cores + 8GB GPU)
- Time: 10-12 min
- Diversity bonus: +1-3%

**Tabular:**
- Models: LightGBM + XGBoost + CatBoost
- Resources: 12 cores + 12 cores + 12 cores (all CPU)
- Time: 8-10 min
- Diversity bonus: +0.5-2%

---

### 4. **Agent Prompt Integration** ✓

**Location:** `/Users/Yifan/canada-research/mle-bench/agents/agent_v5_kaggle/kaggle_agent.py`

**Changes:**

#### A. Added MEMORY SYSTEM section (lines 113-122):
```
**MEMORY SYSTEM (LEARN FROM PAST COMPETITIONS):**
- Location: Python module available via `from memory import CompetitionMemory`
- Purpose: Learn from past competitions to make better decisions faster
- When to use:
  * IMMEDIATELY after data exploration (step 1) - before Oracle consultation
  * Get recommended strategy: memory.get_strategy_for_competition(...)
  * Get similar competitions: memory.get_similar_competitions(...)
- After competition: Record results with memory.record_competition_result(...)
- Contains: Battle-tested model choices, time estimates, expected medals, parallel training patterns
- Format memory insights in Oracle query: "Memory recommends: [models], [strategies], [time estimate]. Does this align?"
```

#### B. Updated step 1 - Data Exploration (line 136):
```
• **CONSULT MEMORY FIRST:** After data exploration, query memory system for learned patterns
```

#### C. Updated step 3 - Oracle Consultation (lines 150-175):
Now requires agent to include memory insights in Oracle query:
```
**Memory System Recommendations:**
- Recommended models: [list from memory.get_strategy_for_competition()]
- Recommended strategies: [from memory]
- Estimated time: [from memory]
- Expected medal: [from memory]
- Similar past competitions: [from memory.get_similar_competitions()]
- Use parallel training: [Yes/No from memory]
```

#### D. Added Deliverables section (lines 519-534):
After submission, agent must record results:
```python
from memory import CompetitionMemory
memory = CompetitionMemory()
memory.record_competition_result(
    competition_id="{competition_id}",
    data_type="[image/tabular/nlp/etc]",
    dataset_size=[number],
    strategy="[parallel/sequential/fine_tuning/etc]",
    models_used=["model1", "model2"],
    final_score=[your_final_score],
    time_minutes=[total_time],
    medal="[gold/silver/bronze/none]",
    notes="[what worked / what didn't / key insights]"
)
```

---

## Workflow Integration

### New Competition Workflow:

**Before (no memory):**
```
1. Data exploration
2. Read playbook
3. Consult Oracle
4. Execute strategy
5. Submit
```

**After (with memory):**
```
1. Data exploration
2. Query memory system → get recommendations
3. Read playbook
4. Consult Oracle with memory insights
5. Execute strategy (informed by memory + Oracle)
6. Submit
7. Record results in memory (for future competitions)
```

---

## Expected Behavior Changes

### Scenario 1: Image Classification (Small Dataset)

**Agent discovers:** 8,000 sample dataset, image classification

**Agent queries memory:**
```python
strategy = memory.get_strategy_for_competition("image", 8000, 30)
```

**Memory returns:**
```json
{
  "recommended_models": ["EfficientNet-B0", "ResNet-34"],
  "recommended_strategies": ["3-fold CV", "3-5 epochs", "224x224"],
  "avoid": ["EfficientNet-B4+", "5-fold CV", ">8 epochs"],
  "estimated_time_min": "8-12",
  "expected_medal": "bronze-silver",
  "use_parallel_training": false
}
```

**Agent to Oracle:**
```
"Memory recommends EfficientNet-B0/ResNet-34, 3-fold CV, 3-5 epochs, estimated 8-12 min, 
expect bronze-silver medal. Playbook agrees. Should I use single model or parallel training?"
```

**Benefit:** Agent starts with proven strategy, saves exploration time

---

### Scenario 2: Tabular (Medium Dataset)

**Agent discovers:** 50,000 samples, tabular data

**Agent queries memory:**
```python
strategy = memory.get_strategy_for_competition("tabular", 50000, 30)
similar = memory.get_similar_competitions("tabular", 50000, 5)
```

**Memory returns:**
```json
{
  "recommended_models": ["LightGBM", "XGBoost", "CatBoost"],
  "recommended_strategies": ["3-fold CV", "minimal features", "early stopping"],
  "estimated_time_min": "5-15",
  "expected_medal": "silver-gold",
  "use_parallel_training": false,
  "similar_competitions": [
    {"id": "tabular-playground-may", "medal": "silver", "time": 12, "models": ["LightGBM", "XGBoost"]},
    {"id": "ventilator-pressure", "medal": "bronze", "time": 18, "models": ["CatBoost"]}
  ]
}
```

**Agent to Oracle:**
```
"Memory recommends LightGBM/XGBoost/CatBoost, 3-fold CV, 5-15 min estimated.
Similar past competition (tabular-playground-may) achieved silver with LightGBM+XGBoost in 12 min.
Should I replicate that approach or try something different?"
```

**Benefit:** Agent learns from similar past competitions, avoids repeating mistakes

---

### Scenario 3: Learning Over Time

**Competition 1 (no history):**
- Agent uses default patterns
- Achieves bronze with EfficientNet-B2, 20 min
- Records: `medal="bronze", time=20, models=["EfficientNet-B2"]`

**Competition 2 (learns from #1):**
- Memory now knows EfficientNet-B2 worked
- Agent tries parallel training (LightGBM + ResNet-34 + EfficientNet-B0)
- Achieves silver in 15 min
- Records: `medal="silver", time=15, models=["LightGBM", "ResNet-34", "EfficientNet-B0"], notes="Parallel training saved 5 min, +2% boost"`

**Competition 3 (learns from #1 and #2):**
- Memory recommends parallel training (proven in #2)
- Agent starts with parallel immediately
- Achieves silver in 14 min
- **Learning loop complete!**

---

## Storage & Persistence

### Memory Directory Structure:
```
/home/.kaggle_memory/  (or $KAGGLE_MEMORY_DIR)
├── patterns.json                 # Learned patterns (JSON)
├── strategies.pkl                # Parallel training patterns (pickle)
└── competition_history.jsonl     # All competition results (JSONL)
```

### Example competition_history.jsonl:
```jsonl
{"competition_id": "cassava-leaf", "timestamp": "2025-10-17T10:30:00", "data_type": "image_classification", "dataset_size": 21397, "strategy": "parallel_training", "models_used": ["LightGBM", "ResNet-34", "EfficientNet-B0"], "final_score": 0.897, "time_minutes": 18.5, "medal": "silver", "notes": "Parallel saved 10 min, ensemble +2%"}
{"competition_id": "tabular-playground-june", "timestamp": "2025-10-17T11:15:00", "data_type": "tabular", "dataset_size": 50000, "strategy": "gradient_boosting", "models_used": ["LightGBM", "XGBoost"], "final_score": 0.923, "time_minutes": 12.3, "medal": "gold", "notes": "Minimal features worked best"}
```

---

## Key Benefits

1. **Faster decisions:** Agent doesn't explore blindly, starts with proven strategies
2. **Time-aware:** Memory includes time estimates to prevent exceeding budget
3. **Medal-aware:** Realistic expectations (don't chase impossible gold)
4. **Learns from experience:** Updates patterns when better approaches found
5. **Similar competition insights:** Learn from competitions with similar data characteristics
6. **Parallel training guidance:** Knows when parallel training is beneficial
7. **Persistent knowledge:** Survives across agent restarts, accumulates over time

---

## Validation

### Test 1: Check memory module exists
```bash
ls -la /Users/Yifan/canada-research/mle-bench/agents/agent_v5_kaggle/memory/
```
Should show: `__init__.py`, `competition_memory.py`

### Test 2: Check agent prompt has memory integration
```bash
grep -n "MEMORY SYSTEM" /Users/Yifan/canada-research/mle-bench/agents/agent_v5_kaggle/kaggle_agent.py
```
Should show line 113 (MEMORY SYSTEM section)

### Test 3: Test memory module
```python
from memory import CompetitionMemory
memory = CompetitionMemory()

# Get strategy
strategy = memory.get_strategy_for_competition("image", 50000, 30)
print(strategy)

# Should show: recommended_models, strategies, time estimate, etc.
```

---

## Summary

**Added:** Memory management system to Kaggle agent
- **Memory module:** 300+ lines (patterns, strategies, recording, retrieval)
- **Agent integration:** 4 key changes (memory section, data exploration, Oracle consultation, deliverables)
- **Default patterns:** 7 competition types × multiple size categories
- **Parallel training patterns:** 2 domains (image, tabular)

**Total additions:** ~350 lines of memory system + prompt integration

**Key principle:** Learn from past competitions → make better decisions faster → improve over time

**Learning loop:** Competition → Record result → Update patterns → Next competition uses learned knowledge

✅ All changes complete and documented!
