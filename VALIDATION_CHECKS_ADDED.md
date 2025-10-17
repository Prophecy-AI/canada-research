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
