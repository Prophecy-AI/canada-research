# Agent V6 Comprehensive Improvement Plan

## Executive Summary

**Current Performance: 3/22 medals (13.6% success rate)**
- 🥈 1 Silver (nomad2018 - gradient boosting tabular)
- 🥉 1 Bronze (whale - bottleneck audio spectrograms)
- 📈 1 Above Median (dog-breed - bottleneck images)

**Average Time: 17.1 minutes per experiment**
**Time Violations (>30m): 3 experiments (86m, 62m, 40m)**

---

## Critical Bugs Identified

### 1. STRATEGY MISMATCH (Highest Impact)

**Problem:** Planning agent ignores data type from EDA

**Example:** leaf-classification
- EDA: "Tabular data with 192 pre-extracted numerical features"
- Planning: Still chose `bottleneck_features` (image strategy)
- Result: Score 4.77345 vs Gold 0.0 (complete disaster)

**Root Cause:** Planning prompt doesn't enforce strict data-type → strategy mapping

**Fix Priority:** 🔴 CRITICAL
- Add explicit data type extraction and conditional logic
- Force tabular data → gradient_boosting ONLY
- Force image data → bottleneck_features or fine_tuning
- Force text data → transformer_features or gradient_boosting with TF-IDF

### 2. CATBOOST PARAMETER BUG (Affects 4 experiments)

**Problem:** Planning still outputs `bootstrap_type="Bayesian"` + `subsample=0.8`

**Evidence:**
- nomad2018-predict-transparent-conductors
- tabular-playground-series-dec-2021
- tabular-playground-series-may-2022
- new-york-city-taxi-fare-prediction

**Current Fix:** Warning in planning prompt (lines 74-76)
**Why it fails:** LLM ignores warnings, generates invalid configs anyway

**Fix Priority:** 🟠 HIGH
- Remove subsample from ALL CatBoost examples
- Add JSON schema validation BEFORE worker execution
- OR: Just remove CatBoost entirely (LightGBM + XGBoost sufficient)

### 3. TIME VIOLATIONS (3 experiments waste >30 min)

**Problem:** No per-experiment timeout, only cumulative 30-minute budget

**Violators:**
- ranzcr-clip: 86.7m (medical images, 27K samples) - 3x over budget!
- siim-isic-melanoma: 62.8m (medical images, 29K samples) - 2x over budget!
- text-normalization: 40.5m (9M tokens) - 1.3x over budget

**Root Cause:** 
- Large datasets (>25K images, >5M tokens) take too long
- No experiment-level timeout enforcement
- Fine-tuning on large datasets wastes time

**Fix Priority:** 🔴 CRITICAL
- Add **per-experiment timeout: 15 minutes MAX**
- Kill experiment if it exceeds timeout
- For datasets >20K samples: Use simpler strategies or subsample

### 4. JSON PARSING ERRORS (Still occurring)

**Problem:** Control characters in JSON despite regex cleanup

**Evidence:** tabular-playground-series-may-2022

**Current Fix:** Strip `\n`, `\r`, `\t` from JSON (lines 157-159 in orchestrator.py)
**Why it fails:** Still getting parse errors

**Fix Priority:** 🟡 MEDIUM
- Increase max_tokens further (60k → 80k)
- Add retry logic for planning if JSON fails
- Simplify planning output format (less verbose hypotheses)

### 5. INCOMPLETE EXPERIMENTS (9 "UNKNOWN" results)

**Problem:** Experiments start but never finish/report scores

**Likely Causes:**
- Worker code crashes silently
- No VALIDATION_SCORE printed
- Import errors or missing dependencies

**Fix Priority:** 🟡 MEDIUM  
- Better error reporting from worker execution
- Validate imports before running train.py
- Add timeout to training execution

---

## What's Working (Don't Break These!)

### ✅ Bottleneck Features for Small Image Datasets
- **dog-breed** (9K images): Above Median in 5.4m
- **whale-audio** (22K audio): Bronze in 11.6m
- Multi-model ensembles (ResNet50 + InceptionV3, etc.)
- LogisticRegression classifier

### ✅ Gradient Boosting for Tabular
- **nomad2018** (2K materials): Silver in 5.4m
- LightGBM, XGBoost work well
- CatBoost has bugs but concept is sound

### ✅ Time Efficiency on Simple Tasks
- Most experiments complete in 5-7 minutes
- Bottleneck approach is FAST (2-4 min training)

---

## Recommended Fixes (Prioritized)

### FIX #1: Enforce Data Type → Strategy Mapping (CRITICAL)

**Current Problem:** Planning sees "image files" and ignores that features are pre-extracted

**Solution:** Add strict conditional logic to planning prompt

```markdown
**CRITICAL - Data Type → Strategy Mapping:**

1. **IF EDA says "Tabular" or "pre-extracted features" or "CSV with X features":**
   → MUST use "gradient_boosting" strategy ONLY
   → DO NOT use bottleneck_features or fine_tuning
   → Example: leaf-classification has 192 tabular features → use gradient_boosting

2. **IF EDA says "Image" AND mentions image files (.jpg, .png) AND NO pre-extracted features:**
   → Use "bottleneck_features" (for <50K images) or "fine_tuning" (for >50K)
   → Load actual images, not CSV features

3. **IF EDA says "Text":**
   → Use "transformer_features" OR "gradient_boosting" with TF-IDF
   
4. **IF EDA says "Audio":**
   → Convert to spectrograms, use "bottleneck_features"

**Double-check:** Read EDA findings carefully - if it mentions "tabular" or "features in CSV", it's NOT an image task!
```

**Implementation:** Update `PLANNING_PROMPT` lines 25-45

### FIX #2: Remove CatBoost Entirely (HIGH)

**Rationale:**
- Affects 4/22 experiments with parameter bugs
- LightGBM + XGBoost cover the same use cases
- Not worth the debugging effort

**Solution:** 
```markdown
**Strategy 3: "gradient_boosting"** (for tabular)
- **When to try:** Tabular data
- **Models:** XGBoost (tree_method='hist'), LightGBM
- **DO NOT use CatBoost** (parameter conflicts, not worth debugging time)
```

**Implementation:** Update `PLANNING_PROMPT` line 71-76

### FIX #3: Per-Experiment Timeout (CRITICAL)

**Current:** Only cumulative 30-minute budget
**Problem:** Single experiment can waste 86 minutes!

**Solution:** Add experiment-level timeout in orchestrator

```python
# In orchestrator.py _run_training():
EXPERIMENT_TIMEOUT_SECONDS = 900  # 15 minutes MAX per experiment

try:
    await asyncio.wait_for(
        process.wait(),
        timeout=EXPERIMENT_TIMEOUT_SECONDS
    )
except asyncio.TimeoutError:
    process.kill()
    return {
        "id": exp_id,
        "status": "timeout",
        "score": None,
        "output": f"Experiment exceeded {EXPERIMENT_TIMEOUT_SECONDS}s timeout",
        ...
    }
```

**Implementation:** Update `orchestrator.py` `_run_training()` method

### FIX #4: Simplify CatBoost Configs (if keeping it)

**Alternative to removing:** Provide ONE safe config

```markdown
**CatBoost safe config:**
```json
{
  "iterations": 500,
  "depth": 7,
  "learning_rate": 0.05,
  "loss_function": "RMSE", // or "Logloss" for classification
  "task_type": "CPU",  // Avoid GPU complications
  "verbose": False
}
```
DO NOT add: subsample, bootstrap_type, bagging_temperature, random_strength
```

### FIX #5: Data Size-Based Strategy Selection

**Problem:** Large datasets (>25K images, >5M rows) take too long

**Solution:** Add size-based guardrails

```markdown
**Data Size Considerations:**
- **Images >25K:** Skip bottleneck/fine-tuning, suggest gradient boosting on metadata if available
- **Tabular >1M rows:** Subsample to 500K for training, use full data for final model
- **Text >1M samples:** Use smaller transformers (distilbert only) or TF-IDF + LightGBM
- **Audio >20K clips:** Bottleneck OK, but limit to 2 models max (not 3)
```

### FIX #6: Worker Robustness for Tabular Data

**Problem:** Worker needs to handle tabular gradient boosting better

**Solution:** Add tabular-specific template to WORKER_PROMPT

```python
**STRATEGY: "gradient_boosting"** (for tabular data):
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error, accuracy_score

# Load data
train = pd.read_csv(f'{data_dir}/train.csv')
test = pd.read_csv(f'{data_dir}/test.csv')

# Separate features and target
X = train.drop(['target_column', 'id'], axis=1, errors='ignore')
y = train['target_column']

# Check class distribution for classification
if y.dtype in ['int64', 'object'] and y.nunique() < 50:
    class_counts = y.value_counts()
    min_class_count = class_counts.min()
    
    # Drop rare classes OR use larger train split
    if min_class_count < 2:
        rare_classes = class_counts[class_counts < 2].index
        print(f"Dropping {len(rare_classes)} rare classes: {rare_classes.tolist()}")
        mask = ~y.isin(rare_classes)
        X = X[mask]
        y = y[mask]
    
    # Safe split
    if y.value_counts().min() >= 2:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)
else:
    # Regression
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

# Train model based on spec
if spec['model'] == 'LightGBM':
    import lightgbm as lgb
    model = lgb.LGBMClassifier(**spec['hyperparameters'])  # or LGBMRegressor
    model.fit(X_train, y_train)
    
elif spec['model'] == 'XGBoost':
    import xgboost as xgb
    model = xgb.XGBClassifier(**spec['hyperparameters'])  # or XGBRegressor
    model.fit(X_train, y_train)

# Validate and report
val_preds = model.predict_proba(X_val)  # or predict() for regression
val_metric = log_loss(y_val, val_preds)  # or appropriate metric
print(f"VALIDATION_SCORE: {val_metric:.6f}")

# Save model
import joblib
joblib.dump(model, 'model.pkl')
```
```

---

## Implementation Roadmap

### Phase 1: Quick Wins (30 minutes)

1. **Remove CatBoost** from planning prompt
   - Update line 71-76 in prompts.py
   - Remove all CatBoost examples

2. **Add per-experiment timeout**
   - Update `_run_training()` in orchestrator.py
   - Set 15-minute max per experiment

3. **Enforce data type mapping**
   - Update planning prompt with strict IF/THEN logic
   - Add data type keywords to match against

### Phase 2: Robustness (1 hour)

4. **Add tabular gradient boosting template** to WORKER_PROMPT
   - Complete working template with all edge cases
   - Safe stratification logic
   - Model selection based on spec

5. **Increase max_tokens** to 80,000
   - Reduce planning verbosity (shorter hypotheses)

6. **Add data size guardrails**
   - Subsample large datasets
   - Skip expensive strategies on large data

### Phase 3: Testing (30 minutes)

7. **Test on known failures:**
   - leaf-classification (tabular mismatch)
   - tabular-playground-series-may-2022 (all errors)
   - dogs-vs-cats (close to median, should get medal)

---

## Expected Impact

### Conservative Estimate:
- **Current:** 3/22 medals (13.6%)
- **After fixes:** 10-12/22 medals (45-55%)

### Breakdown by Fix:
- Fix #1 (Data type mapping): +2-3 medals (leaf, others with tabular data)
- Fix #2 (Remove CatBoost): +1-2 medals (tabular experiments that failed)
- Fix #3 (Per-exp timeout): +0 medals but saves ~150 minutes total
- Fix #4 (Tabular template): +2-3 medals (better gradient boosting code)
- Fix #5 (Data size limits): +1 medal (prevent timeouts, enable completion)

### Target Improvements:
1. ✅ **Tabular experiments** (currently failing) → Use gradient boosting correctly
2. ✅ **Small image experiments** (working) → Keep using bottleneck
3. ✅ **Large experiments** (timing out) → Complete faster with timeouts
4. ✅ **Text experiments** (incomplete) → Simpler strategies that finish

---

## Risks & Mitigations

### Risk 1: Breaking what works
**Mitigation:** Don't touch bottleneck_features logic (it's working for images)

### Risk 2: Timeout too aggressive
**Mitigation:** 15 minutes is generous (dog-breed completed in 5.4m)

### Risk 3: Removing CatBoost loses capability
**Mitigation:** LightGBM + XGBoost cover same use cases, fewer bugs

---

## Next Steps

1. Implement Fix #1 (data type mapping) - 10 minutes
2. Implement Fix #2 (remove CatBoost) - 5 minutes
3. Implement Fix #3 (per-experiment timeout) - 15 minutes
4. Test on leaf-classification locally
5. Deploy and run on full test suite

Total implementation time: ~30-45 minutes
Expected improvement: 13.6% → 45-55% medal rate

