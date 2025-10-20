# EstimateDuration Tool - Usage Examples

## Quick Reference

```python
# Basic usage - minimal parameters
EstimateDuration(
    task_type="image_classification",  # or tabular, nlp_classification, etc.
    dataset_size="medium"               # small/medium/large
)

# Advanced usage - all parameters
EstimateDuration(
    task_type="image_classification",
    dataset_size="medium",
    complexity="complex",               # simple/moderate/complex/very_complex
    num_parallel_models=3,
    description="Cassava leaf disease - ensemble strategy"
)
```

## Example 1: Early in Competition (Plenty of Time)

**Context:** 5 minutes elapsed, exploring image classification dataset with 50K images

**Input:**
```python
EstimateDuration(
    task_type="image_classification",
    dataset_size="medium",
    description="Cassava leaf disease classification - 50K images"
)
```

**Output:**
```
⏱️  TASK DURATION ESTIMATE
============================================================

Task: Cassava leaf disease classification - 50K images

📋 TASK DETAILS:
• Type: image_classification
• Dataset size: medium
• Complexity: moderate
• Parallel models: 1

⏰ TIME ESTIMATES (A100 GPU):
• Optimistic: 12.0 minutes
• Typical: 20.0 minutes
• Pessimistic: 30.0 minutes

📊 TIME BUDGET STATUS:
• Total budget: 30.0 minutes
• Elapsed: 5.0 minutes (16.7% used)
• Remaining: 25.0 minutes
• Estimated need: 20.0 minutes
• Time ratio: 0.80x (estimate/remaining)

🎯 ADAPTIVE STRATEGY:
• Urgency: MEDIUM
• Recommended strategy: STANDARD
• Speed modifier: 1.0x
• Adjusted target: 20.0 minutes

📝 GUIDANCE:
You have 25.0 min remaining, need ~20.0 min.
Time is comfortable - use STANDARD strategy:
• Run 3-fold CV (standard)
• Use medium models (B3 for CV, distilbert for NLP)
• Single model or small ensemble (2 models)
• Focus on core approach
• Should finish on time

🤖 MODEL RECOMMENDATIONS (STANDARD strategy):
• EfficientNet-B3, 3-fold CV, 8-10 epochs, batch_size=256, MixUp
```

**Agent Decision:** Proceed with EfficientNet-B3, 3-fold CV, 8-10 epochs

---

## Example 2: Very Early - Optimize for Accuracy

**Context:** 3 minutes elapsed, small tabular dataset (80K rows)

**Input:**
```python
EstimateDuration(
    task_type="tabular",
    dataset_size="small",
    complexity="complex",
    num_parallel_models=3,
    description="House prices - LightGBM + XGBoost + CatBoost ensemble"
)
```

**Output:**
```
⏱️  TASK DURATION ESTIMATE
============================================================

Task: House prices - LightGBM + XGBoost + CatBoost ensemble

📋 TASK DETAILS:
• Type: tabular
• Dataset size: small
• Complexity: complex
• Parallel models: 3

⏰ TIME ESTIMATES (A100 GPU):
• Optimistic: 6.3 minutes
• Typical: 11.3 minutes
• Pessimistic: 18.0 minutes

📊 TIME BUDGET STATUS:
• Total budget: 30.0 minutes
• Elapsed: 3.0 minutes (10.0% used)
• Remaining: 27.0 minutes
• Estimated need: 11.3 minutes
• Time ratio: 0.42x (estimate/remaining)

🎯 ADAPTIVE STRATEGY:
• Urgency: LOW
• Recommended strategy: FULL
• Speed modifier: 1.0x
• Adjusted target: 11.3 minutes

📝 GUIDANCE:
You have 27.0 min remaining, need ~11.3 min.
Time is abundant - use FULL strategy:
• Run complete CV (3-5 folds)
• Use larger models (B4/B5 for CV, DeBERTa for NLP)
• Consider ensemble (2-3 models in parallel)
• Take time for proper validation
• Aim for best possible score

🤖 MODEL RECOMMENDATIONS (FULL strategy):
• LightGBM + XGBoost + CatBoost ensemble, 5-fold CV, extensive features, stacking
```

**Agent Decision:** Go for full ensemble with extensive feature engineering (time allows)

---

## Example 3: Running Behind Schedule

**Context:** 20 minutes elapsed, need to finish training NLP model

**Input:**
```python
EstimateDuration(
    task_type="nlp_classification",
    dataset_size="medium",
    description="Sentiment analysis - need to finish quickly"
)
```

**Output:**
```
⏱️  TASK DURATION ESTIMATE
============================================================

Task: Sentiment analysis - need to finish quickly

📋 TASK DETAILS:
• Type: nlp_classification
• Dataset size: medium
• Complexity: moderate
• Parallel models: 1

⏰ TIME ESTIMATES (A100 GPU):
• Optimistic: 8.0 minutes
• Typical: 15.0 minutes
• Pessimistic: 22.0 minutes

📊 TIME BUDGET STATUS:
• Total budget: 30.0 minutes
• Elapsed: 20.0 minutes (66.7% used)
• Remaining: 10.0 minutes
• Estimated need: 15.0 minutes
• Time ratio: 1.50x (estimate/remaining)

🎯 ADAPTIVE STRATEGY:
• Urgency: CRITICAL
• Recommended strategy: EMERGENCY
• Speed modifier: 0.5x
• Adjusted target: 7.5 minutes

📝 GUIDANCE:
You have 10.0 min remaining, need ~15.0 min.
Time is CRITICAL - use EMERGENCY strategy:
• NO CV - single train/val split
• Smallest viable model (B0 for CV, tiny for NLP)
• Minimal epochs (3-5 max)
• Large batch size (maximize speed)
• Consider simple baseline (LR, small GBDT)
• Accept lower score to finish on time
• OR: Skip training, use pretrained model directly if possible

🤖 MODEL RECOMMENDATIONS (EMERGENCY strategy):
• TF-IDF + LogisticRegression (fastest) OR distilbert 1 epoch single split
```

**Agent Decision:** Switch to TF-IDF + LogisticRegression (fastest option) or distilbert with 1 epoch only

---

## Example 4: Large Dataset, Tight Timeline

**Context:** 12 minutes elapsed, large segmentation task (100K images)

**Input:**
```python
EstimateDuration(
    task_type="image_segmentation",
    dataset_size="large",
    complexity="moderate"
)
```

**Output:**
```
⏱️  TASK DURATION ESTIMATE
============================================================

📋 TASK DETAILS:
• Type: image_segmentation
• Dataset size: large
• Complexity: moderate
• Parallel models: 1

⏰ TIME ESTIMATES (A100 GPU):
• Optimistic: 25.0 minutes
• Typical: 40.0 minutes
• Pessimistic: 60.0 minutes

📊 TIME BUDGET STATUS:
• Total budget: 30.0 minutes
• Elapsed: 12.0 minutes (40.0% used)
• Remaining: 18.0 minutes
• Estimated need: 40.0 minutes
• Time ratio: 2.22x (estimate/remaining)

🎯 ADAPTIVE STRATEGY:
• Urgency: CRITICAL
• Recommended strategy: EMERGENCY
• Speed modifier: 0.5x
• Adjusted target: 20.0 minutes

📝 GUIDANCE:
You have 18.0 min remaining, need ~40.0 min.
Time is CRITICAL - use EMERGENCY strategy:
• NO CV - single train/val split
• Smallest viable model (B0 for CV, tiny for NLP)
• Minimal epochs (3-5 max)
• Large batch size (maximize speed)
• Consider simple baseline (LR, small GBDT)
• Accept lower score to finish on time
• OR: Skip training, use pretrained model directly if possible

🤖 MODEL RECOMMENDATIONS (EMERGENCY strategy):
• U-Net + ResNet-34 backbone, 256x256 tiles, single split, 5 epochs
```

**Agent Decision:** Realize this task is too large for 30-min budget. Switch to emergency strategy: smaller tiles (256x256), ResNet-34 backbone, single split, 5 epochs max.

---

## Example 5: Parallel Training Strategy

**Context:** 8 minutes elapsed, planning ensemble for image classification

**Input:**
```python
EstimateDuration(
    task_type="image_classification",
    dataset_size="medium",
    complexity="moderate",
    num_parallel_models=3,
    description="Parallel: EfficientNet-B3 + ResNet-50 + ViT-small"
)
```

**Output:**
```
⏱️  TASK DURATION ESTIMATE
============================================================

Task: Parallel: EfficientNet-B3 + ResNet-50 + ViT-small

📋 TASK DETAILS:
• Type: image_classification
• Dataset size: medium
• Complexity: moderate
• Parallel models: 3

⏰ TIME ESTIMATES (A100 GPU):
• Optimistic: 18.0 minutes
• Typical: 30.0 minutes
• Pessimistic: 45.0 minutes

📊 TIME BUDGET STATUS:
• Total budget: 30.0 minutes
• Elapsed: 8.0 minutes (26.7% used)
• Remaining: 22.0 minutes
• Estimated need: 30.0 minutes
• Time ratio: 1.36x (estimate/remaining)

🎯 ADAPTIVE STRATEGY:
• Urgency: CRITICAL
• Recommended strategy: EMERGENCY
• Speed modifier: 0.5x
• Adjusted target: 15.0 minutes

📝 GUIDANCE:
You have 22.0 min remaining, need ~30.0 min.
Time is CRITICAL - use EMERGENCY strategy:
• NO CV - single train/val split
• Smallest viable model (B0 for CV, tiny for NLP)
• Minimal epochs (3-5 max)
• Large batch size (maximize speed)
• Consider simple baseline (LR, small GBDT)
• Accept lower score to finish on time
• OR: Skip training, use pretrained model directly if possible

🤖 MODEL RECOMMENDATIONS (EMERGENCY strategy):
• EfficientNet-B0 or ResNet-34, single train/val split, 3-5 epochs
```

**Agent Decision:** Parallel training of 3 models exceeds budget. Options:
1. Reduce to 2 models in parallel (will fit in ~22-25 min)
2. Train single best model (B3, standard strategy)
3. Switch to emergency: single B0 model, no CV

**Revised strategy:**
```python
# Option 1: 2 models in parallel (more feasible)
EstimateDuration(
    task_type="image_classification",
    dataset_size="medium",
    num_parallel_models=2
)
# → Estimate: ~26 min (tight but feasible)
```

---

## Example 6: Quick Baseline (Simple Task)

**Context:** 4 minutes elapsed, small time series dataset

**Input:**
```python
EstimateDuration(
    task_type="time_series",
    dataset_size="small",
    complexity="simple"
)
```

**Output:**
```
⏱️  TASK DURATION ESTIMATE
============================================================

📋 TASK DETAILS:
• Type: time_series
• Dataset size: small
• Complexity: simple
• Parallel models: 1

⏰ TIME ESTIMATES (A100 GPU):
• Optimistic: 2.1 minutes
• Typical: 3.5 minutes
• Pessimistic: 5.6 minutes

📊 TIME BUDGET STATUS:
• Total budget: 30.0 minutes
• Elapsed: 4.0 minutes (13.3% used)
• Remaining: 26.0 minutes
• Estimated need: 3.5 minutes
• Time ratio: 0.13x (estimate/remaining)

🎯 ADAPTIVE STRATEGY:
• Urgency: LOW
• Recommended strategy: FULL
• Speed modifier: 1.0x
• Adjusted target: 3.5 minutes

📝 GUIDANCE:
You have 26.0 min remaining, need ~3.5 min.
Time is abundant - use FULL strategy:
• Run complete CV (3-5 folds)
• Use larger models (B4/B5 for CV, DeBERTa for NLP)
• Consider ensemble (2-3 models in parallel)
• Take time for proper validation
• Aim for best possible score

🤖 MODEL RECOMMENDATIONS (FULL strategy):
• GBDT ensemble + LSTM/Transformer, extensive features, TimeSeriesSplit(n=5)
```

**Agent Decision:** Task is very quick. Plenty of time to try multiple approaches:
1. Quick baseline: LightGBM (3-5 min)
2. Feature engineering (5-10 min)
3. Try XGBoost + CatBoost ensemble (5-8 min)
4. Add LSTM if time allows (10-15 min)

Total: ~25-30 min → Fits perfectly in budget

---

## Example 7: Audio Classification

**Context:** 6 minutes elapsed, medium audio dataset

**Input:**
```python
EstimateDuration(
    task_type="audio",
    dataset_size="medium",
    description="Bird species classification from audio clips"
)
```

**Output:**
```
⏱️  TASK DURATION ESTIMATE
============================================================

Task: Bird species classification from audio clips

📋 TASK DETAILS:
• Type: audio
• Dataset size: medium
• Complexity: moderate
• Parallel models: 1

⏰ TIME ESTIMATES (A100 GPU):
• Optimistic: 12.0 minutes
• Typical: 20.0 minutes
• Pessimistic: 30.0 minutes

📊 TIME BUDGET STATUS:
• Total budget: 30.0 minutes
• Elapsed: 6.0 minutes (20.0% used)
• Remaining: 24.0 minutes
• Estimated need: 20.0 minutes
• Time ratio: 0.83x (estimate/remaining)

🎯 ADAPTIVE STRATEGY:
• Urgency: MEDIUM
• Recommended strategy: STANDARD
• Speed modifier: 1.0x
• Adjusted target: 20.0 minutes

📝 GUIDANCE:
You have 24.0 min remaining, need ~20.0 min.
Time is comfortable - use STANDARD strategy:
• Run 3-fold CV (standard)
• Use medium models (B3 for CV, distilbert for NLP)
• Single model or small ensemble (2 models)
• Focus on core approach
• Should finish on time

🤖 MODEL RECOMMENDATIONS (STANDARD strategy):
• Mel-spectrogram + EfficientNet-B2/ResNet-50, 3-fold CV, 10 epochs
```

**Agent Decision:** Convert audio to mel-spectrograms, use EfficientNet-B2, 3-fold CV

---

## Recommended Workflow

### Standard Competition Flow

```
1. Data Exploration (3-5 min)
   ├─ Read train/test data
   ├─ Check shapes, dtypes, distributions
   └─ Identify task type and dataset size

2. EstimateDuration (1 min)
   ├─ Input: task_type, dataset_size, complexity
   └─ Output: Strategy recommendation + time estimate

3. Consult Oracle (2-3 min)
   ├─ Share: EstimateDuration output + data insights
   ├─ Get: Detailed strategy validation
   └─ Confirm: Model choice, CV strategy, hyperparameters

4. Write Training Code (3-5 min)
   ├─ Based on recommended strategy
   ├─ Use model recommendations
   └─ Configure folds/epochs/batch_size per guidance

5. Execute Training (10-20 min)
   ├─ Monitor with ReadBashOutput
   ├─ Check ElapsedTime every 5-10 min
   └─ Adjust if needed (kill slow jobs, reduce folds)

6. Generate Predictions (2-5 min)
   ├─ Run inference
   └─ Create submission.csv

Total: 20-35 minutes (fits 30±10 min budget)
```

### Decision Points

**After EstimateDuration, ask:**

1. **Does estimate fit in remaining time?**
   - Yes → Proceed with recommended strategy
   - No → Downgrade strategy (FULL → STANDARD → FAST → EMERGENCY)

2. **Is urgency level acceptable?**
   - LOW/MEDIUM → Good, proceed
   - HIGH → Consider simplifying (reduce folds, smaller model)
   - CRITICAL → Must simplify or will run out of time

3. **Are model recommendations appropriate?**
   - Yes → Use them
   - No → Consult Oracle for alternatives

---

## Tips for Using EstimateDuration

### When to Use

✅ **DO use:**
- After data exploration, before writing train.py
- When planning ensemble (check if parallel training fits)
- When stuck deciding between approaches
- When considering adding more models mid-competition

✅ **DO NOT use:**
- During active training (use ElapsedTime instead)
- For trivial tasks (already obvious)

### Interpreting Results

**Time Ratio:**
- `< 0.6`: Plenty of time → optimize for accuracy
- `0.6 - 1.0`: Comfortable → standard approach
- `1.0 - 1.3`: Tight → optimize for speed
- `> 1.3`: Critical → emergency mode

**Strategy Levels:**
- **FULL**: Maximum accuracy, time abundant
- **STANDARD**: Balanced, should finish on time
- **FAST**: Speed-optimized, cutting corners
- **EMERGENCY**: Survival mode, finish at any cost

### Combining with Other Tools

**EstimateDuration + Memory:**
```python
# Get memory recommendation
memory.get_strategy_for_competition("image_classification", "medium", 30)

# Validate with EstimateDuration
EstimateDuration(task_type="image_classification", dataset_size="medium")

# Combine insights for Oracle
# "Memory recommends EfficientNet-B3 ensemble.
#  EstimateDuration suggests STANDARD strategy (20 min estimate, 22 min remaining).
#  Confirm this aligns with grandmaster assessment?"
```

**EstimateDuration + ElapsedTime:**
```python
# Plan initial strategy
EstimateDuration(...) → "20 min estimate, STANDARD strategy"

# 10 min later, check progress
ElapsedTime() → "15 min elapsed, 15 min remaining"

# Re-estimate if needed
EstimateDuration(...) → "Still need 12 min, now FAST strategy recommended"
```

---

## Summary

**EstimateDuration provides:**
1. ⏰ Realistic time estimates based on empirical A100 data
2. 🎯 Adaptive strategy recommendations (FULL/STANDARD/FAST/EMERGENCY)
3. 🤖 Specific model recommendations tailored to time constraints
4. 📊 Real-time budget tracking
5. 📝 Actionable guidance for decision-making

**Use it to:**
- Plan competitions before training
- Make informed speed vs accuracy tradeoffs
- Avoid running out of time
- Choose appropriate models and CV strategies
- Validate approaches with Oracle

**Bottom line:** Smart time estimation prevents wasted effort and ensures you finish on time!
