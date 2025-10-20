# EstimateDuration - Quick Reference Card

## One-Line Summary
**Smart time estimation + adaptive strategy recommendations based on remaining budget**

---

## Basic Usage

```python
EstimateDuration(
    task_type="image_classification",  # Required
    dataset_size="medium"               # Required
)
```

---

## Task Types

| Code | Description |
|------|-------------|
| `image_classification` | Standard image classification |
| `image_segmentation` | Semantic/instance segmentation |
| `object_detection` | Bounding box detection |
| `tabular` | Tabular data (GBDT models) |
| `nlp_classification` | Text classification |
| `time_series` | Time series forecasting |
| `audio` | Audio classification |

---

## Dataset Sizes

| Size | Images | Rows (Tabular) | Texts |
|------|--------|----------------|-------|
| `small` | <10K | <100K | <50K |
| `medium` | 10K-100K | 100K-1M | 50K-500K |
| `large` | >100K | >1M | >500K |

---

## Strategy Levels (Auto-Selected)

| Time Ratio | Strategy | CV | Model Size | Epochs | Example |
|-----------|----------|----|-----------:|-------:|---------|
| ≤ 0.6 | **FULL** | 5-fold | B4/B5 | 10-15 | 10 min need, 20 min left |
| ≤ 1.0 | **STANDARD** | 3-fold | B3 | 8-10 | 15 min need, 16 min left |
| ≤ 1.3 | **FAST** | 2-fold | B2 | 6-8 | 12 min need, 10 min left |
| > 1.3 | **EMERGENCY** | None | B0 | 3-5 | 15 min need, 5 min left |

**Time Ratio = Estimated Time / Remaining Time**

---

## Quick Model Guide

### Image Classification
- EMERGENCY: `EfficientNet-B0`, no CV, 3-5 epochs
- FAST: `EfficientNet-B2`, 2-fold, 6-8 epochs
- STANDARD: `EfficientNet-B3`, 3-fold, 8-10 epochs, MixUp
- FULL: `EfficientNet-B4/B5`, 5-fold, ensemble

### Tabular
- EMERGENCY: `LightGBM`, no CV, default params
- FAST: `LightGBM`, 2-fold, minimal features
- STANDARD: `LightGBM + XGBoost`, 3-fold
- FULL: `LightGBM + XGBoost + CatBoost`, 5-fold, stacking

### NLP
- EMERGENCY: `TF-IDF + LogisticRegression`
- FAST: `distilbert`, max_len=128, 1 epoch, 2-fold
- STANDARD: `DeBERTa-small`, max_len=256, 2 epochs, 3-fold
- FULL: `DeBERTa-base`, max_len=512, 3 epochs, 5-fold

---

## Typical Time Estimates (A100 GPU)

| Task | Small | Medium | Large |
|------|-------|--------|-------|
| Image Classification | 8-18 min | 12-30 min | 20-50 min |
| Tabular | 3-8 min | 5-12 min | 8-25 min |
| NLP | 5-12 min | 8-22 min | 15-40 min |

---

## Decision Tree

```
1. Run EstimateDuration after data exploration
   ↓
2. Check Time Ratio
   ↓
   ├─ ≤ 0.6 → Plenty of time
   │          Use FULL strategy
   │          Go for best score
   │
   ├─ 0.6-1.0 → Comfortable
   │            Use STANDARD strategy
   │            Should finish on time
   │
   ├─ 1.0-1.3 → Tight
   │            Use FAST strategy
   │            Cut corners to finish
   │
   └─ > 1.3 → CRITICAL
              Use EMERGENCY strategy
              Simplify everything
```

---

## Common Patterns

### Pattern 1: Standard Flow
```python
# After data exploration (5 min elapsed)
EstimateDuration(
    task_type="image_classification",
    dataset_size="medium"
)
# → "20 min estimate, 25 min left → STANDARD: B3, 3-fold"
```

### Pattern 2: Parallel Training
```python
# Check if ensemble fits
EstimateDuration(
    task_type="tabular",
    dataset_size="small",
    num_parallel_models=3
)
# → "12 min estimate, 22 min left → FULL: ensemble feasible"
```

### Pattern 3: Re-estimate Mid-Competition
```python
# 20 min elapsed, check if still on track
EstimateDuration(
    task_type="nlp_classification",
    dataset_size="large"
)
# → "25 min estimate, 10 min left → EMERGENCY: switch to TF-IDF"
```

---

## Integration with Other Tools

### With Oracle
```
EstimateDuration → "STANDARD strategy: B3, 3-fold, 20 min"
       ↓
Oracle → "Confirmed. Also add MixUp augmentation."
```

### With ElapsedTime
```
T=5min:  EstimateDuration → Plan strategy
T=15min: ElapsedTime → Check progress
T=20min: EstimateDuration → Re-evaluate if needed
```

---

## When to Use

✅ **DO use:**
- After data exploration, before writing train.py
- When planning ensemble/parallel training
- When deciding between model sizes
- Mid-competition if running behind

❌ **DON'T use:**
- During active training (use ElapsedTime)
- For trivial tasks (<5 min obvious)

---

## Output Format

```
⏱️  TASK DURATION ESTIMATE
============================

📋 TASK DETAILS: [type, size, complexity, parallel]
⏰ TIME ESTIMATES: [min, typical, max]
📊 TIME BUDGET: [elapsed, remaining, ratio]
🎯 ADAPTIVE STRATEGY: [urgency, strategy, modifier]
📝 GUIDANCE: [what to do]
🤖 MODEL RECOMMENDATIONS: [specific models]
```

---

## Pro Tips

1. **Use early**: Estimate BEFORE writing code, not after
2. **Trust the ratio**: Time ratio > 1.3 = you WILL run out of time
3. **Follow guidance**: Model recommendations are calibrated for A100
4. **Re-estimate**: Check again if falling behind schedule
5. **Combine with Oracle**: Use estimate to inform Oracle consultation

---

## Complexity Levels

| Level | Description | Multiplier |
|-------|-------------|------------|
| `simple` | Single model, basic features | 0.7x |
| `moderate` | Standard approach (default) | 1.0x |
| `complex` | Ensemble, feature engineering | 1.5x |
| `very_complex` | Large ensemble, complex pipelines | 2.0x |

---

## Emergency Strategies by Task

| Task | Emergency Strategy |
|------|-------------------|
| Image | B0, single split, 3-5 epochs, batch_size=512 |
| Tabular | LightGBM only, no features, single split |
| NLP | TF-IDF + LR (skip deep learning) |
| Segmentation | U-Net + ResNet-34, 256x256 tiles, 5 epochs |
| Detection | YOLOv5n pretrained, fine-tune 3 epochs |
| Time Series | LightGBM, basic lag features, single split |
| Audio | Tiny CNN on mel-spec, 5 epochs |

---

## Key Formulas

```python
# Final time estimate
final_time = base_estimate
             × complexity_multiplier
             × parallel_efficiency

# Time ratio (determines strategy)
time_ratio = estimated_time / time_remaining

# Adjusted target (with speed modifier)
adjusted_time = estimated_time × speed_modifier
```

---

## Quick Checklist

Before training, verify:
- [ ] EstimateDuration says strategy is STANDARD or better
- [ ] Time ratio < 1.3 (have enough time)
- [ ] Model recommendation makes sense
- [ ] CV folds appropriate for time budget
- [ ] Oracle confirmed approach

---

**Status:** ✅ Production-ready
**Tests:** 23/23 passing
**Docs:** Complete
**Integration:** Registered in agent_v5

**→ Use it to avoid running out of time!**
