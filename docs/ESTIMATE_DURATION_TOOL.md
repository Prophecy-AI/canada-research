# EstimateDuration Tool Documentation

## Overview

The `EstimateDuration` tool provides **intelligent task duration estimation with adaptive time control** for Kaggle competitions and ML tasks. It combines:

1. **Task-specific duration estimates** based on empirical A100 GPU performance
2. **Real-time budget tracking** to monitor elapsed vs remaining time
3. **Adaptive strategy recommendations** (full/standard/fast/emergency modes)
4. **Specific model recommendations** tailored to time constraints

## Key Features

### 1. Smart Duration Estimation

Estimates time based on:
- **Task type**: image_classification, tabular, NLP, etc.
- **Dataset size**: small/medium/large
- **Complexity**: simple/moderate/complex/very_complex
- **Parallel training**: 1-4 models simultaneously

### 2. Adaptive Time Control

Automatically adjusts strategy based on:
- **Time ratio** = estimated_time / time_remaining
- **Urgency levels**: low/medium/high/critical
- **Strategy modes**: full/standard/fast/emergency

### 3. Hardware-Aware

Calibrated for actual hardware specs:
- NVIDIA A100 40GB GPU
- 36 CPU cores
- 440GB RAM

## Usage

### Basic Usage

```python
# Estimate time for image classification with medium dataset
EstimateDuration(
    task_type="image_classification",
    dataset_size="medium"
)
```

### Advanced Usage

```python
# Complex tabular task with ensemble
EstimateDuration(
    task_type="tabular",
    dataset_size="large",
    complexity="complex",
    num_parallel_models=3,
    description="Customer churn prediction with extensive feature engineering"
)
```

## Input Parameters

### Required Parameters

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `task_type` | string | Type of ML task | `image_classification`, `image_segmentation`, `object_detection`, `tabular`, `nlp_classification`, `time_series`, `audio` |
| `dataset_size` | string | Dataset size category | `small`, `medium`, `large` |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `complexity` | string | `"moderate"` | Task complexity: `simple`, `moderate`, `complex`, `very_complex` |
| `num_parallel_models` | integer | `1` | Number of models to train in parallel (1-4) |
| `description` | string | `""` | Brief task description for context |

## Dataset Size Guidelines

### Computer Vision
- **small**: <10K images
- **medium**: 10K-100K images
- **large**: >100K images

### Tabular
- **small**: <100K rows
- **medium**: 100K-1M rows
- **large**: >1M rows

### NLP
- **small**: <50K texts
- **medium**: 50K-500K texts
- **large**: >500K texts

## Time Estimation Logic

### Base Estimates (A100 GPU)

| Task Type | Small | Medium | Large |
|-----------|-------|--------|-------|
| Image Classification | 8-18 min | 12-30 min | 20-50 min |
| Image Segmentation | 10-22 min | 15-35 min | 25-60 min |
| Object Detection | 8-18 min | 12-28 min | 20-45 min |
| Tabular | 3-8 min | 5-12 min | 8-25 min |
| NLP Classification | 5-12 min | 8-22 min | 15-40 min |
| Time Series | 3-8 min | 5-15 min | 10-28 min |
| Audio | 8-18 min | 12-30 min | 20-45 min |

### Complexity Multipliers

- **simple** (0.7x): Single model, basic features
- **moderate** (1.0x): Standard approach, 2-3 models
- **complex** (1.5x): Ensemble, extensive features
- **very_complex** (2.0x): Large ensemble, complex pipelines

### Parallel Training Efficiency

Training multiple models in parallel is more efficient than sequential:

- **1 model**: 1.0x (baseline)
- **2 models**: 1.3x (not 2x - efficient GPU sharing)
- **3 models**: 1.5x (not 3x)
- **4 models**: 1.8x (not 4x)

## Adaptive Strategies

The tool automatically recommends strategy based on time remaining:

### Strategy Selection Logic

```
Time Ratio = Estimated Time / Remaining Time

≤ 0.6x  → FULL strategy (plenty of time)
≤ 1.0x  → STANDARD strategy (comfortable)
≤ 1.3x  → FAST strategy (tight but feasible)
> 1.3x  → EMERGENCY strategy (critical)
```

### Strategy Details

#### FULL Strategy (Time Ratio ≤ 0.6)
**When**: You have abundant time
**Approach**: Maximize accuracy
- 5-fold CV
- Larger models (EfficientNet-B4/B5, DeBERTa-base)
- Ensemble 2-3 models
- Full augmentations (MixUp, CutMix, TTA)
- Extensive hyperparameter tuning

**Example**: 10 min elapsed, 20 min remaining, need 12 min → ratio = 0.6 ✓

#### STANDARD Strategy (Time Ratio ≤ 1.0)
**When**: Comfortable time window
**Approach**: Balanced accuracy/speed
- 3-fold CV
- Medium models (EfficientNet-B3, distilbert)
- Single model or small ensemble (2 models)
- Standard augmentations
- Focus on core approach

**Example**: 15 min elapsed, 15 min remaining, need 14 min → ratio = 0.93 ✓

#### FAST Strategy (Time Ratio ≤ 1.3)
**When**: Time is tight but feasible
**Approach**: Optimize for speed
- 2-fold CV
- Smaller models (EfficientNet-B2, distilbert)
- Single model only
- Reduce epochs by 20-30%
- Increase batch size
- May cut corners

**Example**: 20 min elapsed, 10 min remaining, need 12 min → ratio = 1.2 ✓

#### EMERGENCY Strategy (Time Ratio > 1.3)
**When**: Critical time pressure
**Approach**: Finish at any cost
- NO CV - single train/val split
- Smallest viable model (B0, tiny transformers)
- Minimal epochs (3-5)
- Large batch size (maximize speed)
- Consider simple baseline (LR, small GBDT)
- Accept lower score to finish

**Example**: 25 min elapsed, 5 min remaining, need 10 min → ratio = 2.0 ✗

## Model Recommendations by Strategy

### Image Classification

| Strategy | Model | CV | Epochs | Batch Size | Augmentation |
|----------|-------|----|---------|-----------|-----------------------------|
| EMERGENCY | EfficientNet-B0 | Single split | 3-5 | 512 | Minimal |
| FAST | EfficientNet-B2 | 2-fold | 6-8 | 384 | Basic flips/rotations |
| STANDARD | EfficientNet-B3 | 3-fold | 8-10 | 256 | MixUp |
| FULL | EfficientNet-B4/B5 | 5-fold | 10-15 | 192 | MixUp+CutMix+TTA |

### Tabular

| Strategy | Models | CV | Features |
|----------|--------|----|-----------------------|
| EMERGENCY | LightGBM | Single split | None (raw features) |
| FAST | LightGBM | 2-fold | Minimal (date features) |
| STANDARD | LightGBM + XGBoost | 3-fold | Basic (interactions, aggregations) |
| FULL | LightGBM + XGBoost + CatBoost | 5-fold | Extensive + stacking |

### NLP Classification

| Strategy | Model | Max Length | Epochs | CV |
|----------|-------|------------|--------|----|
| EMERGENCY | TF-IDF + LR | N/A | N/A | Single split |
| FAST | distilbert | 128 | 1 | 2-fold |
| STANDARD | DeBERTa-v3-small | 256 | 2 | 3-fold |
| FULL | DeBERTa-v3-base | 512 | 3 | 5-fold |

## Example Output

```
⏱️  TASK DURATION ESTIMATE
============================================================

Task: Cassava leaf disease classification

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
• Elapsed: 8.5 minutes (28.3% used)
• Remaining: 21.5 minutes
• Estimated need: 20.0 minutes
• Time ratio: 0.93x (estimate/remaining)

🎯 ADAPTIVE STRATEGY:
• Urgency: MEDIUM
• Recommended strategy: STANDARD
• Speed modifier: 1.0x
• Adjusted target: 20.0 minutes

📝 GUIDANCE:
You have 21.5 min remaining, need ~20.0 min.
Time is comfortable - use STANDARD strategy:
• Run 3-fold CV (standard)
• Use medium models (B3 for CV, distilbert for NLP)
• Single model or small ensemble (2 models)
• Focus on core approach
• Should finish on time

���� MODEL RECOMMENDATIONS (STANDARD strategy):
• EfficientNet-B3, 3-fold CV, 8-10 epochs, batch_size=256, MixUp
```

## Integration with Agent

### Registration

The tool is automatically registered in `ResearchAgent`:

```python
from agent_v5.tools.estimate_duration import EstimateDurationTool

# In _register_core_tools():
self.tools.register(EstimateDurationTool(
    self.workspace_dir,
    self.start_time,
    total_budget_min=30.0
))
```

### Usage in Agent Workflow

**Recommended workflow:**

```
1. Data exploration (5 min)
   ↓
2. EstimateDuration (analyze task complexity)
   ↓
3. Get adaptive strategy recommendation
   ↓
4. Consult Oracle with time estimate
   ↓
5. Execute training with appropriate strategy
   ↓
6. ElapsedTime (check progress every 5-10 min)
   ↓
7. Adjust if needed (kill slow jobs, reduce folds)
```

## Best Practices

### When to Use EstimateDuration

✅ **DO use before training:**
- After data exploration, before writing train.py
- When deciding between single model vs ensemble
- When planning CV folds and epochs
- To inform Oracle consultation

✅ **DO use during competition:**
- When stuck deciding on approach
- When considering adding more models
- To validate time estimates

❌ **DON'T use:**
- After training already started (use ElapsedTime instead)
- For trivial tasks (<5 min)

### Combining with Other Tools

**EstimateDuration + Oracle:**
```
1. EstimateDuration → Get time estimate and strategy
2. Oracle → "Memory recommends X. EstimateDuration suggests Y strategy
   given Z minutes remaining. Confirm approach?"
```

**EstimateDuration + ElapsedTime:**
```
1. EstimateDuration (start) → Plan initial strategy
2. ElapsedTime (10 min later) → Check progress
3. EstimateDuration (updated) → Adjust strategy if needed
```

## Time Budget Scenarios

### Scenario 1: Plenty of Time
```
Elapsed: 5 min, Remaining: 25 min, Need: 12 min
→ Ratio: 0.48 → FULL strategy
→ "Run 5-fold CV with EfficientNet-B4, ensemble 2 models"
```

### Scenario 2: On Track
```
Elapsed: 12 min, Remaining: 18 min, Need: 16 min
→ Ratio: 0.89 → STANDARD strategy
→ "Run 3-fold CV with EfficientNet-B3, single model"
```

### Scenario 3: Running Behind
```
Elapsed: 18 min, Remaining: 12 min, Need: 15 min
→ Ratio: 1.25 → FAST strategy
→ "Reduce to 2-fold CV, use B2, increase batch size"
```

### Scenario 4: Critical
```
Elapsed: 24 min, Remaining: 6 min, Need: 12 min
→ Ratio: 2.0 → EMERGENCY strategy
→ "Single train/val split, B0 model, 3-5 epochs max"
```

## Technical Details

### Estimation Algorithm

1. **Base lookup**: Query `BASE_ESTIMATES[task_type][dataset_size]`
2. **Complexity adjustment**: Multiply by complexity factor (0.7-2.0x)
3. **Parallel adjustment**: Multiply by parallel efficiency (1.0-1.8x)
4. **Time calculation**: Get (min, typical, max) estimates
5. **Adaptive logic**: Compare typical_time vs time_remaining
6. **Strategy selection**: Choose full/standard/fast/emergency
7. **Model recommendations**: Lookup specific models for strategy

### Calibration

Estimates are calibrated on:
- **Hardware**: A100 40GB GPU, 36 cores, 440GB RAM
- **Frameworks**: PyTorch 2.x, LightGBM 4.x, Transformers 4.x
- **Typical configs**: Standard augmentations, mixed precision, optimal batch sizes

**Note**: Estimates assume efficient implementation. Poor code (CPU fallback, small batches, I/O bottlenecks) may be slower.

## Limitations

1. **Estimates are guidelines**: Actual time varies with:
   - Code quality (efficient vs inefficient)
   - Data characteristics (image size, text length)
   - Hyperparameters (learning rate, early stopping)

2. **Assumes standard approaches**: Custom architectures may differ

3. **No guarantee**: Complex competitions may exceed estimates

4. **Hardware-specific**: Calibrated for A100, may differ on other GPUs

## Future Enhancements

Potential improvements:
- [ ] Learn from past competitions (integrate with memory system)
- [ ] Per-epoch time estimation during training
- [ ] Dynamic re-estimation based on actual progress
- [ ] Hardware auto-detection and calibration
- [ ] Competition difficulty assessment

## Related Tools

- **ElapsedTime**: Check elapsed time and remaining budget
- **Oracle**: Strategic consultation with grandmaster AI
- **GPUValidate**: Verify GPU training is working
- **TodoWrite**: Track task progress
- **Memory**: Learn from past competition results

---

**Version**: 1.0.0
**Last Updated**: 2025-10-20
**Author**: Dylan (Celestra)
