# EstimateTaskDuration Tool

## Overview

The `EstimateTaskDurationTool` provides heuristic-based estimates for how long common data science and machine learning tasks should take. This helps agents:

- **Plan work**: Understand time budgets for different tasks
- **Set expectations**: Know when tasks are taking too long
- **Detect stalls**: Identify when processes may be hung or inefficient
- **Optimize workflows**: Choose between quick iterations vs long training runs

## Usage

### Basic Example

```python
EstimateTaskDuration({
    "task_type": "train_simple_model"
})
```

**Output:**
```
⏱️  Task Duration Estimate: train_simple_model

📊 Estimated Duration:
   ⚡ Best case:  10s
   📈 Typical:    1m 0s
   ⚠️  Worst case: 5m 0s

Parameters:
   • Data size: medium
   • Complexity: moderate

💡 Recommendations:
   • This is a quick task - should complete in under a minute
```

### With Data Size and Complexity

```python
EstimateTaskDuration({
    "task_type": "train_complex_model",
    "data_size": "large",
    "complexity": "complex",
    "additional_context": "using GPU with 100+ features"
})
```

**Output:**
```
⏱️  Task Duration Estimate: train_complex_model

📊 Estimated Duration:
   ⚡ Best case:  3m 0s
   📈 Typical:    6m 0s
   ⚠️  Worst case: 1h 0m

Parameters:
   • Data size: large
   • Complexity: complex
   • Context: using GPU with 100+ features

💡 Recommendations:
   • This is a longer task - budget 10-30 minutes
   • Set timeout to at least 1h 30m
   • Consider checkpointing for tasks > 30 minutes
   • Monitor with ReadBashOutput to detect stalls
   • Consider early stopping if no improvement
```

## Supported Task Types

### Data Exploration
- `load_data`
- `explore_data`
- `data_profiling`
- `visualize_data`

### Data Preprocessing
- `clean_data`
- `feature_engineering`
- `handle_missing_values`
- `encode_categorical`
- `scale_features`

### Model Training
- `train_simple_model`
- `train_complex_model`
- `train_deep_learning`
- `hyperparameter_tuning`
- `cross_validation`

### Model Evaluation
- `evaluate_model`
- `generate_predictions`
- `calculate_metrics`

### Large Data Operations
- `process_large_dataset`
- `merge_large_dataframes`
- `aggregate_data`

### Code Operations
- `write_script`
- `debug_code`
- `refactor_code`

### File Operations
- `read_small_file`
- `read_large_file`
- `write_file`

### Kaggle Specific
- `understand_competition`
- `prepare_submission`
- `ensemble_models`

## Parameters

### task_type (required)
Type of task to estimate. See supported task types above.

### data_size (optional)
Size of data being processed:
- `"small"` - <1GB
- `"medium"` - 1-10GB (default)
- `"large"` - >10GB

Affects data operation estimates (load, process, merge, etc.)

### complexity (optional)
Complexity level:
- `"simple"` - Basic operations
- `"moderate"` - Standard workflows (default)
- `"complex"` - Advanced/intensive operations

Affects model training and code task estimates.

### additional_context (optional)
Additional context that might affect duration:
- `"using GPU"` - Speeds up deep learning (3x faster estimate)
- `"distributed training"` - May affect timing
- `"large dataset with 100+ columns"` - Increases processing time

## Fuzzy Matching

The tool supports fuzzy matching for task types:

```python
# These all match valid task types:
EstimateTaskDuration({"task_type": "train_model"})  # Matches train_simple_model
EstimateTaskDuration({"task_type": "load"})         # Matches load_data
EstimateTaskDuration({"task_type": "feature"})      # Matches feature_engineering
```

## Integration with Kaggle Agent

The tool is automatically registered in the Kaggle agent:

```python
# In kaggle_agent.py
from agent_v5.tools.estimate_duration import EstimateTaskDurationTool

# During initialization
self.tools.register(EstimateTaskDurationTool(
    workspace_dir=workspace_dir
))
```

## Use Cases

### 1. Planning Workflow
Before starting a competition:
```python
EstimateTaskDuration({"task_type": "explore_data"})
EstimateTaskDuration({"task_type": "feature_engineering"})
EstimateTaskDuration({"task_type": "train_complex_model", "data_size": "large"})
EstimateTaskDuration({"task_type": "prepare_submission"})
```

### 2. Setting Timeouts
When running background tasks:
```python
# Get estimate
estimate = EstimateTaskDuration({"task_type": "train_deep_learning"})
# Use max_duration * 1.5 as timeout

# Run with appropriate timeout
Bash({
    "command": "python train.py",
    "background": true,
    "timeout": 7200000  # 2 hours based on estimate
})
```

### 3. Detecting Stalls
When monitoring background processes:
```python
# If process has been running longer than max_duration:
# - Check ReadBashOutput for progress
# - Consider KillShell if stalled
# - Investigate bottlenecks
```

## Implementation Details

### Estimate Modifiers

1. **Data Size Modifiers:**
   - `large` → 2-4x increase for data operations
   - `small` → 0.5-0.8x decrease

2. **Complexity Modifiers:**
   - `complex` → 1.5-2.5x increase for model/code tasks
   - `simple` → 0.6-0.8x decrease

3. **GPU Acceleration:**
   - Deep learning with GPU → 0.3-0.4x decrease (3x faster)

### Base Estimates

All estimates are tuples of (min, typical, max) in seconds:
```python
"train_simple_model": (10, 60, 300)     # 10s to 5m
"train_complex_model": (120, 600, 3600) # 2m to 1h
"train_deep_learning": (300, 1800, 7200) # 5m to 2h
```

## Testing

Comprehensive test suite with 11 tests:
```bash
python -m pytest agent_v5/tests/test_estimate_duration.py -v
```

Tests cover:
- Simple task estimates
- Data size modifiers
- Complexity modifiers
- GPU acceleration
- Fuzzy matching
- Unknown tasks
- Recommendations
- Duration formatting

## Future Enhancements

Potential improvements:
1. **Historical data**: Learn from actual task durations
2. **Dataset-specific estimates**: Adjust based on actual data characteristics
3. **Hardware profiles**: Different estimates for CPU/GPU/TPU
4. **Dynamic updates**: Adjust estimates based on observed performance
5. **Confidence intervals**: Add uncertainty bounds to estimates
