# Time-Aware Task Scheduler

## Overview

The **TaskScheduler** is a time-aware scheduling system that helps agents make intelligent decisions about which tasks to run when operating under time constraints (e.g., Kaggle competition epochs with limited runtime).

### Key Features

- **Duration-aware scheduling**: Uses `EstimateTaskDuration` tool to predict task runtimes
- **Dynamic prioritization**: Re-prioritizes tasks based on remaining time budget
- **Dependency resolution**: Handles task dependencies via topological sort
- **Adaptive execution**: Switches strategy mid-execution based on time pressure
- **Smart task selection**: Balances task value, duration, and priority

---

## Core Concepts

### 1. Task Priority

Tasks have four priority levels:

```python
class TaskPriority(Enum):
    CRITICAL = 1    # Must run (e.g., load data, make submission)
    HIGH = 2        # Important but can be skipped (e.g., complex model training)
    MEDIUM = 3      # Nice to have (e.g., advanced feature engineering)
    LOW = 4         # Optional (e.g., extensive visualization)
```

**Behavior:**
- **CRITICAL** tasks run even if they exceed time budget
- **HIGH** tasks run if time permits
- **MEDIUM/LOW** tasks skipped when time is tight

### 2. Task Complexity

Characterizes the time/value trade-off:

```python
class TaskComplexity(Enum):
    QUICK_WIN = 1       # Fast, high value (e.g., basic preprocessing)
    EFFICIENT = 2       # Medium time, good value (e.g., simple model)
    EXPENSIVE = 3       # Long time, high value (e.g., complex model)
    EXPLORATORY = 4     # Variable time, uncertain value (e.g., experimentation)
```

**When time is tight, the scheduler prefers QUICK_WIN tasks over EXPENSIVE ones.**

### 3. Priority Scoring

Each task gets a dynamic priority score based on:

```python
def get_priority_score(time_remaining: float) -> float:
    # Base score from priority level (0-100)
    score = priority_base_score

    # Boost by efficiency (value/time ratio)
    score += efficiency_score * 20

    # Adjust based on time pressure
    if task_duration > time_remaining:
        # Task won't fit - penalize unless CRITICAL
        score *= 0.1
    elif task_duration > 0.5 * time_remaining:
        # Task takes > 50% of remaining time
        # Prefer quick wins
        if complexity == QUICK_WIN:
            score *= 1.5
        elif complexity == EXPENSIVE:
            score *= 0.7
    else:
        # Plenty of time - prefer high value
        score *= (1 + value_score)

    return score
```

---

## Usage

### Basic Example

```python
from agent_v5.task_scheduler import TaskScheduler, TaskPriority, TaskComplexity

# Create scheduler with 30 minute budget
scheduler = TaskScheduler(time_budget_seconds=30 * 60)

# Add tasks
scheduler.add_task(
    task_id="load_data",
    name="Load Training Data",
    execute_fn=load_data_function,
    priority=TaskPriority.CRITICAL,
    complexity=TaskComplexity.QUICK_WIN,
    value_score=1.0,
    # Manual duration estimates
    duration_min=5,
    duration_typical=10,
    duration_max=30
)

scheduler.add_task(
    task_id="train_model",
    name="Train LightGBM",
    execute_fn=train_model_function,
    priority=TaskPriority.HIGH,
    complexity=TaskComplexity.EFFICIENT,
    value_score=0.85,
    depends_on=["load_data"],  # Runs after load_data
    duration_min=60,
    duration_typical=120,
    duration_max=300
)

# Execute all tasks with adaptive scheduling
results = await scheduler.execute_all(adaptive=True)

# Check results
print(f"Completed: {len(results['completed'])} tasks")
print(f"Skipped: {len(results['skipped'])} tasks")
print(f"Failed: {len(results['failed'])} tasks")
```

### Integration with EstimateTaskDuration

Let the scheduler automatically estimate durations:

```python
from agent_v5.tools.estimate_duration import EstimateTaskDurationTool

# Create estimator
estimate_tool = EstimateTaskDurationTool(workspace_dir="/workspace")

# Pass to scheduler
scheduler = TaskScheduler(
    time_budget_seconds=60 * 60,
    estimate_tool=estimate_tool
)

# Add task with automatic estimation
scheduler.add_task(
    task_id="load_data",
    name="Load Large Dataset",
    execute_fn=load_data_function,
    priority=TaskPriority.CRITICAL,
    complexity=TaskComplexity.QUICK_WIN,
    value_score=1.0,
    # Use tool for estimation
    task_type="load_data",
    data_size_mb=5120  # 5 GB
)
```

---

## Real-World Scenarios

### Scenario 1: Plenty of Time (60 min budget)

**Strategy:** Run everything, optimize for quality

```python
scheduler = TaskScheduler(time_budget_seconds=60 * 60)

# Critical foundation
scheduler.add_task("load", ..., priority=CRITICAL, duration=10s)
scheduler.add_task("preprocess", ..., priority=CRITICAL, duration=30s)

# High-value models
scheduler.add_task("simple_model", ..., priority=HIGH, duration=60s)
scheduler.add_task("lgbm", ..., priority=HIGH, duration=180s)
scheduler.add_task("xgb", ..., priority=HIGH, duration=240s)

# Nice-to-haves
scheduler.add_task("neural_net", ..., priority=MEDIUM, duration=600s)
scheduler.add_task("hyperparameter_tuning", ..., priority=LOW, duration=1800s)

# Final submission
scheduler.add_task("ensemble", ..., priority=HIGH, duration=30s)
scheduler.add_task("submit", ..., priority=CRITICAL, duration=5s)

# Result: All HIGH/CRITICAL tasks run, maybe some MEDIUM/LOW
```

### Scenario 2: Time Crunch (15 min budget)

**Strategy:** Quick wins only, skip expensive tasks

```python
scheduler = TaskScheduler(time_budget_seconds=15 * 60)

# Critical foundation
scheduler.add_task("load", ..., priority=CRITICAL, duration=30s)
scheduler.add_task("basic_preprocess", ..., priority=CRITICAL, duration=60s)

# ONE fast model
scheduler.add_task("simple_model", ..., priority=HIGH, duration=120s)

# Skip advanced features (too slow)
scheduler.add_task("adv_features", ..., priority=LOW, duration=600s)

# Skip expensive models
scheduler.add_task("lgbm", ..., priority=LOW, duration=480s)
scheduler.add_task("xgb", ..., priority=LOW, duration=720s)

# Must submit
scheduler.add_task("submit", ..., priority=CRITICAL, duration=5s)

# Result: Load → Basic Preprocess → Simple Model → Submit
# (Skips advanced features and complex models)
```

### Scenario 3: Mid-Epoch Replanning

**Situation:** Started with 30 min, 20 min elapsed, 10 min remaining

```python
# Agent realizes it's running behind schedule
scheduler = TaskScheduler(time_budget_seconds=30 * 60)
scheduler.started_at = time.time() - (20 * 60)  # Simulate 20 min elapsed

# Re-evaluate remaining tasks
remaining_time = scheduler.get_time_remaining()  # 10 minutes

# Reprioritize: focus on finishing and submitting
scheduler.add_task("finish_lgbm", ..., priority=HIGH, duration=300s)
scheduler.add_task("skip_xgb", ..., priority=LOW, duration=600s)  # Will skip
scheduler.add_task("skip_nn", ..., priority=LOW, duration=900s)   # Will skip
scheduler.add_task("ensemble", ..., priority=HIGH, duration=30s)
scheduler.add_task("submit", ..., priority=CRITICAL, duration=5s)

# Result: Finish LightGBM → Ensemble → Submit
# (Skips XGBoost and Neural Network to make deadline)
```

---

## Advanced Features

### 1. Adaptive Scheduling

With `adaptive=True`, the scheduler continuously re-prioritizes:

```python
results = await scheduler.execute_all(
    adaptive=True,       # Re-prioritize after each task
    safety_margin=1.2    # Use 1.2x typical duration for estimates
)
```

**How it works:**
- After each task completes, recalculate time remaining
- Re-sort pending tasks by priority score
- Dynamically adjust which tasks run next

**Example:**
```
Initial plan: [TaskA (10min), TaskB (20min), TaskC (5min)]
Time budget: 30 min

After TaskA completes (actual: 15min instead of 10min):
  Time remaining: 15 min
  Re-prioritize: [TaskC (5min), TaskB (20min)]
  → Run TaskC first (quick win)
  → Skip TaskB (won't fit in 10min remaining)
```

### 2. Safety Margin

Accounts for estimation uncertainty:

```python
# Conservative: use 1.5x typical duration
results = await scheduler.execute_all(safety_margin=1.5)

# Aggressive: use 1.0x typical duration (no margin)
results = await scheduler.execute_all(safety_margin=1.0)
```

**Recommended values:**
- **1.5x**: Very conservative, for production systems
- **1.2x**: Balanced, for most use cases
- **1.0x**: Aggressive, when estimates are very accurate

### 3. Dependency Chains

Tasks can depend on other tasks:

```python
# Chain: load → preprocess → model → submission
scheduler.add_task("load", ..., depends_on=[])
scheduler.add_task("preprocess", ..., depends_on=["load"])
scheduler.add_task("model", ..., depends_on=["preprocess"])
scheduler.add_task("submit", ..., depends_on=["model"])
```

**Scheduler guarantees:**
- Dependencies run before dependents
- If a dependency is skipped, dependents are skipped
- If a CRITICAL dependency fails, execution aborts

### 4. Monitoring & Debugging

Track execution progress:

```python
# During execution
time_remaining = scheduler.get_time_remaining()
time_elapsed = scheduler.get_time_elapsed()

print(f"Progress: {time_elapsed:.0f}s / {scheduler.time_budget:.0f}s")
print(f"Remaining: {time_remaining:.0f}s")

# After execution
summary = scheduler.get_summary()
print(summary)
```

**Output:**
```
📊 Task Scheduler Summary
============================================================

Time Budget: 1800s
Time Elapsed: 1650s
Time Remaining: 150s

✅ COMPLETED (5):
   • Load Data (2.1s)
   • Basic Preprocessing (5.3s)
   • Train LightGBM (145.7s)
   • Ensemble Models (15.2s)
   • Create Submission (2.8s)

⏭️ SKIPPED (2):
   • Advanced Feature Engineering
   • Hyperparameter Tuning

❌ FAILED (0):
```

---

## Integration with Kaggle Agent

### Example: MLE-Bench Integration

```python
# In your Kaggle agent's run_epoch() method

async def run_epoch(self, time_budget_seconds: int):
    """Run one epoch with time-aware scheduling"""

    # Create scheduler
    scheduler = TaskScheduler(
        time_budget_seconds=time_budget_seconds,
        estimate_tool=self.estimate_tool
    )

    # Check dataset size
    train_size_mb = os.path.getsize("train.csv") / (1024 * 1024)

    # Plan tasks dynamically based on available time
    if time_budget_seconds > 3600:  # > 1 hour
        # Full pipeline
        self._add_full_pipeline(scheduler, train_size_mb)
    elif time_budget_seconds > 1800:  # 30-60 min
        # Standard pipeline
        self._add_standard_pipeline(scheduler, train_size_mb)
    else:  # < 30 min
        # Quick pipeline
        self._add_quick_pipeline(scheduler, train_size_mb)

    # Execute with monitoring
    results = await scheduler.execute_all(adaptive=True)

    # Report results
    if results["failed"]:
        raise Exception(f"Critical tasks failed: {results['failed']}")

    if not results["completed"]:
        raise Exception("No tasks completed")

    return {
        "completed_tasks": len(results["completed"]),
        "skipped_tasks": len(results["skipped"]),
        "time_used": scheduler.get_time_elapsed()
    }

def _add_quick_pipeline(self, scheduler, data_size_mb):
    """Add minimal viable pipeline for time crunch"""
    scheduler.add_task(
        "load", "Load Data", self.load_data,
        TaskPriority.CRITICAL, TaskComplexity.QUICK_WIN,
        value_score=1.0,
        task_type="load_data", data_size_mb=data_size_mb
    )

    scheduler.add_task(
        "preprocess", "Basic Preprocessing", self.basic_preprocess,
        TaskPriority.CRITICAL, TaskComplexity.QUICK_WIN,
        value_score=0.9, depends_on=["load"],
        task_type="clean_data", data_size_mb=data_size_mb
    )

    scheduler.add_task(
        "model", "Simple Model", self.train_simple_model,
        TaskPriority.HIGH, TaskComplexity.QUICK_WIN,
        value_score=0.7, depends_on=["preprocess"],
        task_type="train_simple_model"
    )

    scheduler.add_task(
        "submit", "Create Submission", self.create_submission,
        TaskPriority.CRITICAL, TaskComplexity.QUICK_WIN,
        value_score=1.0, depends_on=["model"],
        task_type="prepare_submission"
    )
```

---

## Best Practices

### 1. Set Realistic Priorities

```python
# ✅ Good: Critical is truly critical
CRITICAL: Must complete for valid submission
HIGH: Significantly improves results
MEDIUM: Marginal improvement
LOW: Experimental/optional

# ❌ Bad: Everything is critical
CRITICAL: Load data, train model, visualize, debug, optimize, ...
```

### 2. Provide Accurate Duration Estimates

```python
# ✅ Good: Use actual measurements or estimation tool
duration_typical = measured_runtime  # From profiling
# OR
task_type="train_complex_model", data_size_mb=actual_size

# ❌ Bad: Wild guesses
duration_typical = 100  # "Probably around 100 seconds?"
```

### 3. Use Value Scores Wisely

```python
# ✅ Good: Relative value across tasks
value_score=1.0   # Load data (essential)
value_score=0.9   # Train good model (very important)
value_score=0.7   # Train baseline (useful but not critical)
value_score=0.3   # Create visualization (nice to have)

# ❌ Bad: Everything is max value
value_score=1.0 for all tasks  # No differentiation
```

### 4. Handle Task Failures Gracefully

```python
async def resilient_task():
    try:
        return await risky_operation()
    except Exception as e:
        # Return partial result or fallback
        return {"status": "partial", "error": str(e)}

scheduler.add_task(
    "risky", "Risky Operation", resilient_task,
    priority=TaskPriority.MEDIUM,  # NOT critical
    # ... rest of config
)
```

### 5. Monitor and Log

```python
# Before execution
logger.info(f"Starting scheduler with {len(scheduler.tasks)} tasks")
logger.info(f"Time budget: {scheduler.time_budget}s")

# After execution
logger.info(scheduler.get_summary())
logger.info(f"Efficiency: {len(results['completed']) / scheduler.get_time_elapsed():.2f} tasks/sec")
```

---

## API Reference

### TaskScheduler

```python
class TaskScheduler:
    def __init__(
        self,
        time_budget_seconds: float,
        estimate_tool: Optional[EstimateTaskDurationTool] = None
    )
```

**Methods:**

- `add_task(task_id, name, execute_fn, priority, complexity, value_score, ...)`: Add task
- `execute_all(adaptive=True, safety_margin=1.2) -> Dict`: Execute all tasks
- `get_time_remaining() -> float`: Get remaining time in budget
- `get_time_elapsed() -> float`: Get elapsed time
- `get_summary() -> str`: Get human-readable summary

**Properties:**

- `tasks: Dict[str, Task]`: All tasks
- `time_budget: float`: Total time budget
- `started_at: float`: Start timestamp
- `completed_at: float`: End timestamp

### Task

```python
@dataclass
class Task:
    id: str
    name: str
    execute_fn: Callable
    duration_min: float
    duration_typical: float
    duration_max: float
    priority: TaskPriority
    complexity: TaskComplexity
    value_score: float
    depends_on: List[str]
```

**Methods:**

- `get_priority_score(time_remaining: float) -> float`: Calculate priority score

**Properties:**

- `efficiency_score: float`: Value per second (value_score / duration_typical)
- `status: str`: "pending", "running", "completed", "skipped", "failed"
- `result: Any`: Task result (if completed)
- `error: str`: Error message (if failed)

---

## Testing

Run the test suite:

```bash
pytest agent_v5/tests/test_task_scheduler.py -v
```

**Test coverage:**
- ✅ Task priority scoring
- ✅ Dependency resolution
- ✅ Time budget enforcement
- ✅ Adaptive reprioritization
- ✅ Task failure handling
- ✅ Critical task override

---

## Summary

The TaskScheduler enables **intelligent time management** for autonomous agents by:

1. **Estimating task durations** using historical data or the EstimateTaskDuration tool
2. **Prioritizing dynamically** based on time remaining and task value
3. **Adapting mid-execution** when tasks take longer than expected
4. **Guaranteeing critical tasks** run even under time pressure
5. **Skipping low-value tasks** when time runs short

This is especially valuable for:
- **Kaggle competitions** with epoch time limits
- **Production systems** with SLA requirements
- **Resource-constrained environments** (e.g., edge devices)
- **Cost-sensitive applications** (e.g., cloud compute budgets)

**Use it when:** Your agent needs to make strategic decisions about which tasks to run given limited time.
