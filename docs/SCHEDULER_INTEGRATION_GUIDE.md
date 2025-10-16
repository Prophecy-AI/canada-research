# TaskScheduler Integration Guide

> **How to integrate the TaskScheduler into your existing agent for time-aware task management**

---

## Quick Start

### 1. Import the scheduler

```python
from agent_v5.task_scheduler import TaskScheduler, TaskPriority, TaskComplexity
from agent_v5.tools.estimate_duration import EstimateTaskDurationTool
```

### 2. Create scheduler instance

```python
# In your agent's __init__ or run method
self.scheduler = TaskScheduler(
    time_budget_seconds=self.epoch_time_limit,
    estimate_tool=EstimateTaskDurationTool(workspace_dir=self.workspace_dir)
)
```

### 3. Add tasks

```python
# Add each task with duration estimates
self.scheduler.add_task(
    task_id="load_data",
    name="Load training data",
    execute_fn=self.load_data,
    priority=TaskPriority.CRITICAL,
    complexity=TaskComplexity.QUICK_WIN,
    value_score=1.0,
    task_type="load_data",
    data_size_mb=self.dataset_size_mb
)
```

### 4. Execute

```python
# Run all tasks with adaptive scheduling
results = await self.scheduler.execute_all(adaptive=True)
```

---

## Pattern 1: Agent with Fixed Pipeline

**Use case:** Agent always runs the same sequence of tasks, but wants to skip low-priority tasks when time is tight.

```python
class KaggleAgent:
    def __init__(self, workspace_dir: str, time_budget: int):
        self.workspace_dir = workspace_dir
        self.time_budget = time_budget
        self.estimate_tool = EstimateTaskDurationTool(workspace_dir)

    async def run_epoch(self):
        """Run one epoch with time-aware scheduling"""

        # Create scheduler
        scheduler = TaskScheduler(
            time_budget_seconds=self.time_budget,
            estimate_tool=self.estimate_tool
        )

        # Get dataset info
        train_size = os.path.getsize(f"{self.workspace_dir}/train.csv")
        train_size_mb = train_size / (1024 * 1024)

        # Add pipeline tasks
        self._add_pipeline_tasks(scheduler, train_size_mb)

        # Execute
        results = await scheduler.execute_all(adaptive=True)

        # Handle results
        if results["failed"]:
            raise Exception(f"Tasks failed: {results['failed']}")

        return results

    def _add_pipeline_tasks(self, scheduler, data_size_mb):
        """Add all pipeline tasks to scheduler"""

        # 1. Load data (CRITICAL)
        scheduler.add_task(
            task_id="load",
            name="Load Data",
            execute_fn=self._load_data,
            priority=TaskPriority.CRITICAL,
            complexity=TaskComplexity.QUICK_WIN,
            value_score=1.0,
            task_type="load_data",
            data_size_mb=data_size_mb
        )

        # 2. EDA (LOW - skip if time is tight)
        scheduler.add_task(
            task_id="eda",
            name="Exploratory Data Analysis",
            execute_fn=self._run_eda,
            priority=TaskPriority.LOW,
            complexity=TaskComplexity.EXPLORATORY,
            value_score=0.3,
            depends_on=["load"],
            task_type="explore_data",
            data_size_mb=data_size_mb
        )

        # 3. Preprocessing (CRITICAL)
        scheduler.add_task(
            task_id="preprocess",
            name="Data Preprocessing",
            execute_fn=self._preprocess,
            priority=TaskPriority.CRITICAL,
            complexity=TaskComplexity.EFFICIENT,
            value_score=0.9,
            depends_on=["load"],
            task_type="clean_data",
            data_size_mb=data_size_mb
        )

        # 4. Feature engineering (HIGH)
        scheduler.add_task(
            task_id="features",
            name="Feature Engineering",
            execute_fn=self._engineer_features,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.EFFICIENT,
            value_score=0.85,
            depends_on=["preprocess"],
            task_type="feature_engineering",
            data_size_mb=data_size_mb
        )

        # 5. Train baseline model (HIGH)
        scheduler.add_task(
            task_id="baseline",
            name="Train Baseline Model",
            execute_fn=self._train_baseline,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.QUICK_WIN,
            value_score=0.7,
            depends_on=["features"],
            task_type="train_simple_model"
        )

        # 6. Train advanced model (MEDIUM)
        scheduler.add_task(
            task_id="advanced",
            name="Train Advanced Model",
            execute_fn=self._train_advanced,
            priority=TaskPriority.MEDIUM,
            complexity=TaskComplexity.EXPENSIVE,
            value_score=0.9,
            depends_on=["features"],
            task_type="train_complex_model"
        )

        # 7. Hyperparameter tuning (LOW - only if time permits)
        scheduler.add_task(
            task_id="tuning",
            name="Hyperparameter Tuning",
            execute_fn=self._tune_model,
            priority=TaskPriority.LOW,
            complexity=TaskComplexity.EXPENSIVE,
            value_score=0.85,
            depends_on=["advanced"],
            task_type="hyperparameter_tuning"
        )

        # 8. Ensemble (HIGH)
        scheduler.add_task(
            task_id="ensemble",
            name="Ensemble Models",
            execute_fn=self._ensemble,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.QUICK_WIN,
            value_score=0.95,
            depends_on=["baseline", "advanced"],
            task_type="ensemble_models"
        )

        # 9. Create submission (CRITICAL)
        scheduler.add_task(
            task_id="submit",
            name="Create Submission",
            execute_fn=self._create_submission,
            priority=TaskPriority.CRITICAL,
            complexity=TaskComplexity.QUICK_WIN,
            value_score=1.0,
            depends_on=["ensemble"],
            task_type="prepare_submission"
        )

    async def _load_data(self):
        """Load training data"""
        # Your implementation
        pass

    async def _run_eda(self):
        """Run exploratory data analysis"""
        # Your implementation
        pass

    # ... other task implementations
```

---

## Pattern 2: Agent with Dynamic Pipeline

**Use case:** Agent plans tasks on-the-fly based on data characteristics and available time.

```python
class DynamicKaggleAgent:
    async def run_epoch(self):
        """Dynamically plan tasks based on data and time budget"""

        scheduler = TaskScheduler(
            time_budget_seconds=self.time_budget,
            estimate_tool=self.estimate_tool
        )

        # Analyze dataset
        dataset_info = self._analyze_dataset()

        # Build pipeline based on data characteristics
        if dataset_info["type"] == "tabular":
            self._add_tabular_pipeline(scheduler, dataset_info)
        elif dataset_info["type"] == "image":
            self._add_image_pipeline(scheduler, dataset_info)
        elif dataset_info["type"] == "text":
            self._add_text_pipeline(scheduler, dataset_info)

        # Adjust priorities based on time budget
        if self.time_budget < 1800:  # < 30 min
            # Time crunch: downgrade expensive tasks
            for task_id, task in scheduler.tasks.items():
                if task.complexity == TaskComplexity.EXPENSIVE:
                    task.priority = TaskPriority.LOW
                elif task.complexity == TaskComplexity.QUICK_WIN:
                    task.priority = TaskPriority.HIGH

        # Execute
        results = await scheduler.execute_all(adaptive=True)

        return results

    def _add_tabular_pipeline(self, scheduler, info):
        """Add tasks for tabular data competition"""
        data_size_mb = info["size_mb"]

        # Core tasks
        scheduler.add_task("load", ..., task_type="load_data", data_size_mb=data_size_mb)
        scheduler.add_task("clean", ..., task_type="clean_data", data_size_mb=data_size_mb)

        # Model selection based on data size
        if data_size_mb < 100:
            # Small dataset: try many models
            scheduler.add_task("lr", ..., task_type="train_simple_model")
            scheduler.add_task("rf", ..., task_type="train_complex_model")
            scheduler.add_task("lgbm", ..., task_type="train_complex_model")
            scheduler.add_task("xgb", ..., task_type="train_complex_model")
        else:
            # Large dataset: focus on scalable models
            scheduler.add_task("sgd", ..., task_type="train_simple_model")
            scheduler.add_task("lgbm", ..., task_type="train_complex_model")

        # Submission
        scheduler.add_task("submit", ..., task_type="prepare_submission")
```

---

## Pattern 3: Incremental Improvement

**Use case:** Agent runs multiple rounds, building on previous results.

```python
class IncrementalAgent:
    async def run_epoch(self):
        """Run multiple improvement rounds within time budget"""

        scheduler = TaskScheduler(
            time_budget_seconds=self.time_budget,
            estimate_tool=self.estimate_tool
        )

        # Check what was done in previous epochs
        previous_results = self._load_previous_results()

        # Round 1: Foundation (if not done)
        if not previous_results.get("foundation_complete"):
            self._add_foundation_tasks(scheduler)

        # Round 2: Baseline models (if not done)
        if not previous_results.get("baseline_complete"):
            self._add_baseline_tasks(scheduler)

        # Round 3: Improvements (always try)
        self._add_improvement_tasks(scheduler, previous_results)

        # Execute
        results = await scheduler.execute_all(adaptive=True)

        # Save progress
        self._save_results(results)

        return results

    def _add_improvement_tasks(self, scheduler, previous_results):
        """Add incremental improvement tasks"""

        # Feature engineering v2
        if previous_results.get("baseline_score", 0) < 0.8:
            scheduler.add_task(
                "feature_v2",
                "Advanced Feature Engineering",
                self._engineer_features_v2,
                priority=TaskPriority.HIGH,
                complexity=TaskComplexity.EFFICIENT,
                value_score=0.8,
                task_type="feature_engineering"
            )

        # Model stacking
        if len(previous_results.get("models", [])) >= 2:
            scheduler.add_task(
                "stack",
                "Stack Models",
                self._stack_models,
                priority=TaskPriority.MEDIUM,
                complexity=TaskComplexity.EFFICIENT,
                value_score=0.85,
                task_type="ensemble_models"
            )

        # Hyperparameter tuning (only if time permits)
        best_model = previous_results.get("best_model")
        if best_model:
            scheduler.add_task(
                "tune",
                f"Tune {best_model}",
                lambda: self._tune_model(best_model),
                priority=TaskPriority.LOW,
                complexity=TaskComplexity.EXPENSIVE,
                value_score=0.9,
                task_type="hyperparameter_tuning"
            )
```

---

## Pattern 4: Monitoring and Intervention

**Use case:** Agent monitors task progress and can intervene if needed.

```python
class MonitoredAgent:
    async def run_epoch(self):
        """Run with real-time monitoring"""

        scheduler = TaskScheduler(
            time_budget_seconds=self.time_budget,
            estimate_tool=self.estimate_tool
        )

        # Add tasks
        self._add_pipeline_tasks(scheduler)

        # Start execution in background
        execution_task = asyncio.create_task(
            scheduler.execute_all(adaptive=True)
        )

        # Monitor progress
        while not execution_task.done():
            await asyncio.sleep(10)  # Check every 10 seconds

            # Get current state
            time_remaining = scheduler.get_time_remaining()
            time_elapsed = scheduler.get_time_elapsed()

            # Log progress
            print(f"⏱️  {time_elapsed:.0f}s elapsed, {time_remaining:.0f}s remaining")

            # Check for stalls
            if self._is_stalled(scheduler):
                print("⚠️  Detected stall, intervening...")
                # Could cancel current task, adjust priorities, etc.

            # Adjust strategy if running low on time
            if time_remaining < 300:  # < 5 min
                print("⚠️  Low on time, prioritizing submission...")
                self._emergency_mode(scheduler)

        # Get results
        results = await execution_task
        return results

    def _is_stalled(self, scheduler) -> bool:
        """Detect if current task is stalled"""
        for task in scheduler.tasks.values():
            if task.status == "running":
                runtime = time.time() - task.started_at
                expected_max = task.duration_max * 1.5

                if runtime > expected_max:
                    print(f"⚠️  Task {task.name} exceeded max duration")
                    return True

        return False

    def _emergency_mode(self, scheduler):
        """Switch to emergency mode - ensure submission happens"""
        # Downgrade all non-critical tasks
        for task_id, task in scheduler.tasks.items():
            if task.priority != TaskPriority.CRITICAL:
                task.priority = TaskPriority.LOW

            # Boost submission task
            if "submit" in task_id.lower():
                task.priority = TaskPriority.CRITICAL
```

---

## Integration with Existing Tools

### Using with BackgroundProcess

Combine scheduler with background process execution:

```python
from agent_v5.background_process import BackgroundProcess

class AgentWithBackground:
    async def _train_model_background(self):
        """Train model in background process"""

        # Start background process
        process = BackgroundProcess(
            command="python train.py",
            working_dir=self.workspace_dir
        )

        # Monitor until complete
        while process.is_running():
            await asyncio.sleep(5)

            # Check for timeout (handled by scheduler)
            if self.scheduler.get_time_remaining() < 60:
                print("⚠️  Timeout approaching, stopping training...")
                process.stop()
                break

        return process.get_output()

    def _add_tasks(self, scheduler):
        # Add task that uses background process
        scheduler.add_task(
            "train_model",
            "Train Model (Background)",
            self._train_model_background,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.EXPENSIVE,
            value_score=0.9,
            task_type="train_complex_model"
        )
```

### Using with EstimateTaskDuration

Get duration estimates on-demand:

```python
async def should_run_task(self, task_type: str, data_size_mb: float) -> bool:
    """Decide if task should run based on time remaining"""

    # Get estimate
    result = await self.estimate_tool.execute({
        "task_type": task_type,
        "data_size_mb": data_size_mb
    })

    # Parse typical duration (hacky - in production, tool should return structured data)
    # For now, check if we have time
    time_remaining = self.scheduler.get_time_remaining()

    # Use heuristic: need 2x typical duration for safety
    estimated_duration = self._parse_typical_duration(result["content"])

    return time_remaining > (estimated_duration * 2)
```

---

## Best Practices

### 1. Always Set CRITICAL Priority for Must-Have Tasks

```python
# ✅ Good
scheduler.add_task("load_data", ..., priority=TaskPriority.CRITICAL)
scheduler.add_task("submit", ..., priority=TaskPriority.CRITICAL)

# ❌ Bad - agent might skip these!
scheduler.add_task("load_data", ..., priority=TaskPriority.HIGH)
scheduler.add_task("submit", ..., priority=TaskPriority.HIGH)
```

### 2. Use Dependencies to Enforce Order

```python
# ✅ Good - explicit dependencies
scheduler.add_task("load", ...)
scheduler.add_task("process", ..., depends_on=["load"])
scheduler.add_task("train", ..., depends_on=["process"])

# ❌ Bad - implicit ordering through priority
# (scheduler might reorder based on time remaining)
scheduler.add_task("load", ..., priority=TaskPriority.CRITICAL)
scheduler.add_task("process", ..., priority=TaskPriority.HIGH)
scheduler.add_task("train", ..., priority=TaskPriority.MEDIUM)
```

### 3. Provide Accurate Estimates

```python
# ✅ Good - use actual file size
data_size_mb = os.path.getsize("train.csv") / (1024 * 1024)
scheduler.add_task(..., task_type="load_data", data_size_mb=data_size_mb)

# ❌ Bad - hardcoded guess
scheduler.add_task(..., task_type="load_data", data_size_mb=1000)
```

### 4. Handle Task Failures

```python
# ✅ Good - graceful error handling
async def resilient_task():
    try:
        return await risky_operation()
    except Exception as e:
        logger.error(f"Task failed: {e}")
        return {"status": "failed", "error": str(e)}

scheduler.add_task("task", ..., execute_fn=resilient_task)

# ❌ Bad - unhandled exceptions
async def fragile_task():
    return await risky_operation()  # Will crash if it fails
```

### 5. Log Progress

```python
# ✅ Good - comprehensive logging
logger.info("Starting epoch")
logger.info(f"Time budget: {scheduler.time_budget}s")
logger.info(f"Tasks planned: {len(scheduler.tasks)}")

results = await scheduler.execute_all(adaptive=True)

logger.info(scheduler.get_summary())
logger.info(f"Completed: {len(results['completed'])}")
logger.info(f"Skipped: {len(results['skipped'])}")
```

---

## Troubleshooting

### Problem: All tasks are being skipped

**Cause:** Estimates are too conservative or time budget is too small

**Solution:**
```python
# Reduce safety margin
results = await scheduler.execute_all(safety_margin=1.0)

# Or check your estimates
for task_id, task in scheduler.tasks.items():
    print(f"{task.name}: {task.duration_typical}s")
```

### Problem: Critical task is skipped

**Cause:** Task has wrong priority level

**Solution:**
```python
# Ensure critical tasks have CRITICAL priority
scheduler.add_task(..., priority=TaskPriority.CRITICAL)  # Will always run
```

### Problem: Tasks run in wrong order

**Cause:** Missing dependencies

**Solution:**
```python
# Add explicit dependencies
scheduler.add_task("task_b", ..., depends_on=["task_a"])
```

### Problem: Scheduler is too slow

**Cause:** Too many tasks or too complex priority calculations

**Solution:**
```python
# Reduce number of tasks
# Or disable adaptive mode
results = await scheduler.execute_all(adaptive=False)
```

---

## Summary

The TaskScheduler integrates into your agent with 4 steps:

1. **Create scheduler** with time budget
2. **Add tasks** with priorities, complexities, and estimates
3. **Execute** with adaptive scheduling
4. **Handle results** (completed, skipped, failed)

Use it to:
- ✅ Skip low-priority tasks when time is tight
- ✅ Guarantee critical tasks complete
- ✅ Adapt dynamically to changing conditions
- ✅ Make intelligent trade-offs between time and quality

See [TASK_SCHEDULER.md](TASK_SCHEDULER.md) for full API documentation.
