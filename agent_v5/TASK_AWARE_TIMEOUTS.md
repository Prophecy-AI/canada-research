# Task-Aware Timeouts with EstimateTaskDuration

Agent V5 now automatically integrates `EstimateTaskDuration` tool with the timeout system for intelligent, per-task timeout management.

## Overview

**Problem:** How do you know if a training job has stalled or if it's just taking a normal amount of time?

**Solution:** Agent estimates task duration BEFORE starting, then uses that estimate to detect when something is taking too long.

## How It Works

### 1. Agent Estimates Task Duration

```python
# Agent calls EstimateTaskDuration before starting training
Tool: EstimateTaskDuration
Input: {
  "task_type": "train_complex_model",
  "data_size_mb": 2300,
  "complexity": "moderate"
}

Output:
⏱️ Task Duration Estimate: train_complex_model

📊 Estimated Duration:
   ⚡ Best case:  12m 0s
   📈 Typical:    30m 0s
   ⚠️ Worst case: 1h 30m
```

### 2. TimeoutManager Automatically Registers Task

```python
# Agent V5 automatically parses the estimate and starts tracking
timeout_manager.start_task(
    task_name="train_complex_model",
    estimated_duration=1800,  # 30 minutes
    max_duration=5400          # 1h 30m
)

# Logs:
# ⏱️ Started task: train_complex_model (est: 30m 0s, max: 1h 30m)
```

### 3. Agent Starts the Task

```python
# Agent runs training in background
Tool: Bash
Input: {"command": "python train.py", "background": true}

Result: "Started background process: bash_abc123"
```

### 4. TimeoutManager Monitors Progress

Every turn, the timeout check includes task-specific logic:

```python
timeout_check = timeout_manager.check_timeout()

# Checks:
# ✓ Global timeout (2 hours)
# ✓ Turn limit (100 turns)
# ✓ Stall (20 min no activity)
# ✓ Task timeout (training > 1h 30m)  ← NEW!
```

### 5. Automatic Task Completion

When agent checks output:

```python
Tool: ReadBashOutput
Input: {"shell_id": "bash_abc123"}

Result:
=== Status: COMPLETED (exit code: 0) ===
Training complete! Best val_acc: 0.847

# Agent V5 automatically marks task as complete
timeout_manager.complete_task()
```

## Complete Example Workflow

### Turn-by-Turn Breakdown

**Turn 15: Estimate Task**

```
Agent: "Let me estimate how long training will take"

EstimateTaskDuration({
  "task_type": "train_complex_model",
  "data_size_mb": 2300
})

→ TimeoutManager registers task:
   - Task: train_complex_model
   - Estimated: 30 minutes
   - Max: 1h 30m
```

**Turn 16: Start Training**

```
Agent: "Starting training in background"

Bash({"command": "python train.py", "background": true})

→ Process starts: bash_abc123
→ TimeoutManager tracking: 0m 0s / 30m 0s (0% complete)
```

**Turn 17-25: Monitor (Normal Progress)**

```
Turn 17 (2 min elapsed):
  ReadBashOutput({"shell_id": "bash_abc123"})

  Output: "Epoch 1/15: loss=2.145"

  Timeout check:
    ✓ Global: 2m / 120m OK
    ✓ Turns: 17 / 100 OK
    ✓ Stall: 0m / 20m OK (activity registered)
    ✓ Task: 2m / 90m OK (2% of max)

Turn 22 (15 min elapsed):
  ReadBashOutput(...)

  Output: "Epoch 8/15: loss=1.234"

  Timeout check:
    ✓ Task: 15m / 90m OK (50% of estimated, 17% of max)

Turn 25 (30 min elapsed):
  ReadBashOutput(...)

  Output: "Epoch 12/15: loss=0.892"

  Timeout check:
    ⚠️ Task: 30m / 90m (REACHED ESTIMATED DURATION)
    ℹ️  Still OK - within max duration
```

**Turn 28: Task Complete**

```
Turn 28 (42 min elapsed):
  ReadBashOutput(...)

  Output:
  === Status: COMPLETED (exit code: 0) ===
  Training complete! Best val_acc: 0.847

  → TimeoutManager automatically completes task
  → Task tracking stops

  Timeout check:
    ✓ All checks OK (no active task)
```

## Timeout Scenarios with Task Tracking

### Scenario A: Task Exceeds Maximum Duration

```
Turn 50 (95 min elapsed):
  ReadBashOutput(...)

  Output: "Epoch 14/15: loss=0.542" (still RUNNING)

  Timeout check:
    ❌ Task timeout: 'train_complex_model' exceeded 1h 30m

  → Agent receives timeout message:

  "⏱️ TIMEOUT: Task 'train_complex_model' exceeded 1h 30m
   (estimated 30m 0s)

   Please stop this task and proceed with current best model."

Turn 51 (final):
  Agent: "Stopping training and creating submission"

  KillShell({"shell_id": "bash_abc123"})
  Bash({"command": "python predict.py --model checkpoint_epoch12.pth"})
```

### Scenario B: Task Exceeds Estimate (Warning Only)

```
Turn 30 (35 min elapsed):
  ReadBashOutput(...)

  Output: "Epoch 11/15: loss=1.023"

  Task status:
    Elapsed: 35m
    Estimated: 30m (OVERRUN by 5m)
    Max: 90m (still within bounds)

  Warning: ⚠️ Task is taking longer than expected
           Expected: 30m, Actual: 35m (5m overrun)
           Max allowed: 1h 30m

  → No timeout yet, just a warning
  → Agent can choose to continue or adjust
```

### Scenario C: Multiple Sequential Tasks

```
Turn 10: Estimate + start EDA
  EstimateTaskDuration("explore_data") → 5m typical, 15m max
  Bash("python eda.py", background=true)

  TimeoutManager tracking: EDA (0m / 5m / 15m)

Turn 15: EDA completes
  ReadBashOutput() → "Status: COMPLETED"
  TimeoutManager: EDA task marked complete

Turn 16: Estimate + start training
  EstimateTaskDuration("train_complex_model") → 30m typical, 90m max
  Bash("python train.py", background=true)

  TimeoutManager tracking: Training (0m / 30m / 90m)
  Previous task: EDA (completed in 7m)

Turn 45: Training completes
  ReadBashOutput() → "Status: COMPLETED"
  TimeoutManager: Training task marked complete

  Summary:
    - Tasks completed: 2 (EDA, Training)
    - Total runtime: 52 minutes
    - All tasks within estimated durations
```

## Configuration

### Enable Task-Aware Timeouts

**Already enabled by default!** Just ensure EstimateTaskDuration tool is registered:

```python
# In KaggleAgent (already done if ENABLE_ESTIMATE_DURATION=1)
if os.getenv("ENABLE_ESTIMATE_DURATION", "0") == "1":
    self.tools.register(EstimateTaskDurationTool(workspace_dir))
```

**For mle-bench runs:**

```bash
export ENABLE_ESTIMATE_DURATION=1  # Enable task estimation
./RUN_AGENT_V5_KAGGLE.sh
```

### Adjust Task Timeout Multiplier

The timeout is set to 1.5x the max duration estimate:

```python
# In timeout_manager.py (line 66)
if task_elapsed > self.current_task.max_duration * 1.5:
    # Task timeout triggered
```

To adjust:

```python
# More lenient (2x max)
if task_elapsed > self.current_task.max_duration * 2.0:

# Stricter (1.2x max)
if task_elapsed > self.current_task.max_duration * 1.2:
```

## Benefits

### 1. **Early Stall Detection**

Without task estimates:
```
Training starts → 20 min no output → STALL timeout
(But training might legitimately take 15-20 min per epoch!)
```

With task estimates:
```
Training starts → Estimated 30m →
  - 15 min: "Still within estimate, no warning"
  - 30 min: "Reached estimate, but within max (90m)"
  - 95 min: "Exceeded max duration → TIMEOUT"
```

### 2. **Intelligent Resource Allocation**

Agent knows if it has time for another experiment:

```python
# Current time: 1h 45m into 2h budget
# Remaining: 15 minutes

Agent: "Should I try feature engineering?"

EstimateTaskDuration("feature_engineering") → 25m typical

Agent: "Not enough time. I'll create submission with current model."
```

### 3. **Progress Visibility**

```
Logs show:
[12:34:56] ⏱️ Started task: train_complex_model (est: 30m 0s, max: 1h 30m)
[12:45:23] → API call (turn 25)
[12:45:23] Task status: 10m 27s / 30m 0s (35% complete)
[13:05:42] ✓ Task completed: train_complex_model (actual: 30m 46s)
```

### 4. **Adaptive Workflow**

Agent can adjust strategy based on estimates:

```
Turn 10: EstimateTaskDuration("train_deep_learning") → 2h typical

Agent: "Deep learning would exceed my 2h budget.
        I'll use XGBoost instead (30m estimate)."

Turn 11: EstimateTaskDuration("train_simple_model") → 30m typical

Agent: "This fits my budget. Starting XGBoost training."
```

## API Reference

### TimeoutManager.parse_estimate_result()

```python
def parse_estimate_result(self, estimate_result: str) -> Optional[Dict]:
    """
    Parse EstimateTaskDuration output.

    Args:
        estimate_result: Text output from tool

    Returns:
        {
            "task_name": str,
            "typical": float,  # seconds
            "max": float       # seconds
        } or None if parse failed
    """
```

### TimeoutManager.start_task()

```python
def start_task(self, task_name: str, estimated_duration: float, max_duration: float):
    """
    Start tracking a task.

    Args:
        task_name: Task identifier
        estimated_duration: Typical duration (seconds)
        max_duration: Worst-case duration (seconds)
    """
```

### TimeoutManager.complete_task()

```python
def complete_task(self):
    """
    Mark current task as completed.
    Automatically called when ReadBashOutput shows COMPLETED.
    """
```

### TimeoutManager.get_task_status()

```python
def get_task_status(self) -> Optional[Dict]:
    """
    Get current task status.

    Returns:
        {
            "task_name": str,
            "elapsed_seconds": float,
            "estimated_duration": float,
            "max_duration": float,
            "progress_percent": float,  # based on estimate
            "is_overdue": bool,          # exceeded estimate
            "formatted_elapsed": str,
            "formatted_estimated": str
        } or None if no active task
    """
```

## Troubleshooting

### Task timeout triggers too early

**Cause:** Estimate was too optimistic

**Solution 1:** Provide more context to EstimateTaskDuration
```python
EstimateTaskDuration({
    "task_type": "train_complex_model",
    "data_size_mb": 10240,  # Specify actual data size
    "complexity": "complex",  # Not "moderate"
    "additional_context": "Large dataset, slow GPU"
})
```

**Solution 2:** Increase timeout multiplier (see Configuration above)

### Task never completes automatically

**Cause:** ReadBashOutput not detecting "COMPLETED" status

**Solution:** Ensure your scripts exit properly
```python
# train.py
if __name__ == "__main__":
    train()
    print("Training complete!")  # This triggers COMPLETED status
    sys.exit(0)  # Proper exit
```

### Multiple tasks estimated but only last one tracked

**Expected behavior:** TimeoutManager tracks ONE task at a time

**Explanation:**
```python
EstimateTaskDuration("explore_data")  # Task 1 tracked
EstimateTaskDuration("train_model")   # Task 1 auto-completed, Task 2 tracked

# Only the most recent estimate is active
```

## Summary

**Key Features:**
- ✅ Automatic task registration from EstimateTaskDuration
- ✅ Per-task timeout monitoring (1.5x max duration)
- ✅ Automatic task completion detection
- ✅ Progress tracking and warnings
- ✅ No manual integration needed

**Workflow:**
1. Agent calls `EstimateTaskDuration("train_model")`
2. TimeoutManager parses output and starts tracking
3. Agent starts task (`Bash(background=true)`)
4. TimeoutManager monitors: elapsed vs estimate vs max
5. Agent checks progress (`ReadBashOutput`)
6. TimeoutManager auto-completes when "COMPLETED" detected

**Result:** Intelligent timeouts that adapt to the actual task being performed!

---

**Last updated:** 2025-10-17
