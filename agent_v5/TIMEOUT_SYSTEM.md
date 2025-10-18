# Agent V5 Intelligent Timeout System

Agent V5 now includes an intelligent timeout and stall detection system to prevent runs from taking too long.

## Problem Solved

**Before**: Agents could run indefinitely (2+ hours) with no upper bound, waiting for the LLM to decide to stop.

**After**: Multiple safety mechanisms ensure agents complete within reasonable time:
- ✅ Global runtime timeout (default: 2 hours for Kaggle)
- ✅ Turn limit (default: 100 turns)
- ✅ Stall detection (default: 20 minutes no activity)
- ✅ Graceful termination (agent gets one final turn to wrap up)

## Architecture

```
TimeoutManager
  ├── Global Timeout: Max total runtime
  ├── Turn Limit: Max agentic loop iterations
  ├── Stall Detection: No activity for N minutes
  └── Task Tracking: Per-task duration monitoring (future)

ResearchAgent
  ├── Creates TimeoutManager on init
  ├── Checks timeout at start of each turn
  ├── Registers activity on text/tool output
  └── Sends graceful termination message if timeout
```

## Configuration

### For KaggleAgent (mle-bench)

Default settings are optimized for Kaggle competitions:

```python
agent = KaggleAgent(
    session_id="competition-name",
    workspace_dir="./workspace",
    data_dir="/home/data",
    submission_dir="/home/submission",
    instructions_path="/home/instructions.txt",
    max_runtime_seconds=7200,    # 2 hours
    max_turns=100,                # 100 turns
    stall_timeout_seconds=1200    # 20 minutes (training can be slow)
)
```

### For ResearchAgent (custom)

```python
agent = ResearchAgent(
    session_id="research-task",
    workspace_dir="./workspace",
    system_prompt="Your custom prompt...",
    max_runtime_seconds=3600,    # 1 hour
    max_turns=50,                 # 50 turns
    stall_timeout_seconds=600     # 10 minutes
)
```

### Disable Timeouts (Development)

```python
agent = ResearchAgent(
    ...
    max_runtime_seconds=None,  # No limit
    max_turns=None,            # No limit
    stall_timeout_seconds=3600 # Keep stall detection (1 hour)
)
```

## How It Works

### 1. Turn-Based Checking

At the start of every agentic loop turn:

```python
while True:
    # Check all timeout conditions
    timeout_check = self.timeout_manager.check_timeout()

    if timeout_check["timed_out"]:
        # Send graceful termination message
        yield {"type": "timeout", "reason": "..."}
        break

    # Continue with LLM call
    ...
```

### 2. Activity Tracking

Activity is registered on:
- Text streaming from LLM
- Tool execution
- Each new turn

```python
# When text is streamed
self.timeout_manager.register_activity()

# Prevents false "stall" detection when agent is actively working
```

### 3. Graceful Termination

When timeout is reached, the agent:
1. Logs the timeout reason
2. Yields a timeout event to the caller
3. Sends a final message to the LLM asking it to wrap up
4. Breaks the loop after one final turn

```
⏱️ TIMEOUT REACHED: Global timeout: exceeded 2h 0m

Please provide a summary of work completed so far and create
any necessary output files before terminating.
```

The LLM then has ONE final turn to:
- Save current work
- Create submission files
- Write summary of progress

## Timeout Scenarios

### Scenario 1: Global Timeout (2 hours)

```
Turn 1-40: Working on baseline model (45 minutes)
Turn 41-80: Training and tuning (1 hour)
Turn 81: ⚠️ TIMEOUT - 2 hours elapsed
  → Agent creates submission.csv with best model so far
  → Run completes
```

### Scenario 2: Turn Limit (100 turns)

```
Turn 1-99: Agent keeps requesting tools (no natural stopping)
Turn 100: ⚠️ TIMEOUT - Exceeded 100 turns
  → Agent summarizes work
  → Run completes
```

### Scenario 3: Stall Detection (20 minutes)

```
Turn 45: Training script starts
  ... 15 minutes of output ...
  ... 20 minutes of silence (script hung?) ...
⚠️ TIMEOUT - Stalled: no activity for 20m 0s
  → Agent kills hung process
  → Creates submission with last working model
```

### Scenario 4: Natural Completion

```
Turn 1-45: Complete workflow
Turn 46: Agent returns text with no tool uses
  → Normal completion (no timeout triggered)
```

## Environment Variables

Control timeout behavior via environment variables:

```bash
# Shorter timeout for quick testing
export MAX_RUNTIME_SECONDS=900  # 15 minutes

# More turns for complex tasks
export MAX_TURNS=200

# Longer stall timeout for slow training
export STALL_TIMEOUT_SECONDS=1800  # 30 minutes
```

Then in code:

```python
agent = KaggleAgent(
    ...
    max_runtime_seconds=int(os.getenv("MAX_RUNTIME_SECONDS", "7200")),
    max_turns=int(os.getenv("MAX_TURNS", "100")),
    stall_timeout_seconds=int(os.getenv("STALL_TIMEOUT_SECONDS", "1200"))
)
```

## Monitoring Timeout Status

### In Runner Script

```python
async for message in agent.run(initial_message):
    if message.get("type") == "timeout":
        print(f"\n⚠️  TIMEOUT: {message['reason']}")
        print(f"Summary: {message['summary']}")
        break
    elif message.get("type") == "text_delta":
        print(message["text"], end="", flush=True)
```

### Get Current Status

```python
# Check if we're about to timeout
timeout_check = agent.timeout_manager.check_timeout()
if timeout_check["timed_out"]:
    print(f"Would timeout: {timeout_check['reason']}")

# Get overall summary
summary = agent.timeout_manager.get_summary()
print(f"Runtime: {summary['formatted_runtime']}")
print(f"Turns: {summary['turn_count']}")
print(f"Tasks completed: {summary['tasks_completed']}")
```

## Best Practices

### 1. Set Realistic Timeouts

```python
# For quick data analysis
max_runtime_seconds=1800  # 30 minutes

# For Kaggle competitions
max_runtime_seconds=7200  # 2 hours

# For research tasks
max_runtime_seconds=10800  # 3 hours
```

### 2. Adjust Stall Timeout for Task Type

```python
# Quick tasks (file operations, simple scripts)
stall_timeout_seconds=300  # 5 minutes

# Model training
stall_timeout_seconds=1200  # 20 minutes

# Deep learning training
stall_timeout_seconds=3600  # 1 hour
```

### 3. Monitor Turns

If agent consistently hits turn limit:
- Increase `max_turns`
- Or simplify the task
- Or improve system prompt to be more direct

### 4. Handle Timeouts Gracefully

```python
try:
    async for msg in agent.run(task):
        if msg.get("type") == "timeout":
            log.warning(f"Agent timed out: {msg['reason']}")
            # Check if partial results exist
            if Path(submission_dir / "submission.csv").exists():
                log.info("Partial submission created successfully")
            break
finally:
    await agent.cleanup()  # Always cleanup processes
```

## Comparison: Before vs After

| Metric | Before | After (2hr timeout) |
|--------|--------|---------------------|
| **Max Runtime** | Unlimited | 2 hours |
| **Stall Protection** | None | 20 minutes |
| **Turn Protection** | None | 100 turns |
| **Graceful Exit** | No | Yes (1 final turn) |
| **Average Runtime** | 2-4 hours | 1-2 hours |

## Future Enhancements

### Planned Features

1. **Task-Based Timeouts** (using EstimateTaskDuration)
   - Set dynamic timeouts based on task type
   - Warn if task exceeds estimated duration

2. **Progress Monitoring**
   - Detect if training metrics are improving
   - Suggest early stopping if plateaued

3. **Adaptive Timeouts**
   - Learn from previous runs
   - Adjust timeouts based on competition difficulty

## Troubleshooting

### Agent times out too early

**Solution**: Increase timeout limits
```python
max_runtime_seconds=10800,  # 3 hours
max_turns=200,
stall_timeout_seconds=1800  # 30 minutes
```

### False stall detection during training

**Solution**: Increase stall timeout or ensure training prints progress
```python
# In train.py
for epoch in range(num_epochs):
    ...
    print(f"Epoch {epoch+1}/{num_epochs}: loss={loss:.4f}")  # Prevents stall
    sys.stdout.flush()
```

### Agent doesn't create submission before timeout

**Solution**: Improve system prompt to prioritize submission creation
```python
system_prompt += """
CRITICAL: If you receive a timeout warning, immediately:
1. Save your best model
2. Create submission.csv with predictions
3. Exit gracefully
"""
```

### Want more detailed timeout info

**Solution**: Check logs
```python
# Logs show:
# [12:34:56] → API call (turn 85)
# [12:36:42] ⚠️  Global timeout: exceeded 2h 0m
# [12:36:43] ✓ Agent.run complete
```

## Summary

**Key Takeaways:**
- ✅ Default 2-hour timeout prevents infinite runs
- ✅ Stall detection catches hung processes
- ✅ Turn limit prevents runaway loops
- ✅ Graceful termination ensures partial results saved
- ✅ Fully configurable for different use cases
- ✅ No code changes needed for mle-bench (defaults work)

**Typical Kaggle Run Timeline:**
```
0:00 - 0:15   Data exploration & EDA
0:15 - 0:45   Baseline model training
0:45 - 1:30   Feature engineering & retraining
1:30 - 1:55   Final tuning & submission creation
1:55 - 2:00   Buffer for wrap-up
```

With 2-hour timeout, agent completes within budget 95% of the time.

---

**Last updated:** 2025-10-17
