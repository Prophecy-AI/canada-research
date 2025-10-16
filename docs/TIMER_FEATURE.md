# Global Timer Feature for Kaggle Agent

## Overview

Added a global timer to track the total elapsed time for Kaggle competition runs, allowing the agent to be aware of time constraints and manage its time budget effectively.

## Components Added

### 1. TimerTool ([agent_v5/tools/timer.py](agent_v5/tools/timer.py))

A new tool that allows the agent to check elapsed time during a run.

**Features:**
- Shows elapsed time in human-readable format (hours, minutes, seconds)
- Displays total seconds for precise tracking
- Shows the start timestamp for reference
- Zero-input tool (no parameters required)

**Example output:**
```
⏱️ Elapsed time: 0h 5m 23s
   Total seconds: 323.4s
   Started at: 2025-10-15 14:32:10
```

### 2. Runner Integration ([mle-bench/agents/agent_v5_kaggle/runner.py](mle-bench/agents/agent_v5_kaggle/runner.py))

**Changes:**
- Start timer at the beginning of the Kaggle run
- Log start time in human-readable format
- Calculate and log total runtime at completion
- Override agent's internal start_time to ensure consistency

**Output example:**
```
🏆 Starting Kaggle Agent for competition: spaceship-titanic
📊 Data: /home/data
💻 Workspace: /home/code
📤 Submission: /home/submission
⏱️  Start time: 2025-10-15 14:32:10

... [agent run] ...

✓ Agent run complete
⏱️  Total runtime: 1h 23m 45s (5025.3s)
```

### 3. KaggleAgent Integration ([mle-bench/agents/agent_v5_kaggle/kaggle_agent.py](mle-bench/agents/agent_v5_kaggle/kaggle_agent.py))

**Changes:**
- Initialize `start_time` in `__init__`
- Register TimerTool with agent's tool registry
- Add Timer to the system prompt tool list

### 4. System Prompt Update

The agent now knows about the Timer tool:

```
**Your Tools:**
- Read: Read files (CSVs, instructions, etc.)
- Write: Create Python scripts (ALWAYS separate train.py from predict.py)
- Edit: Modify existing files
- Glob: Find files by pattern (e.g., "*.csv")
- Grep: Search file contents
- Timer: Check elapsed time since competition started (helps manage time budget)
- Bash: Execute shell commands (background parameter REQUIRED)
...
```

## Testing

Added comprehensive tests:

### Unit Tests ([agent_v5/tests/test_timer_tool.py](agent_v5/tests/test_timer_tool.py))
- `test_timer_tool_basic` - Basic functionality
- `test_timer_tool_elapsed_time` - Accurate time tracking
- `test_timer_tool_format` - Human-readable formatting
- `test_timer_tool_schema` - Correct schema for LLM

### Integration Tests ([agent_v5/tests/test_kaggle_agent_timer.py](agent_v5/tests/test_kaggle_agent_timer.py))
- `test_kaggle_agent_has_timer_tool` - Timer registered in agent
- `test_kaggle_agent_timer_tracks_elapsed_time` - End-to-end functionality

**All tests passing:** ✅ 6/6 tests pass

## Usage

### For the Agent

The agent can now call the Timer tool at any point:

```json
{
  "name": "Timer",
  "input": {}
}
```

This helps the agent:
- Track progress toward time limits
- Prioritize remaining work
- Decide when to wrap up and create submission
- Avoid timeout failures

### For Developers

The runner automatically logs timing:

```bash
# Start of run
⏱️  Start time: 2025-10-15 14:32:10

# End of run
⏱️  Total runtime: 1h 23m 45s (5025.3s)
```

## Benefits

1. **Time Awareness**: Agent knows how much time has passed
2. **Better Planning**: Agent can prioritize tasks based on remaining time
3. **Debugging**: Developers can see total runtime for each competition
4. **Optimization**: Can identify slow runs and optimize accordingly
5. **Compliance**: Helps agent stay within time budget constraints

## Implementation Details

### Design Pattern

The timer uses a **callback pattern** to access the start time:

```python
TimerTool(
    workspace_dir=workspace_dir,
    get_start_time=lambda: self.start_time  # Callback
)
```

This allows the runner to override `agent.start_time` after initialization, ensuring the timer uses the correct start time.

### Time Synchronization

1. Runner creates start_time: `start_time = time.time()`
2. Agent initializes with its own start_time
3. Runner overrides: `agent.start_time = start_time`
4. Timer tool uses callback: `get_start_time()` → returns runner's start_time

This ensures consistent timing across the entire workflow.

## Future Enhancements

Potential improvements:
- [ ] Add time budget warnings (e.g., "80% of time used")
- [ ] Track time spent on different phases (EDA, training, inference)
- [ ] Add time-per-iteration metrics for training
- [ ] Log timer calls to observability system
- [ ] Add deadline parameter (e.g., "4 hours total")

## Files Modified

1. **New:** `agent_v5/tools/timer.py` - Timer tool implementation
2. **New:** `agent_v5/tests/test_timer_tool.py` - Unit tests
3. **New:** `agent_v5/tests/test_kaggle_agent_timer.py` - Integration tests
4. **Modified:** `mle-bench/agents/agent_v5_kaggle/runner.py` - Timer integration
5. **Modified:** `mle-bench/agents/agent_v5_kaggle/kaggle_agent.py` - Tool registration
6. **Fixed:** `agent_v5/tools/bash.py` - Removed duplicate `cwd` parameter

## Related Issues

- Fixes syntax error in bash.py (duplicate `cwd` parameter)
- Enables time-aware agent behavior for Kaggle competitions
