# Environment Variables

This document lists the environment variables used to configure the agent.

## Feature Toggles

### ENABLE_ESTIMATE_DURATION

**Default:** `0` (disabled)

**Values:**
- `0` - EstimateTaskDuration tool is **not** registered (default)
- `1` - EstimateTaskDuration tool is registered and available

**Description:**
Controls whether the `EstimateTaskDuration` tool is available to the agent. This tool provides heuristic-based estimates for task durations (e.g., "training a model should take 10-30 minutes").

**Usage:**
```bash
# Enable the tool
export ENABLE_ESTIMATE_DURATION=1

# Disable the tool (default)
export ENABLE_ESTIMATE_DURATION=0
# or simply don't set it
```

**When enabled:**
- The tool appears in the agent's system prompt
- The agent can call `EstimateTaskDuration` to get duration estimates
- Useful for task planning and detecting stalled processes

**When disabled (default):**
- The tool is not registered in the tool registry
- The tool does not appear in the system prompt
- Reduces token usage and simplifies the agent's tool set

## Other Environment Variables

### DEBUG

**Default:** `0` (disabled)

**Values:**
- `0` - Debug logging disabled
- `1` - Debug logging enabled

**Description:**
Enables detailed debug logging for development and troubleshooting.

### ANTHROPIC_API_KEY

**Required:** Yes

**Description:**
API key for Anthropic Claude models. Required for the agent to function.

### PLANNING_MODEL

**Default:** `claude-opus-4-20250514`

**Description:**
The model to use for the PlanTask tool (reasoning-optimized planning).
