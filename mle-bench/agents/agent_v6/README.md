# Agent V6: Parallel-First Kaggle Agent

Built for parallel experiment execution on Kaggle competitions using mle-bench.

## Quick Start

```bash
cd /path/to/mle-bench

python run_agent.py \
  --agent-id agent_v6 \
  --competition-set experiments/splits/spaceship-titanic.txt
```

## Architecture

```
┌─────────────────────────────────────────┐
│   Orchestrator (manages rounds)         │
│                                          │
│   EDA → Planning → Parallel Execution → │
│         Analysis → Submit                │
└─────────────────────────────────────────┘
              ↓
    ┌─────────────────────────┐
    │   3 Workers in parallel │
    │   (via asyncio.gather)  │
    └─────────────────────────┘
```

## Key Differences from agent_v5_kaggle

| Feature | agent_v5_kaggle | agent_v6 |
|---------|----------------|----------|
| **Execution** | Sequential (background processes) | Parallel (asyncio) |
| **Tools** | 10 tools (Bash with monitoring) | 3 tools (Bash, Read, Write) |
| **Complexity** | ~2000 lines | ~750 lines |
| **Timeouts** | Per-tool (120s, 600s) | None (runs as long as needed) |
| **Monitoring** | ReadBashOutput, KillShell | No monitoring needed |

## File Structure

```
agent_v6/
├── agent_v6 → ../../../agent_v6  # Symlink to root
├── config.yaml                    # mle-bench config
├── Dockerfile                     # Docker image
├── requirements.txt               # Dependencies
├── start.sh                       # Entry point
└── README.md                      # This file
```

## Source Code Location

Agent V6 source is in `/canada-research/agent_v6/` (root of repo).

This directory uses a symlink to avoid duplication, just like agent_v5_kaggle does.

## Expected Behavior

**On spaceship-titanic:**
1. EDA: Analyzes train.csv (21 samples)
2. Planning: Generates 3 experiment specs
3. Execution: Runs 3 models in parallel (RandomForest, XGBoost, LogisticRegression)
4. Analysis: Compares results, decides to submit or continue
5. Submission: Creates submission.csv from best model

**Time:** Depends on training - no timeouts

**Success:** submission.csv created at /home/submission/submission.csv

