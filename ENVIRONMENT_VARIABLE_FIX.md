# Environment Variable Fix - GEMINI_API_KEY

**Date**: 2025-10-24
**Status**: ✅ RESOLVED

## Problem

The agent failed to start with the following error:

```
File "/home/ubuntu/.../mle-bench/agents/utils.py", line 39, in parse_env_var_values
    raise ValueError(f"Environment variable `{env_var}` is not set!")
ValueError: Environment variable `GEMINI_API_KEY` is not set!
```

## Root Cause Analysis

### The Complete Flow

```
1. GitHub Actions Workflow (.github/workflows/run-mle-bench.yml)
   ├─ Step: "Run MLE-Bench"
   ├─ Sets environment variables (lines 132-140):
   │  ├─ ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
   │  ├─ OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
   │  ├─ DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
   │  └─ ❌ GEMINI_API_KEY: MISSING!
   │
   └─ Executes: ./RUN_AGENT_V5_KAGGLE.sh

2. RUN_AGENT_V5_KAGGLE.sh
   └─ Calls: python -u run_agent.py --agent-id agent_v5_kaggle

3. run_agent.py
   └─ Calls: agent_registry.get_agent("agent_v5_kaggle")

4. agents/registry.py (line 87)
   ├─ Reads: agents/agent_v5_kaggle/config.yaml
   ├─ Parses: env_vars = parse_env_var_values(env_vars)
   │
   └─ config.yaml declares:
      env_vars:
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
        GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}  # ← Requires this env var

5. agents/utils.py:parse_env_var_values (line 38-39)
   ├─ For each env_var in config.yaml:
   │  └─ Check: if os.getenv(env_var) is None:
   │     └─ ❌ FAILS: os.getenv("GEMINI_API_KEY") returns None
   │        └─ raise ValueError("Environment variable `GEMINI_API_KEY` is not set!")
   │
   └─ Process stops before Docker container even starts
```

### Why It Failed

**Mismatch between config.yaml and GitHub Actions workflow:**

- **config.yaml** (added in previous fix) declares: `GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}`
- **GitHub Actions workflow** did not export `GEMINI_API_KEY` as an environment variable
- **mle-bench registry** validates ALL environment variables declared in config.yaml
- **Validation fails** because `GEMINI_API_KEY` is not in the process environment

### Why Other API Keys Work

The other API keys (ANTHROPIC, OPENAI, DEEPSEEK) work because they are **BOTH**:
1. Declared in `config.yaml`
2. Exported in `.github/workflows/run-mle-bench.yml`

## Solution

Add `GEMINI_API_KEY` to the GitHub Actions workflow environment variables section.

## Changes Made

### File: `.github/workflows/run-mle-bench.yml`

**Location**: Lines 132-140 (Step: "Run MLE-Bench" → env section)

**Before**:
```yaml
- name: Run MLE-Bench
  timeout-minutes: 1400
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
    IMAGE_TAG: agent_v5_kaggle:run-${{ github.run_id }}
    RUN_ID: ${{ github.run_id }}
    DRY_RUN: ${{ github.event.inputs.dry_run }}
    REBUILD_IMAGE: ${{ github.event.inputs.rebuild_image }}
```

**After** (added line 136):
```yaml
- name: Run MLE-Bench
  timeout-minutes: 1400
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
    GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}  # <-- ADDED
    IMAGE_TAG: agent_v5_kaggle:run-${{ github.run_id }}
    RUN_ID: ${{ github.run_id }}
    DRY_RUN: ${{ github.event.inputs.dry_run }}
    REBUILD_IMAGE: ${{ github.event.inputs.rebuild_image }}
```

## How It Works Now

### Environment Variable Flow

```
1. GitHub Actions reads secret from repository secrets
   └─> secrets.GEMINI_API_KEY

2. GitHub Actions exports as environment variable
   └─> GEMINI_API_KEY=<secret_value>

3. RUN_AGENT_V5_KAGGLE.sh inherits environment
   └─> GEMINI_API_KEY is available in shell environment

4. python run_agent.py inherits environment
   └─> GEMINI_API_KEY available via os.getenv("GEMINI_API_KEY")

5. agents/registry.py validates environment variables
   └─> parse_env_var_values() checks os.getenv("GEMINI_API_KEY")
   └─> ✅ PASSES: Environment variable is set

6. Docker container receives environment variable
   └─> config.yaml: GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
   └─> Replaced with actual value from step 4

7. Agent initializes Gemini client
   └─> agent.py: genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
   └─> ✅ SUCCESS: Client initialized with valid API key
```

## Prerequisites

### ⚠️ IMPORTANT: GitHub Repository Secret Required

Before running the workflow, you MUST ensure the `GEMINI_API_KEY` secret is set in your GitHub repository:

**Steps**:
1. Go to: **Repository Settings** → **Secrets and variables** → **Actions**
2. Click: **New repository secret**
3. Name: `GEMINI_API_KEY`
4. Value: Your Gemini API key (get from https://aistudio.google.com/apikey)
5. Click: **Add secret**

**If the secret is not set**:
- GitHub Actions will pass an empty string `""` to the environment
- The validation in `parse_env_var_values()` will still **pass** (empty string is not None)
- But the agent will **fail later** when initializing the Gemini client:
  ```
  ValueError: Missing key inputs argument! To use the Google AI API, provide (`api_key`) arguments.
  ```

### Verification

To verify the secret is set correctly:
1. Go to: **Repository Settings** → **Secrets and variables** → **Actions**
2. Check that `GEMINI_API_KEY` appears in the list of repository secrets
3. Note: You cannot view the value (security), only verify it exists

## Context: Why Gemini?

The agent was recently switched from Anthropic/OpenAI to Gemini API:

```bash
$ git log --oneline -5 -- agent_v5/agent.py
4b0f58f fix to use gemini       # <-- This commit switched to Gemini
8bfdc6c Enhance ResearchAgent...
36d28e4 Refactor ResearchAgent to integrate OpenAI API
```

**Commit 4b0f58f** changed the agent to use Gemini, but:
- ✅ Updated `agent_v5/agent.py` to use `genai.Client`
- ✅ Updated `config.yaml` to declare `GEMINI_API_KEY`
- ❌ **Forgot to update** `.github/workflows/run-mle-bench.yml`

This fix completes the migration to Gemini.

## Files Modified (Total: 2)

### Previous Fixes (from import error)
1. **mle-bench/agents/agent_v5_kaggle/tools/__init__.py** (created)
   - Makes `tools/` a proper Python package

2. **mle-bench/agents/agent_v5_kaggle/kaggle_agent.py** (line 602)
   - Changed relative import to absolute import

3. **mle-bench/agents/agent_v5_kaggle/config.yaml** (line 9)
   - Added `GEMINI_API_KEY` to env_vars section

### This Fix (environment variable)
4. **.github/workflows/run-mle-bench.yml** (line 136)
   - Added `GEMINI_API_KEY` to GitHub Actions environment

## Testing

### Expected Behavior

After this fix, the workflow should:

1. ✅ Export `GEMINI_API_KEY` from GitHub secrets
2. ✅ Pass environment variable validation in `parse_env_var_values()`
3. ✅ Build Docker image successfully
4. ✅ Pass `GEMINI_API_KEY` to Docker container
5. ✅ Initialize Gemini client in agent
6. ✅ Start agent and begin competition

### Test Locally (Optional)

To test locally before running GitHub Actions:

```bash
# 1. Set environment variable
export GEMINI_API_KEY="your-api-key-here"

# 2. Run the agent
cd mle-bench
./RUN_AGENT_V5_KAGGLE.sh
```

If it works locally, it will work in GitHub Actions (assuming the secret is set).

## Prevention

To prevent similar issues in the future:

### Checklist: Adding New Environment Variables

When adding a new environment variable to an agent:

- [ ] Add to `config.yaml` env_vars section
- [ ] Add to `.github/workflows/run-mle-bench.yml` env section
- [ ] Add to GitHub repository secrets (if using secrets)
- [ ] Document in README or agent documentation
- [ ] Test locally with environment variable set
- [ ] Test in GitHub Actions

### Validation Script (Future Enhancement)

Consider adding a validation script to check consistency:

```python
# validate_env_vars.py
import yaml
import sys

# Read config.yaml
with open('mle-bench/agents/agent_v5_kaggle/config.yaml') as f:
    config = yaml.safe_load(f)
    config_env_vars = set(config['agent_v5_kaggle']['env_vars'].keys())

# Read workflow file
with open('.github/workflows/run-mle-bench.yml') as f:
    workflow = yaml.safe_load(f)
    # Parse env section...
    workflow_env_vars = set(...)  # Extract from workflow

# Compare
missing = config_env_vars - workflow_env_vars
if missing:
    print(f"❌ ERROR: config.yaml declares env vars not in workflow:")
    for var in missing:
        print(f"   - {var}")
    sys.exit(1)

print("✅ All environment variables are consistent")
```

## Summary

**Issue**: Agent failed because `GEMINI_API_KEY` was declared in `config.yaml` but not exported in GitHub Actions workflow

**Fix**: Added `GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}` to `.github/workflows/run-mle-bench.yml`

**Prerequisite**: Must set `GEMINI_API_KEY` as a GitHub repository secret

**Status**: ✅ Fixed - Agent should now start successfully

---

**Related Documentation**:
- [COMPLETE_FIX_SUMMARY.md](COMPLETE_FIX_SUMMARY.md) - Import error fix
- [IMPORT_FIX_SUMMARY.md](IMPORT_FIX_SUMMARY.md) - Detailed import analysis
- GitHub Actions Secrets: https://docs.github.com/en/actions/security-guides/encrypted-secrets
