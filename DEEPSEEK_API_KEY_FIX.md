# DEEPSEEK_API_KEY Fix - Complete

## Problem

The upgraded Oracle tool requires `DEEPSEEK_API_KEY` to query DeepSeek-R1 in parallel with O3, but the environment variable was not being passed to the Docker container.

**Error seen:**
```
[2025-10-16 22:10:27,002] dog-breed-identification ✗ Oracle: Error: DEEPSEEK_API_KEY environment variable not set. Cannot consult Oracle.
```

Even though `DEEPSEEK_API_KEY` was set in GitHub Actions secrets, it wasn't configured to be passed through to the container.

---

## Root Cause

The environment variable flow is:

```
GitHub Actions Secrets
  ↓
GitHub Actions Workflow (run-mle-bench.yml)
  ↓
Host Environment Variables
  ↓
Agent Config (config.yaml)
  ↓
Docker Container
  ↓
Oracle Tool
```

**Missing links:** Steps 2 and 3 were not configured for `DEEPSEEK_API_KEY`.

---

## Files Changed

### 1. `.github/workflows/run-mle-bench.yml`

**Before:**
```yaml
env:
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  IMAGE_TAG: agent_v5_kaggle:run-${{ github.run_id }}
```

**After:**
```yaml
env:
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}  # ← ADDED
  IMAGE_TAG: agent_v5_kaggle:run-${{ github.run_id }}
```

**Line:** [.github/workflows/run-mle-bench.yml:140](.github/workflows/run-mle-bench.yml#L140)

### 2. `mle-bench/agents/agent_v5_kaggle/config.yaml`

**Before:**
```yaml
env_vars:
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  DEBUG: "1"
```

**After:**
```yaml
env_vars:
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}  # ← ADDED
  DEBUG: "1"
```

**Line:** [mle-bench/agents/agent_v5_kaggle/config.yaml:8](mle-bench/agents/agent_v5_kaggle/config.yaml#L8)

---

## How It Works Now

1. **GitHub Actions reads secret:** `${{ secrets.DEEPSEEK_API_KEY }}`
2. **Workflow sets env var:** `DEEPSEEK_API_KEY` in host environment
3. **Agent config reads it:** `parse_env_var_values()` replaces `${{ secrets.DEEPSEEK_API_KEY }}` with actual value
4. **Container receives it:** `agents/run.py:142-145` passes `agent.env_vars` to Docker container
5. **Oracle can use it:** `os.environ.get("DEEPSEEK_API_KEY")` works inside container

---

## Verification

**Before fix:**
```
✗ Oracle: Error: DEEPSEEK_API_KEY environment variable not set
```

**After fix (expected):**
```
🔮 Oracle: Consulting O3 and DeepSeek-R1 in parallel...
✓ Oracle: Received 2 plans, synthesizing optimal strategy...
```

---

## Next Steps

1. **Commit these changes:**
   ```bash
   git add .github/workflows/run-mle-bench.yml
   git add mle-bench/agents/agent_v5_kaggle/config.yaml
   git commit -m "Fix: Pass DEEPSEEK_API_KEY to Docker container for Oracle tool"
   git push
   ```

2. **Verify GitHub secret is set:**
   - Go to: https://github.com/YOUR_REPO/settings/secrets/actions
   - Confirm `DEEPSEEK_API_KEY` exists
   - If not, add it with your DeepSeek API key from https://platform.deepseek.com/api_keys

3. **Test the fix:**
   - Run a workflow manually
   - Check logs for: `🔮 Oracle: Consulting O3 and DeepSeek-R1 in parallel...`
   - Verify no more `DEEPSEEK_API_KEY environment variable not set` errors

---

## Technical Details

**Config syntax:** `${{ secrets.DEEPSEEK_API_KEY }}`
- This is **not** standard environment variable syntax
- It's parsed by `agents/utils.py::parse_env_var_values()`
- Regex pattern: `r"\$\{\{\s*secrets\.(\w+)\s*\}\}"`
- Extracts variable name and reads from `os.getenv()`

**Why ANTHROPIC_API_KEY worked but DEEPSEEK_API_KEY didn't:**
- `ANTHROPIC_API_KEY` was already configured in both files
- `DEEPSEEK_API_KEY` is new (added for Oracle upgrade)
- Both files needed updating for new key

---

## Related Files

- [agents/run.py:142-145](agents/run.py#L142-L145) - Where env vars pass to container
- [agents/registry.py:82](agents/registry.py#L82) - Where env_vars are read from config
- [agents/utils.py:27-43](agents/utils.py#L27-L43) - Parser for `${{ secrets.X }}` syntax
- [agent_v5/tools/oracle.py:97-104](agent_v5/tools/oracle.py#L97-L104) - Where DeepSeek client is created

---

**Status:** ✅ **FIXED** - All configuration updated, ready to test.
