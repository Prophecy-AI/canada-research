# Kaggle Competition Rules Acceptance

## Problem

When running `mlebench` in automated/CI environments, you may encounter this error:

```
kaggle.rest.ApiException: (403)
Reason: Forbidden
HTTP response body: "You must accept this competition's rules before you'll be able to download files."
```

Followed by:

```
EOFError: EOF when reading a line
Would you like to open the competition page in your browser now? (y/n):
```

This happens because:
1. Kaggle requires you to accept competition rules before downloading datasets
2. The code tries to prompt for input, but there's no stdin in CI/CD environments

## Solution

### For Automated Environments (CI/CD, GitHub Actions)

**Step 1: Manually Accept Rules**

Before running your automated pipeline, you MUST manually accept the competition rules:

1. Go to the competition page: `https://www.kaggle.com/c/{competition-id}/rules`
2. Scroll to the bottom and click "I Understand and Accept"
3. This only needs to be done ONCE per competition per Kaggle account

**Step 2: Set Environment Variable**

The `RUN_AGENT_V5_KAGGLE.sh` script automatically sets this:

```bash
export KAGGLE_AUTO_ACCEPT_RULES=1
```

This tells `mlebench` to:
- ✅ Skip interactive prompts
- ✅ Provide clear error messages with the rules URL
- ✅ Fail gracefully in non-interactive environments

### For Local/Interactive Environments

When running locally with a terminal, the original behavior works:

```bash
# Don't set KAGGLE_AUTO_ACCEPT_RULES
mlebench prepare -c {competition-id}

# You'll be prompted:
# "Would you like to open the competition page in your browser now? (y/n):"
# Type 'y' and accept rules in browser
```

## How It Works

The updated `mlebench/data.py` now:

1. **Detects environment type:**
   - `sys.stdin.isatty()` checks if running interactively
   - `KAGGLE_AUTO_ACCEPT_RULES` env var for automated mode

2. **Handles three cases:**

   **Case A: Interactive mode (local terminal)**
   - Original behavior: prompt user, open browser

   **Case B: Non-interactive + auto-accept flag**
   - Fails with clear instructions and URL
   - User must accept rules manually, then re-run

   **Case C: Non-interactive + no flag**
   - Fails with error explaining the requirement
   - Prevents cryptic `EOFError`

## Helper Scripts for Multiple Competitions

If you're running against multiple competitions (e.g., `custom-set.txt`), we provide helper scripts to accept rules in bulk:

### Option 1: Python Script (Recommended)

**Features:**
- ✅ Checks acceptance status via Kaggle API
- ✅ Only opens unaccepted competitions
- ✅ Verifies acceptance after you click
- ✅ Progress tracking

**Usage:**

```bash
cd mle-bench

# Check which competitions need acceptance (no browser opens)
python scripts/accept_kaggle_rules.py --check-only

# Output:
# [1/5] competition-1... ✅ Accepted
# [2/5] competition-2... ❌ Not accepted
# [3/5] competition-3... ✅ Accepted
# ...

# Open rules pages for all unaccepted competitions
python scripts/accept_kaggle_rules.py experiments/splits/custom-set.txt

# The script will:
# 1. Open each unaccepted competition in your browser
# 2. Wait for you to click "I Understand and Accept"
# 3. Verify the acceptance
# 4. Move to next competition
```

### Option 2: Bash Script (Simpler)

**Features:**
- ✅ Simple and fast
- ✅ Works on macOS, Linux, WSL
- ✅ No API calls needed

**Usage:**

```bash
cd mle-bench

# Opens all competitions in your browser (one at a time)
./scripts/accept-kaggle-rules.sh experiments/splits/custom-set.txt

# Press Enter after accepting each competition's rules
```

### Manual Approach

If you prefer to do it manually:

```bash
# View all competitions in your run
cat mle-bench/experiments/splits/custom-set.txt

# For each competition, visit:
# https://www.kaggle.com/c/{competition-id}/rules
# and click "I Understand and Accept"
```

## Example: CI/CD Workflow

```yaml
# .github/workflows/mle-bench.yml
name: Run MLE Bench
on: push

jobs:
  run-agent:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v3

      - name: Setup Kaggle credentials
        run: |
          mkdir -p ~/.kaggle
          echo '{"username":"${{ secrets.KAGGLE_USERNAME }}","key":"${{ secrets.KAGGLE_KEY }}"}' > ~/.kaggle/kaggle.json
          chmod 600 ~/.kaggle/kaggle.json

      # IMPORTANT: Accept rules manually BEFORE running CI!
      # Use the helper scripts locally first:
      #   python scripts/accept_kaggle_rules.py experiments/splits/custom-set.txt

      - name: Run agent
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          KAGGLE_AUTO_ACCEPT_RULES: 1  # Prevents interactive prompts
        run: |
          cd mle-bench
          ./RUN_AGENT_V5_KAGGLE.sh
```

## Troubleshooting

### Error: "Competition rules not yet accepted"

**Cause:** You haven't accepted the competition rules on Kaggle yet.

**Fix:**
1. Visit the URL shown in the error message
2. Click "I Understand and Accept"
3. Re-run your script

### Error: "Cannot prompt for input in non-interactive mode"

**Cause:** Running in CI/CD without `KAGGLE_AUTO_ACCEPT_RULES=1`

**Fix:**
```bash
export KAGGLE_AUTO_ACCEPT_RULES=1
```

### Still getting EOFError?

**Cause:** You're using an old version of `mlebench/data.py`

**Fix:**
```bash
# Ensure you have the updated version
cd mle-bench
pip install -e . --force-reinstall
```

## Technical Details

### Modified Function

```python
# mlebench/data.py
def _prompt_user_to_accept_rules(competition_id: str) -> None:
    import sys

    is_interactive = sys.stdin.isatty()
    auto_accept = os.getenv("KAGGLE_AUTO_ACCEPT_RULES", "0") == "1"
    rules_url = f"https://www.kaggle.com/c/{competition_id}/rules"

    if auto_accept:
        # Fail gracefully with instructions
        raise RuntimeError(f"Visit {rules_url} to accept rules")

    if not is_interactive:
        # Prevent EOFError
        raise RuntimeError(f"Cannot prompt in non-interactive mode")

    # Original interactive behavior
    response = input("Open competition page? (y/n): ")
    # ...
```

### Retry Logic

Note that `download_dataset()` is decorated with `@retry`:

```python
@retry(
    retry=retry_if_exception(is_api_exception),
    stop=stop_after_attempt(3),
    wait=wait_fixed(5)
)
def download_dataset(...):
    # ...
```

This means:
- If rules aren't accepted, it will retry 3 times with 5s wait
- After accepting rules, the retry will succeed
- No need to restart the entire process

## Summary

✅ **Do this ONCE per competition:**
- Visit `https://www.kaggle.com/c/{competition-id}/rules`
- Click "I Understand and Accept"

✅ **Set in CI/CD environments:**
```bash
export KAGGLE_AUTO_ACCEPT_RULES=1
```

✅ **Local development:**
- Just run `mlebench prepare -c {competition}`
- Follow the prompts

---

**Last updated:** 2025-10-16
