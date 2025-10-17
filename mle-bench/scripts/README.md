# MLE-Bench Helper Scripts

Utility scripts for working with MLE-Bench.

## accept_kaggle_rules.py

**Python script to accept Kaggle competition rules for multiple competitions.**

### Features
- ✅ Checks which competitions you've already accepted via Kaggle API
- ✅ Only opens unaccepted competitions
- ✅ Verifies acceptance after you click
- ✅ Progress tracking with status indicators

### Usage

```bash
# Check which competitions need acceptance (read-only, no browser)
python scripts/accept_kaggle_rules.py --check-only

# Accept rules for all unaccepted competitions in a set
python scripts/accept_kaggle_rules.py experiments/splits/custom-set.txt

# Force open all competitions, even if already accepted
python scripts/accept_kaggle_rules.py --force experiments/splits/custom-set.txt
```

### Requirements
- Kaggle API credentials configured (`~/.kaggle/kaggle.json`)
- `mlebench` package installed (`pip install -e .`)

### Example Output

```
==================================================
Kaggle Competition Rules Accepter
==================================================
Competition set: experiments/splits/custom-set.txt
Total competitions: 10

Authenticating with Kaggle API...
✅ Authenticated successfully

Checking competition acceptance status...

[1/10] spaceship-titanic... ✅ Accepted
[2/10] whale-redux... ❌ Not accepted
[3/10] connectx... ✅ Accepted
[4/10] house-prices... ❌ Not accepted
...

==================================================
Summary
==================================================
✅ Already accepted: 8
❌ Not accepted: 2
⚠️  Unknown status: 0

Will process 2 competitions
```

---

## accept-kaggle-rules.sh

**Bash script to accept Kaggle competition rules (simpler, no API calls).**

### Features
- ✅ Simple and fast
- ✅ Works on macOS, Linux, WSL
- ✅ Opens rules pages sequentially
- ✅ No Python/API dependencies

### Usage

```bash
# Accept rules for all competitions in a set
./scripts/accept-kaggle-rules.sh experiments/splits/custom-set.txt

# Use default competition set (custom-set.txt)
./scripts/accept-kaggle-rules.sh
```

### Requirements
- `open` (macOS), `xdg-open` (Linux), or `wslview` (WSL)

### Example Output

```
==========================================
Kaggle Competition Rules Accepter
==========================================
Competition set: experiments/splits/custom-set.txt
Total competitions: 10

Press Enter to start...

==========================================
[1/10] spaceship-titanic
==========================================
Opening: https://www.kaggle.com/c/spaceship-titanic/rules

Instructions:
  1. Scroll to bottom of the page
  2. Click 'I Understand and Accept'
  3. Return here and press Enter to continue

Press Enter after accepting rules...
✅ Marked as accepted
```

---

## When to Use Which Script

| Scenario | Recommended Script |
|----------|-------------------|
| Large competition set (>5 competitions) | `accept_kaggle_rules.py` - checks acceptance status first |
| Small set or first-time acceptance | Either script works |
| No Python/dependencies available | `accept-kaggle-rules.sh` |
| Want to verify acceptance | `accept_kaggle_rules.py` |
| Just need to open URLs quickly | `accept-kaggle-rules.sh` |

---

## Common Workflow

**Before running large MLE-Bench experiments:**

```bash
# 1. Check which competitions need acceptance
python scripts/accept_kaggle_rules.py --check-only experiments/splits/custom-set.txt

# 2. Accept rules for unaccepted competitions
python scripts/accept_kaggle_rules.py experiments/splits/custom-set.txt

# 3. Run your experiment (won't get 403 errors)
./RUN_AGENT_V5_KAGGLE.sh
```

**In CI/CD:**

```bash
# Accept rules locally BEFORE pushing to CI
python scripts/accept_kaggle_rules.py experiments/splits/custom-set.txt

# Then in CI, set this env var to handle any remaining unaccepted competitions
export KAGGLE_AUTO_ACCEPT_RULES=1
```

---

## Troubleshooting

### "No module named 'mlebench'"

**Fix:** Install mlebench package
```bash
cd mle-bench
pip install -e .
```

### "Could not authenticate with Kaggle API"

**Fix:** Configure Kaggle credentials
```bash
mkdir -p ~/.kaggle
# Add your credentials to ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

See: https://github.com/Kaggle/kaggle-api#api-credentials

### "Could not auto-open browser"

**Bash script issue:** Your system doesn't have `open`, `xdg-open`, or `wslview`.

**Fix:** Use the Python script instead (uses `webbrowser` module):
```bash
python scripts/accept_kaggle_rules.py
```

Or manually visit the URLs printed by the script.

---

## Related Documentation

- [KAGGLE_RULES_ACCEPTANCE.md](../KAGGLE_RULES_ACCEPTANCE.md) - Full guide to handling competition rules
- [Kaggle API docs](https://github.com/Kaggle/kaggle-api) - Official Kaggle API documentation

---

**Last updated:** 2025-10-16
