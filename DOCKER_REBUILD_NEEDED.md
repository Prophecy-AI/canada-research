# Docker Rebuild Required for Gemini Migration

## Why Rebuild?

The `requirements.txt` file was updated to replace `anthropic` with `google-genai`:

```diff
- anthropic==0.34.1
+ google-genai==1.0.1
```

The Docker image needs to be rebuilt to install the new package.

## How to Rebuild

### Option 1: Rebuild Base Image

```bash
cd /Users/Yifan/canada-research/mle-bench

# Rebuild the base environment image
docker build -t mlebench-env:latest -f environment/Dockerfile .

# Then rebuild the agent image
docker build -t agent_v5_kaggle:latest -f agents/agent_v5_kaggle/Dockerfile .
```

### Option 2: Use Build Script (if available)

```bash
cd /Users/Yifan/canada-research/mle-bench
./build.sh  # Or whatever build script you use
```

### Option 3: Modal Deploy (if using Modal)

```bash
cd /Users/Yifan/canada-research/mle-bench
modal deploy agents/agent_v5_kaggle/main.py
```

Modal will automatically rebuild the image with the new requirements.

## Environment Variable

**Don't forget to set the API key!**

```bash
# For local Docker:
docker run -e GEMINI_API_KEY="AIza..." ...

# For Modal:
modal secret create gemini-api-key GEMINI_API_KEY="AIza..."
```

## Verification

After rebuilding, verify the package is installed:

```bash
# Test in Docker container:
docker run -it mlebench-env:latest bash
conda activate agent
python -c "from google import genai; print('✓ google-genai installed')"
```

Expected output: `✓ google-genai installed`

## Troubleshooting

### Error: "cannot import name 'genai'"

**Cause:** Docker image not rebuilt or build cache used old requirements.txt

**Fix:**
```bash
# Force rebuild without cache
docker build --no-cache -t mlebench-env:latest -f environment/Dockerfile .
```

### Error: "GEMINI_API_KEY not set"

**Cause:** Environment variable not passed to container

**Fix:**
```bash
# Add -e flag when running docker
docker run -e GEMINI_API_KEY="your-key" ...
```

### Build is slow

**Why:** Installing all packages takes time (~10-15 min)

**Tip:** Only rebuild when requirements change. Use cached layers when possible.

## Files Changed

1. ✅ `mle-bench/environment/requirements.txt` - Line 3: `anthropic` → `google-genai`

## Ready to Rebuild! 🔨

Once rebuilt, the agent will use Gemini 2.5 Pro instead of Claude Sonnet 4.5.
