# Oracle Tool Configuration Guide

## Overview

The upgraded Oracle tool requires **two API keys**:
1. **OpenAI API Key** - for O3 model (initial plan + critic synthesis)
2. **DeepSeek API Key** - for DeepSeek-R1 model (alternative plan)

---

## Setup Instructions

### Step 1: Get OpenAI API Key

**URL**: https://platform.openai.com/api-keys

1. Sign in to OpenAI Platform
2. Navigate to API Keys section
3. Click "Create new secret key"
4. Copy the key (starts with `sk-...`)

**Pricing** (O3):
- Input: ~$0.03/1K tokens
- Output: ~$0.06/1K tokens

### Step 2: Get DeepSeek API Key

**URL**: https://platform.deepseek.com/api_keys

1. Sign in to DeepSeek Platform
2. Navigate to API Keys section
3. Click "Create API Key"
4. Copy the key (starts with `sk-...`)

**Pricing** (DeepSeek-R1):
- Both input and CoT reasoning: $2.19 per million tokens
- **Much cheaper than O3!** (~1/10th the cost)

### Step 3: Set Environment Variables

#### On Linux/Mac:

**Option 1: Export (temporary - current session only)**
```bash
export OPENAI_API_KEY="sk-your-openai-key-here"
export DEEPSEEK_API_KEY="sk-your-deepseek-key-here"
```

**Option 2: Add to `.bashrc` or `.bash_profile` (permanent)**
```bash
echo 'export OPENAI_API_KEY="sk-your-openai-key-here"' >> ~/.bashrc
echo 'export DEEPSEEK_API_KEY="sk-your-deepseek-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**Option 3: Create `.env` file (project-specific)**
```bash
# In project root: /Users/Yifan/canada-research/
cat > .env << 'EOF'
OPENAI_API_KEY=sk-your-openai-key-here
DEEPSEEK_API_KEY=sk-your-deepseek-key-here
EOF

# Load .env in your script
# (python-dotenv package required: pip install python-dotenv)
from dotenv import load_dotenv
load_dotenv()
```

#### On Windows:

**Option 1: Command Prompt (temporary)**
```cmd
set OPENAI_API_KEY=sk-your-openai-key-here
set DEEPSEEK_API_KEY=sk-your-deepseek-key-here
```

**Option 2: PowerShell (temporary)**
```powershell
$env:OPENAI_API_KEY="sk-your-openai-key-here"
$env:DEEPSEEK_API_KEY="sk-your-deepseek-key-here"
```

**Option 3: System Environment Variables (permanent)**
```cmd
setx OPENAI_API_KEY "sk-your-openai-key-here"
setx DEEPSEEK_API_KEY "sk-your-deepseek-key-here"
```

### Step 4: Verify Configuration

**Python test script:**
```python
import os

# Check OpenAI API key
openai_key = os.environ.get("OPENAI_API_KEY")
if openai_key:
    print(f"✅ OPENAI_API_KEY set (starts with {openai_key[:7]}...)")
else:
    print("❌ OPENAI_API_KEY not set")

# Check DeepSeek API key
deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
if deepseek_key:
    print(f"✅ DEEPSEEK_API_KEY set (starts with {deepseek_key[:7]}...)")
else:
    print("❌ DEEPSEEK_API_KEY not set")

# Test imports
try:
    from openai import OpenAI
    print("✅ OpenAI SDK installed")
except ImportError:
    print("❌ OpenAI SDK not installed (run: pip install openai)")

# Test Oracle import
try:
    from agent_v5.tools.oracle import OracleTool
    print("✅ OracleTool imports successfully")
except Exception as e:
    print(f"❌ OracleTool import failed: {e}")
```

**Run verification:**
```bash
python verify_oracle_config.py
```

**Expected output:**
```
✅ OPENAI_API_KEY set (starts with sk-proj...)
✅ DEEPSEEK_API_KEY set (starts with sk-abc1...)
✅ OpenAI SDK installed
✅ OracleTool imports successfully
```

---

## API Details

### OpenAI O3 API

**Base URL**: `https://api.openai.com/v1` (default)

**Model**: `o3`

**Usage in Oracle**:
- Initial strategic plan (Phase 1)
- Critic synthesis (Phase 2)

**Example request:**
```python
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
response = client.chat.completions.create(
    model="o3",
    messages=[{"role": "user", "content": "..."}],
    max_completion_tokens=8192,
    temperature=1.0
)
```

### DeepSeek-R1 API

**Base URL**: `https://api.deepseek.com`

**Model**: `deepseek-reasoner` (reasoning mode with Chain of Thought)

**Usage in Oracle**:
- Alternative strategic plan (Phase 1)

**Example request:**
```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"  # CRITICAL: Must set base_url
)
response = client.chat.completions.create(
    model="deepseek-reasoner",  # NOT "deepseek-r1"
    messages=[{"role": "user", "content": "..."}],
    max_completion_tokens=8192,
    temperature=1.0
)
```

**Key differences from OpenAI:**
- Must explicitly set `base_url="https://api.deepseek.com"`
- Model name is `deepseek-reasoner` (not `deepseek-r1`)
- Generates Chain of Thought (CoT) reasoning before final answer
- Much cheaper pricing ($2.19/million tokens vs OpenAI's higher rates)

---

## Implementation in Oracle Tool

The Oracle tool (`agent_v5/tools/oracle.py`) initializes both clients:

```python
async def execute(self, input: Dict) -> Dict:
    # Check OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        return {"content": "Error: OPENAI_API_KEY not set", "is_error": True}

    # Check DeepSeek API key
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        return {
            "content": "Error: DEEPSEEK_API_KEY not set.\n"
                      "Get your key at: https://platform.deepseek.com/api_keys",
            "is_error": True
        }

    # Initialize clients
    openai_client = OpenAI(api_key=openai_api_key)
    deepseek_client = OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com"
    )

    # Query both in parallel
    o3_plan, deepseek_plan = await asyncio.gather(
        self._query_o3(openai_client, messages),
        self._query_deepseek_r1(deepseek_client, messages)
    )

    # Critic synthesis
    final_plan = await self._critic_synthesis(openai_client, messages, ...)
```

---

## Cost Estimation

### Per Oracle Consultation

**Assumptions:**
- Conversation history: ~30K input tokens (shared by both models)
- Oracle query: ~100 input tokens
- O3 output (Phase 1): ~8K tokens
- DeepSeek-R1 output (Phase 1): ~8K tokens
- O3 Critic output (Phase 2): ~16K tokens

**OpenAI O3 Costs:**
- Input: 30K tokens × 2 calls = 60K tokens × $0.03/1K = **$1.80**
- Output: (8K + 16K) tokens × $0.06/1K = **$1.44**
- **O3 Total: ~$3.24**

**DeepSeek-R1 Costs:**
- Input + CoT + Output: 30K + 8K = 38K tokens × $0.00219/1K = **$0.08**
- **DeepSeek Total: ~$0.08**

**Grand Total per consultation: ~$3.32**

### Cost Optimization Strategies

#### 1. Selective Ensemble (Recommended)

Use multi-model only for critical queries:

```python
CRITICAL_KEYWORDS = [
    "mismatch", "stuck", "strategy", "gold medal",
    "bug", "leaderboard", "pivot", "plan"
]

use_ensemble = any(kw in query.lower() for kw in CRITICAL_KEYWORDS)

if use_ensemble:
    # Use full multi-model Oracle (~$3.32)
else:
    # Use single O3 call (~$0.50)
```

**Estimated savings**: 50-70% reduction in Oracle costs

#### 2. Tiered Oracle

```python
# Tier 1: Quick questions - O3-mini (~$0.10)
# Tier 2: Standard questions - O3 only (~$0.50)
# Tier 3: Critical questions - Full ensemble (~$3.32)

if "quick" in query or len(query) < 50:
    tier = 1  # O3-mini
elif any(kw in query for kw in CRITICAL_KEYWORDS):
    tier = 3  # Full ensemble
else:
    tier = 2  # O3 only
```

#### 3. Caching

Cache Oracle responses for similar queries:

```python
import hashlib

query_hash = hashlib.sha256(query.encode()).hexdigest()
cache_file = f".oracle_cache/{query_hash}.json"

if os.path.exists(cache_file):
    # Return cached response (~$0)
    return json.load(open(cache_file))
else:
    # Query Oracle and cache result
    result = await oracle.execute(...)
    json.dump(result, open(cache_file, "w"))
    return result
```

---

## Troubleshooting

### Issue 1: "OPENAI_API_KEY not set"

**Solution:**
```bash
# Check if set
echo $OPENAI_API_KEY

# If empty, set it
export OPENAI_API_KEY="sk-your-key-here"

# Verify
python -c "import os; print(os.environ.get('OPENAI_API_KEY'))"
```

### Issue 2: "DEEPSEEK_API_KEY not set"

**Solution:**
```bash
# Check if set
echo $DEEPSEEK_API_KEY

# If empty, set it
export DEEPSEEK_API_KEY="sk-your-key-here"

# Verify
python -c "import os; print(os.environ.get('DEEPSEEK_API_KEY'))"
```

### Issue 3: "OpenAI SDK not installed"

**Solution:**
```bash
pip install openai
# or
pip install -r requirements.txt
```

### Issue 4: DeepSeek API returns 401 Unauthorized

**Causes:**
- Invalid API key
- Expired API key
- Wrong base_url

**Solution:**
```python
# Test DeepSeek API key
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

try:
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": "Hello"}],
        max_completion_tokens=100
    )
    print("✅ DeepSeek API key valid")
except Exception as e:
    print(f"❌ Error: {e}")
```

### Issue 5: "Model 'deepseek-reasoner' not found"

**Cause:** Wrong model name or base_url not set

**Solution:**
```python
# WRONG:
client = OpenAI(api_key=deepseek_key)  # Missing base_url!
response = client.chat.completions.create(model="deepseek-r1", ...)  # Wrong name!

# CORRECT:
client = OpenAI(
    api_key=deepseek_key,
    base_url="https://api.deepseek.com"  # Required!
)
response = client.chat.completions.create(
    model="deepseek-reasoner",  # Correct name
    ...
)
```

### Issue 6: Rate Limits

**OpenAI O3 limits:**
- Varies by tier (check platform.openai.com)
- Typically: 10K-100K requests/day

**DeepSeek limits:**
- Check platform.deepseek.com for current limits

**Solution:** Implement exponential backoff:
```python
import time

for attempt in range(3):
    try:
        response = client.chat.completions.create(...)
        break
    except Exception as e:
        if "rate_limit" in str(e).lower():
            wait_time = 2 ** attempt  # 1s, 2s, 4s
            print(f"Rate limited. Waiting {wait_time}s...")
            time.sleep(wait_time)
        else:
            raise
```

---

## Security Best Practices

### ✅ DO:
- Store API keys in environment variables
- Use `.env` files (add to `.gitignore`)
- Rotate API keys regularly
- Use separate keys for dev/prod
- Monitor API usage on platforms

### ❌ DON'T:
- Hardcode API keys in source code
- Commit API keys to version control
- Share API keys in Slack/email
- Use same key across multiple projects
- Expose keys in logs or error messages

### Example `.gitignore`:
```
# API Keys
.env
.env.local
.env.*.local

# Credentials
credentials.json
secrets.yaml
api_keys.txt
```

---

## Testing the Oracle Tool

### Minimal Test Script

```python
import asyncio
import os
from agent_v5.tools.oracle import OracleTool

async def test_oracle():
    # Mock conversation history getter
    def get_history():
        return [
            {"role": "user", "content": "I need help with a Kaggle competition"},
            {"role": "assistant", "content": [{"type": "text", "text": "Sure! Tell me about it."}]}
        ]

    # Initialize Oracle
    oracle = OracleTool(
        workspace_dir="/tmp/test",
        get_conversation_history=get_history
    )

    # Test query
    result = await oracle.execute({
        "query": "What's the best strategy for a binary classification task with imbalanced data?"
    })

    # Check result
    if result["is_error"]:
        print(f"❌ Error: {result['content']}")
    else:
        print("✅ Oracle consultation successful!")
        print(f"\n{result['content'][:500]}...")  # Show first 500 chars

# Run test
asyncio.run(test_oracle())
```

**Expected output:**
```
🔮 Oracle: Consulting O3 and DeepSeek-R1 in parallel...
🔮 Oracle: O3 Critic synthesizing optimal plan...
✅ Oracle consultation successful!

🔮 **ORACLE CONSULTATION (Multi-Model Ensemble)**

The Oracle consulted two reasoning models in parallel...
[Full response with 3 plans]
```

---

## Summary

**Required Environment Variables:**
```bash
export OPENAI_API_KEY="sk-your-openai-key"      # Get at: platform.openai.com
export DEEPSEEK_API_KEY="sk-your-deepseek-key"  # Get at: platform.deepseek.com
```

**API Endpoints:**
- OpenAI O3: `https://api.openai.com/v1` (default)
- DeepSeek R1: `https://api.deepseek.com` (must specify)

**Model Names:**
- OpenAI: `o3`
- DeepSeek: `deepseek-reasoner` (NOT `deepseek-r1`)

**Cost per consultation:** ~$3.32
- O3: ~$3.24
- DeepSeek-R1: ~$0.08

**Ready to use!** 🚀
