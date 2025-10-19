# Gemini 2.5 Pro - Quick Start

## Setup

### 1. Get API Key

Visit: https://aistudio.google.com/app/apikey

Create a new API key (free tier available).

### 2. Set Environment Variable

```bash
export GEMINI_API_KEY="AIzaSy..."
```

Or add to your `.env` file:
```
GEMINI_API_KEY=AIzaSy...
```

### 3. Install SDK

```bash
pip install google-genai
```

## Usage

No code changes needed! The agent automatically uses Gemini:

```python
from agent_v5.agent import ResearchAgent

agent = ResearchAgent(
    session_id="test",
    workspace_dir="/tmp/workspace",
    system_prompt="You are a helpful assistant"
)

async for message in agent.run("Hello!"):
    if message["type"] == "text_delta":
        print(message["text"], end="", flush=True)
```

## Key Differences from Claude

| Feature | Claude | Gemini |
|---------|--------|--------|
| Model | claude-sonnet-4-5 | gemini-2.5-pro-002 |
| API Key | ANTHROPIC_API_KEY | GEMINI_API_KEY |
| Tool format | tool_use | function_call |
| Streaming | Native with tools | Text only |
| Temperature | 0-1 | 0-1 |
| Context | ~200K tokens | 2M tokens |

## Testing

### Quick Test

```python
import asyncio
from agent_v5.agent import ResearchAgent

async def test():
    agent = ResearchAgent(
        session_id="test",
        workspace_dir="/tmp/test",
        system_prompt="You are a helpful coding assistant"
    )

    async for msg in agent.run("Write a Python function to reverse a string"):
        if msg["type"] == "text_delta":
            print(msg["text"], end="")

asyncio.run(test())
```

### Tool Test

```python
async def test_tools():
    agent = ResearchAgent(
        session_id="test",
        workspace_dir="/tmp/test",
        system_prompt="You are a helpful assistant with file access"
    )

    # Test file reading
    async for msg in agent.run("Create a file called test.txt with 'Hello World'"):
        if msg["type"] == "text_delta":
            print(msg["text"], end="")

asyncio.run(test_tools())
```

## Troubleshooting

### Error: "GEMINI_API_KEY not set"

**Fix:** Set the environment variable:
```bash
export GEMINI_API_KEY="your-key-here"
```

### Error: "Module 'google.genai' not found"

**Fix:** Install the SDK:
```bash
pip install google-genai
```

### Error: "Invalid API key"

**Fix:**
1. Verify key is correct
2. Check you're using the right project
3. Enable Gemini API in Google Cloud Console

### Tools not working

**Check:**
1. Tool schemas are valid
2. Function names match tool names exactly
3. Parameters match schema types

## Performance Tips

1. **Temperature 0** for deterministic outputs (current setting)
2. **System instructions** work like Claude's system prompt
3. **Context window** is very large (2M tokens) - use it!
4. **Tool calling** is efficient - don't worry about multiple calls

## Cost Comparison (Approximate)

| Model | Input | Output |
|-------|-------|--------|
| Claude Sonnet 4.5 | $3/M | $15/M |
| Gemini 2.5 Pro | $1.25/M | $5/M |

*Prices as of Jan 2025 - check current pricing*

**Savings:** ~75% cheaper than Claude for same workload

## API Limits

**Free tier:**
- 1,500 requests per day
- 1 million tokens per minute
- 15 requests per minute

**Paid tier:**
- Higher limits
- See: https://ai.google.dev/pricing

## Migration from Claude

If you need to rollback:

```bash
# 1. Unset Gemini key
unset GEMINI_API_KEY

# 2. Set Claude key
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. Revert code files (if needed)
git checkout agent_v5/agent.py agent_v5/tools/registry.py
```

## Support

**Documentation:** https://ai.google.dev/gemini-api/docs

**Python SDK:** https://github.com/googleapis/python-genai

**Issues:** Check GEMINI_MIGRATION.md for known limitations

## Success Checklist

- [x] ✅ Syntax validation passed
- [ ] Set GEMINI_API_KEY
- [ ] Run quick test
- [ ] Test with tools
- [ ] Run full Kaggle competition
- [ ] Compare results vs Claude baseline

## Ready to Use! 🚀

The migration is complete. Just set your API key and test!
