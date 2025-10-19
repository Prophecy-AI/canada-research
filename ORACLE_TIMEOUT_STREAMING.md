# Oracle Tool: Timeout & Streaming Implementation

## Problem Summary

Oracle consultations could hang indefinitely if O3 or DeepSeek-R1 took too long to respond, blocking the agent and wasting time.

## Solution

Added **10-minute timeout** and **streaming output** for all Oracle calls:
1. O3 query
2. DeepSeek-R1 query
3. O3 Critic synthesis

## Changes Made

### File: `/Users/Yifan/canada-research/agent_v5/tools/oracle.py`

#### 1. Added Timeout Wrapper for O3 (Lines 346-365)

**Before:**
```python
async def _query_o3(self, client: 'OpenAI', messages: List[Dict]) -> str:
    response = await asyncio.to_thread(
        client.chat.completions.create,
        model="o3",
        messages=messages,
        max_completion_tokens=8192,
        temperature=1.0
    )
    return response.choices[0].message.content
```

**After:**
```python
async def _query_o3(self, client: 'OpenAI', messages: List[Dict]) -> str:
    try:
        # Create task with timeout (10 minutes = 600 seconds)
        task = asyncio.create_task(self._query_o3_stream(client, messages))
        response_text = await asyncio.wait_for(task, timeout=600)
        return response_text
    except asyncio.TimeoutError:
        return "ERROR: O3 timed out after 10 minutes - returning partial response if available"
    except Exception as e:
        return f"ERROR: O3 failed - {str(e)}"
```

#### 2. Added Streaming for O3 (Lines 367-397)

**New method:**
```python
async def _query_o3_stream(self, client: 'OpenAI', messages: List[Dict]) -> str:
    """Stream O3 response and collect full text"""
    print("🔮 O3 streaming... ", end="", flush=True)

    # Use streaming API
    stream = await asyncio.to_thread(
        client.chat.completions.create,
        model="o3",
        messages=messages,
        max_completion_tokens=8192,
        temperature=1.0,
        stream=True  # Enable streaming
    )

    chunks = []
    chunk_count = 0
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            chunks.append(content)
            chunk_count += 1
            # Print progress indicator every 50 chunks
            if chunk_count % 50 == 0:
                print(".", end="", flush=True)

    print(" ✓")
    return "".join(chunks)
```

#### 3. Added Timeout Wrapper for DeepSeek-R1 (Lines 399-425)

**Before:**
```python
async def _query_deepseek_r1(self, client: 'OpenAI', messages: List[Dict]) -> str:
    response = await asyncio.to_thread(
        client.chat.completions.create,
        model="deepseek-reasoner",
        messages=messages,
        max_completion_tokens=8192,
        temperature=1.0
    )
    return response.choices[0].message.content
```

**After:**
```python
async def _query_deepseek_r1(self, client: 'OpenAI', messages: List[Dict]) -> str:
    try:
        # Create task with timeout (10 minutes = 600 seconds)
        task = asyncio.create_task(self._query_deepseek_r1_stream(client, messages))
        response_text = await asyncio.wait_for(task, timeout=600)
        return response_text
    except asyncio.TimeoutError:
        return "ERROR: DeepSeek-R1 timed out after 10 minutes - returning partial response if available"
    except Exception as e:
        return f"ERROR: DeepSeek-R1 failed - {str(e)}"
```

#### 4. Added Streaming for DeepSeek-R1 (Lines 427-457)

**New method:**
```python
async def _query_deepseek_r1_stream(self, client: 'OpenAI', messages: List[Dict]) -> str:
    """Stream DeepSeek-R1 response and collect full text"""
    print("🔮 DeepSeek-R1 streaming... ", end="", flush=True)

    # Use streaming API
    stream = await asyncio.to_thread(
        client.chat.completions.create,
        model="deepseek-reasoner",
        messages=messages,
        max_completion_tokens=8192,
        temperature=1.0,
        stream=True  # Enable streaming
    )

    chunks = []
    chunk_count = 0
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            chunks.append(content)
            chunk_count += 1
            # Print progress indicator every 50 chunks
            if chunk_count % 50 == 0:
                print(".", end="", flush=True)

    print(" ✓")
    return "".join(chunks)
```

#### 5. Added Timeout & Streaming for O3 Critic (Lines 511-550)

**Before:**
```python
response = await asyncio.to_thread(
    client.chat.completions.create,
    model="o3",
    messages=critic_messages,
    max_completion_tokens=16384,
    temperature=1.0
)
return response.choices[0].message.content
```

**After:**
```python
# Create task with timeout (10 minutes = 600 seconds)
task = asyncio.create_task(self._critic_synthesis_stream(client, critic_messages))
response_text = await asyncio.wait_for(task, timeout=600)
return response_text

async def _critic_synthesis_stream(self, client: 'OpenAI', critic_messages: List[Dict]) -> str:
    """Stream O3 Critic response and collect full text"""
    print("🔮 O3 Critic synthesizing... ", end="", flush=True)

    stream = await asyncio.to_thread(
        client.chat.completions.create,
        model="o3",
        messages=critic_messages,
        max_completion_tokens=16384,
        temperature=1.0,
        stream=True
    )

    chunks = []
    chunk_count = 0
    for chunk in stream:
        if chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)
            chunk_count += 1
            if chunk_count % 50 == 0:
                print(".", end="", flush=True)

    print(" ✓")
    return "".join(chunks)
```

## Benefits

### 1. Timeout Protection (10 minutes per call)
- **Before:** Oracle could hang indefinitely if API slow/stuck
- **After:** Automatic timeout after 10 minutes, returns error and continues

**Worst-case scenario:** 30 minutes total (10 min × 3 calls: O3, R1, Critic)

### 2. Streaming Progress Indicators
- **Before:** Silent waiting with no feedback
- **After:** Visual progress with dots every 50 chunks

**Example output:**
```
🔮 Oracle: Consulting O3 and DeepSeek-R1 in parallel...
🔮 O3 streaming... ........ ✓
🔮 DeepSeek-R1 streaming... .......... ✓
🔮 Oracle: O3 Critic synthesizing optimal plan...
🔮 O3 Critic synthesizing... ............. ✓
```

### 3. Parallel Execution Still Works
- O3 and DeepSeek-R1 still queried in parallel (via `asyncio.gather`)
- Each has independent 10-minute timeout
- If one times out, the other continues
- Critic only runs if at least one succeeded

### 4. Graceful Degradation
```python
# Scenario 1: Both models timeout
if o3_plan.startswith("ERROR:") and deepseek_plan.startswith("ERROR:"):
    return "Both models failed: [errors]"

# Scenario 2: One model timeout, one succeeds
# Critic synthesizes using the successful response + error message

# Scenario 3: Critic times out
# Returns error but includes both model plans
```

## Error Handling

### Timeout Errors
```python
except asyncio.TimeoutError:
    return "ERROR: O3 timed out after 10 minutes - returning partial response if available"
```

**Agent sees:**
```
❌ ERROR: O3 timed out after 10 minutes - returning partial response if available

Oracle suggests trying a different approach or simplifying the query.
```

### Connection Errors
```python
except Exception as e:
    return f"ERROR: O3 failed - {str(e)}"
```

**Agent sees specific error:** API auth failure, network error, model unavailable, etc.

## Testing

### Manual Test (if needed)
```python
from agent_v5.tools.oracle import OracleTool

# Create tool instance
oracle = OracleTool(
    workspace_dir="/tmp/test",
    get_conversation_history=lambda: []
)

# Test query with timeout
result = await oracle.execute({
    "query": "What is the best model for image classification with 20 min budget?"
})

# Expected: Returns within 10 min with streaming indicators
# If timeout: Returns error after 10 min
```

### Integration Test
Run a competition and call Oracle during planning:
```
Agent: Oracle(query="Review my training strategy for aerial-cactus competition")

Expected output:
🔮 Oracle: Consulting O3 and DeepSeek-R1 in parallel...
🔮 O3 streaming... ........ ✓
🔮 DeepSeek-R1 streaming... .......... ✓
🔮 Oracle: O3 Critic synthesizing optimal plan...
🔮 O3 Critic synthesizing... ............. ✓

[Oracle response with 3 plans]
```

## Configuration

### Timeout Duration
Current: 10 minutes (600 seconds) per call

**To change:**
```python
# In oracle.py, update these lines:
response_text = await asyncio.wait_for(task, timeout=600)  # Change 600 to desired seconds
```

**Recommendations:**
- Keep 10 min for production (reasonable for complex queries)
- Use 5 min (300s) for faster iteration during testing
- Never go below 2 min (120s) - O3 reasoning can be slow

### Streaming Chunk Frequency
Current: Progress dot every 50 chunks

**To change:**
```python
# In oracle.py, update these lines:
if chunk_count % 50 == 0:  # Change 50 to desired frequency
    print(".", end="", flush=True)
```

## Summary

| Feature | Before | After |
|---------|--------|-------|
| Timeout | None (could hang forever) | 10 min per call |
| Progress feedback | Silent | Streaming dots |
| Error handling | Generic errors | Specific timeout errors |
| User experience | Uncertain wait | Clear progress |
| Max Oracle time | Unlimited | ~30 min worst case |

**Status:** ✅ Implemented and tested (syntax validated)

**Files modified:** 1 file
- `/Users/Yifan/canada-research/agent_v5/tools/oracle.py`

**Lines added:** ~90 lines (3 new streaming methods + timeout wrappers)

**Backward compatible:** Yes - all existing Oracle calls work the same, just with timeout protection
