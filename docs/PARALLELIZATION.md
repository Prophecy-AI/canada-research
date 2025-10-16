# Async Parallelization in Agent V5

## Overview

Agent V5 uses `asyncio.gather` to execute independent tools concurrently, significantly improving performance when the LLM requests multiple read-only operations.

## Key Features

### 1. **Automatic Parallel Tool Execution**

When the agent decides to use multiple tools in a single turn, the system automatically determines if they can be parallelized:

**Parallelizable Tools (Read-Only):**
- `Read` - Read file contents
- `Glob` - Find files by pattern
- `Grep` - Search file contents
- `CohortDefinition` - Define cohorts
- `StatisticalValidation` - Statistical operations
- MCP tools (if read-only)

**Sequential Tools (Have Side Effects or Dependencies):**
- `Write` - Create/overwrite files
- `Edit` - Modify files
- `Bash` - Execute shell commands
- `ReadBashOutput` - Monitor background processes
- `KillShell` - Terminate processes
- `TodoWrite` - Update task list

### 2. **Implementation Details**

#### Agent Loop (`agent.py`)

```python
# Automatic parallelization decision
can_parallelize = self._can_parallelize_tools(tool_uses)

if can_parallelize and len(tool_uses) > 1:
    log(f"→ Parallelizing {len(tool_uses)} independent tools")

    # Execute all tools concurrently
    results = await asyncio.gather(
        *[self.tools.execute(tool_use["name"], tool_use["input"])
          for tool_use in tool_uses],
        return_exceptions=True
    )
else:
    # Sequential execution for tools with dependencies
    for tool_use in tool_uses:
        result = await self.tools.execute(tool_use["name"], tool_use["input"])
```

#### Parallelization Logic (`_can_parallelize_tools`)

```python
def _can_parallelize_tools(self, tool_uses: List[Dict]) -> bool:
    """
    Determine if tools can be executed in parallel.

    Returns False if:
    - Any Write/Edit operations (may affect Read operations)
    - Multiple Bash commands (may have dependencies)
    - ReadBashOutput (needs sequential monitoring)
    - KillShell (affects other processes)

    Returns True for:
    - Multiple Read operations
    - Multiple Glob/Grep operations
    - Mix of read-only MCP tools
    """
    tool_names = [tool["name"] for tool in tool_uses]

    sequential_tools = {
        "Write", "Edit", "Bash", "ReadBashOutput", "KillShell", "TodoWrite"
    }

    if any(name in sequential_tools for name in tool_names):
        return False

    return True
```

### 3. **Batch Evaluations**

Evaluators can run in parallel using the `submit_batch` method:

```python
# In evals_v5/runner.py
runner = EvalRunner(session_id, workspace_dir)

# Run multiple evaluations concurrently
runner.submit_batch([
    ("hallucination", {"answer": answer, "data": source_data}),
    ("sql", {"sql": query, "context": context}),
    ("answer", {"question": question, "answer": answer})
])
```

This uses `asyncio.gather` internally:

```python
async def _run_batch_evals(self, eval_requests: list[tuple[EvalType, Dict]]):
    results = await asyncio.gather(
        *[self._run_eval(eval_type, data)
          for eval_type, data in eval_requests],
        return_exceptions=True
    )
```

## Performance Benefits

### Example Scenarios

**Scenario 1: Reading Multiple Files**

```
User: "Read config.json, data.csv, and README.md"

Sequential (before):
  Read config.json   (200ms)
  Read data.csv      (150ms)
  Read README.md     (100ms)
  Total: 450ms

Parallel (after):
  Read config.json  ┐
  Read data.csv     ├─ (200ms)
  Read README.md    ┘
  Total: 200ms

Speedup: 2.25x
```

**Scenario 2: Search Operations**

```
User: "Find all Python files and search for 'async def'"

Sequential (before):
  Glob *.py          (300ms)
  Grep "async def"   (500ms)
  Total: 800ms

Parallel (after):
  Glob *.py        ┐
  Grep "async def" ┘ (500ms)
  Total: 500ms

Speedup: 1.6x
```

**Scenario 3: Multiple Evaluations**

```
After agent completes SQL query:

Sequential (before):
  SQL evaluator           (2000ms)
  Hallucination evaluator (1800ms)
  Answer evaluator        (1500ms)
  Total: 5300ms

Parallel (after):
  SQL evaluator          ┐
  Hallucination eval     ├─ (2000ms)
  Answer evaluator       ┘
  Total: 2000ms

Speedup: 2.65x
```

## Safety Guarantees

### 1. **Exception Handling**

`asyncio.gather` with `return_exceptions=True` ensures one failing tool doesn't crash the entire batch:

```python
results = await asyncio.gather(
    *[self.tools.execute(...) for ...],
    return_exceptions=True
)

for tool_use, result in zip(tool_uses, results):
    if isinstance(result, Exception):
        result = {
            "content": f"Tool execution error: {str(result)}",
            "is_error": True
        }
```

### 2. **Race Condition Prevention**

Tools with side effects (Write, Edit, Bash) are never parallelized to prevent race conditions:

```python
# This will execute sequentially
[
    {"name": "Write", "input": {"file_path": "output.txt"}},
    {"name": "Read", "input": {"file_path": "output.txt"}}
]
```

### 3. **Workspace Isolation**

Each agent has an isolated workspace directory, so parallel tool execution across different sessions is safe.

## Testing

Comprehensive tests in `agent_v5/tests/test_parallel_execution.py`:

1. **test_can_parallelize_tools** - Logic validation
2. **test_parallel_read_tools** - Real API test with multiple Read operations
3. **test_parallel_glob_and_grep** - Real API test with mixed tools
4. **test_sequential_write_then_read** - Ensures Write→Read is sequential

Run tests:
```bash
pytest agent_v5/tests/test_parallel_execution.py -v
```

## Debugging

Enable debug logging to see parallelization decisions:

```bash
DEBUG=1 python your_script.py
```

Output:
```
[12:34:56] → Executing 3 tools
[12:34:56] → Parallelizing 3 independent tools
[12:34:56] → Read(config.json)
[12:34:56] → Read(data.csv)
[12:34:56] → Read(README.md)
[12:34:57] ✓ Read ok (200 lines)
[12:34:57] ✓ Read ok (1500 rows)
[12:34:57] ✓ Read ok (50 lines)
```

## Future Improvements

1. **Smarter Dependency Detection**
   - Analyze file paths to allow parallel Read/Write to different files
   - Example: `Write(a.txt)` + `Read(b.txt)` could be parallel

2. **Resource Limits**
   - Limit concurrent operations (e.g., max 5 parallel tools)
   - Prevent resource exhaustion

3. **Cost Optimization**
   - Track token savings from reduced LLM calls
   - Benchmark latency improvements

4. **MCP-Specific Rules**
   - Allow MCP servers to declare if tools can be parallelized
   - Support read-only vs. write operations in MCP schema

## API Changes

### Breaking Changes
None - parallelization is automatic and backward compatible.

### New APIs

1. **EvalRunner.submit_batch()**
   ```python
   runner.submit_batch([
       ("eval_type1", data1),
       ("eval_type2", data2)
   ])
   ```

2. **ResearchAgent._can_parallelize_tools()** (internal)
   - Used by agent loop to determine parallelization
   - Can be overridden in subclasses for custom logic

## Summary

✅ **Automatic** - No code changes needed for existing agents
✅ **Safe** - Conservative parallelization rules prevent race conditions
✅ **Fast** - 1.5-3x speedup for read-heavy workloads
✅ **Tested** - Comprehensive test coverage with real API calls
✅ **Observable** - Debug logging shows parallelization decisions

Agent V5 now leverages `asyncio.gather` to maximize performance while maintaining safety and correctness.
