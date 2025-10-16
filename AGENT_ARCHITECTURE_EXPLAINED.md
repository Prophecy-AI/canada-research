# Agent V5 Architecture: Complete Deep Dive

## Table of Contents
1. [High-Level Architecture](#high-level-architecture)
2. [The Agentic Loop Explained](#the-agentic-loop-explained)
3. [Tool System Deep Dive](#tool-system-deep-dive)
4. [Memory & Context Management](#memory--context-management)
5. [Oracle Tool: Current vs Upgraded](#oracle-tool-current-vs-upgraded)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INPUT                               │
│                    "Train a gold-medal model"                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    KAGGLE AGENT (ResearchAgent)                  │
│  - Session ID: "competition-name"                                │
│  - Workspace: /path/to/workspace/                                │
│  - System Prompt: 258-line Kaggle-specific instructions          │
│  - Conversation History: List[Dict] (grows each turn)            │
│  - Anthropic Client: Claude Sonnet 4.5                           │
└───────────┬──────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AGENTIC LOOP (agent.py:67-144)                │
│  1. Add user message to conversation_history                     │
│  2. Call Claude API with:                                        │
│     - system_prompt                                              │
│     - conversation_history                                       │
│     - tools (from registry)                                      │
│     - temperature=0 (deterministic)                              │
│  3. Stream text deltas to user                                   │
│  4. If tool_uses exist: execute them via registry                │
│  5. Add assistant response + tool results to history             │
│  6. Loop back to step 2 until no more tool_uses                  │
└───────────┬──────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TOOL REGISTRY (registry.py)                   │
│  - tools: Dict[str, BaseTool] = {                                │
│      "Bash": BashTool instance,                                  │
│      "Read": ReadTool instance,                                  │
│      "Oracle": OracleTool instance,                              │
│      ...                                                          │
│    }                                                              │
│  - execute(tool_name, tool_input):                               │
│      1. Validate tool exists                                     │
│      2. Run tool.prehook(input) for validation                   │
│      3. Run tool.execute(input)                                  │
│      4. Return {content: str, is_error: bool}                    │
└───────────┬──────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TOOLS (Inherits BaseTool)                     │
│  Each tool implements:                                           │
│  - name: str (e.g., "Bash", "Read", "Oracle")                    │
│  - schema: Dict (Anthropic tool schema for Claude)               │
│  - prehook(input): validation/normalization                      │
│  - execute(input): actual implementation                         │
│                                                                   │
│  Available Tools:                                                │
│  - BashTool: Execute shell commands (fg/bg modes)                │
│  - ReadTool: Read file contents                                  │
│  - WriteTool: Create new files                                   │
│  - EditTool: Modify existing files                               │
│  - GlobTool: Find files by pattern                               │
│  - GrepTool: Search file contents                                │
│  - TodoWriteTool: Task management                                │
│  - OracleTool: Consult O3 for strategic guidance                 │
│  - RunSummaryTool: Log experiment results                        │
│  - ReadBashOutputTool: Monitor background processes              │
│  - KillShellTool: Terminate background processes                 │
│  - ListBashProcessesTool: List running processes                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Agentic Loop Explained

### Flow Diagram

```
START: User sends "Train a model"
│
├─> ADD TO HISTORY:
│   conversation_history.append({
│     "role": "user",
│     "content": "Train a model"
│   })
│
├─> CALL CLAUDE API (agent.py:82-101):
│   │
│   ├─> Stream Response:
│   │   ├─> Text: "Let me check the data first..."
│   │   │   └─> Yield {"type": "text_delta", "text": "..."}
│   │   │
│   │   └─> Tool Uses: [
│   │         {type: "tool_use", name: "Read", input: {file_path: "data/train.csv"}},
│   │         {type: "tool_use", name: "Bash", input: {command: "head data/train.csv", background: false}}
│   │       ]
│   │
│   └─> final_message.content = [text blocks, tool_use blocks]
│
├─> ADD ASSISTANT RESPONSE TO HISTORY:
│   conversation_history.append({
│     "role": "assistant",
│     "content": [
│       {"type": "text", "text": "Let me check the data first..."},
│       {"type": "tool_use", "id": "toolu_123", "name": "Read", "input": {...}},
│       {"type": "tool_use", "id": "toolu_456", "name": "Bash", "input": {...}}
│     ]
│   })
│
├─> EXECUTE TOOLS (agent.py:122-137):
│   │
│   ├─> For each tool_use:
│   │   ├─> registry.execute(tool_name, tool_input)
│   │   │   ├─> Log: "→ Read(data/train.csv)"
│   │   │   ├─> Validate: tool.prehook(input)
│   │   │   ├─> Execute: tool.execute(input)
│   │   │   └─> Log: "✓ Read 1000 lines, 50KB"
│   │   │
│   │   └─> Collect result: {content: "...", is_error: false}
│   │
│   └─> Build tool_results: [
│         {"type": "tool_result", "tool_use_id": "toolu_123", "content": "...", "is_error": false},
│         {"type": "tool_result", "tool_use_id": "toolu_456", "content": "...", "is_error": false}
│       ]
│
├─> ADD TOOL RESULTS TO HISTORY:
│   conversation_history.append({
│     "role": "user",
│     "content": [
│       {"type": "tool_result", "tool_use_id": "toolu_123", "content": "...", "is_error": false},
│       {"type": "tool_result", "tool_use_id": "toolu_456", "content": "...", "is_error": false}
│     ]
│   })
│
└─> LOOP: Go back to "CALL CLAUDE API" with updated history
    │
    ├─> Claude sees tool results, reasons about them
    ├─> Either: Use more tools (loop continues)
    └─> Or: Provide final response (no tool_uses → break loop)

END: Yield {"type": "done"}
```

### Key Mechanisms

#### 1. Streaming (agent.py:82-101)
```python
with self.anthropic_client.messages.stream(...) as stream:
    for event in stream:
        if event.type == "content_block_delta":
            # Real-time text streaming to user
            yield {"type": "text_delta", "text": event.delta.text}
```

**Why?** User sees progress in real-time instead of waiting for full response.

#### 2. Tool Execution (agent.py:122-137)
```python
for tool_use in tool_uses:
    result = await self.tools.execute(tool_use["name"], tool_use["input"])
    tool_results.append({
        "type": "tool_result",
        "tool_use_id": tool_use["id"],
        "content": result["content"],
        "is_error": result.get("is_error", False)
    })
```

**Why?** Claude decides what tools to use, agent executes them, results feed back into next iteration.

#### 3. Conversation History Management (agent.py:71-74, 111-114, 139-142)
```python
# Turn structure:
# 1. User message
conversation_history.append({"role": "user", "content": "..."})

# 2. Assistant response (text + tool_uses)
conversation_history.append({"role": "assistant", "content": [...]})

# 3. Tool results (as "user" message for Claude API format)
conversation_history.append({"role": "user", "content": [tool_results]})
```

**Why?** Claude needs full context to reason across turns. Every tool use and result is preserved.

---

## Tool System Deep Dive

### BaseTool Architecture (tools/base.py)

All tools inherit from `BaseTool` abstract class:

```python
class BaseTool(ABC):
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self._custom_prehook = None  # For validation injection

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool identifier (must be unique)"""
        pass

    @property
    @abstractmethod
    def schema(self) -> Dict:
        """Claude-compatible schema describing tool usage"""
        return {
            "name": "MyTool",
            "description": "What this tool does (Claude reads this)",
            "input_schema": {
                "type": "object",
                "properties": {
                    "param": {"type": "string", "description": "..."}
                },
                "required": ["param"]
            }
        }

    async def prehook(self, input: Dict) -> None:
        """Run validation/normalization before execute()"""
        if self._custom_prehook:
            await self._custom_prehook(input)

    @abstractmethod
    async def execute(self, input: Dict) -> Dict:
        """Do the actual work"""
        return {"content": "result", "is_error": False}
```

### Example: BashTool Deep Dive (tools/bash.py)

BashTool has **two execution modes**:

#### Mode 1: Foreground (Blocking)
```python
input = {"command": "ls -la", "background": false, "timeout": 120000}

# Execution flow:
1. Wrap command with 'script' for PTY allocation:
   wrapped_cmd = 'script -q -c "ls -la" "/workspace/.pty_logs/bash_abc123.typescript"'

2. Create subprocess with PYTHONUNBUFFERED=1 (force flush):
   process = await asyncio.create_subprocess_shell(
       wrapped_cmd,
       stdout=PIPE,
       stderr=PIPE,
       cwd=workspace_dir,
       env={"PYTHONUNBUFFERED": "1"}
   )

3. Wait with timeout:
   stdout, stderr = await asyncio.wait_for(
       process.communicate(),
       timeout=120.0  # seconds
   )

4. Return combined output:
   return {
       "content": stdout + stderr,
       "is_error": False,
       "debug_summary": f"exit {returncode}: {output[:200]}"
   }
```

**When to use:** Quick commands (<2 min): `ls`, `head`, `wc`, `nvidia-smi`

#### Mode 2: Background (Non-blocking)
```python
input = {"command": "python train.py", "background": true}

# Execution flow:
1. Generate unique shell_id: "bash_abc12345"

2. Start process (same subprocess as foreground)

3. Register in BashProcessRegistry:
   bg_process = BackgroundProcess(
       process=process,
       command="python train.py",
       start_time=time.time(),
       stdout_buffer=b"",
       stderr_buffer=b""
   )
   registry.register(shell_id, bg_process)

4. Start output collector task (runs in background):
   asyncio.create_task(_collect_output(shell_id))

5. Return IMMEDIATELY:
   return {
       "content": "Started background process: bash_abc12345\n"
                  "Use ReadBashOutput(shell_id='bash_abc12345') to monitor.",
       "is_error": False
   }
```

**Output Collector** (runs concurrently):
```python
async def _collect_output(shell_id):
    # Non-blocking incremental reads
    while process.returncode is None:
        chunk = await process.stdout.read(1024)  # Read 1KB at a time
        if chunk:
            bg_process.append_stdout(chunk)  # Accumulate in buffer
        await asyncio.sleep(0.05)  # Brief pause

    # Drain remaining output after process exits
    remaining = await process.stdout.read()
    bg_process.append_stdout(remaining)
```

**Monitoring with ReadBashOutput:**
```python
# Agent calls this every 30s during training
input = {"shell_id": "bash_abc12345"}

# Returns:
# - Latest output since last read (incremental)
# - Process status: RUNNING, COMPLETED, FAILED
# - Runtime duration

# Example output:
"Epoch 5/100 - loss: 0.234 - val_loss: 0.456
 Epoch 6/100 - loss: 0.228 - val_loss: 0.449

 [Process still RUNNING - 45s elapsed]"
```

**When to use:** Long tasks (>2 min): training, inference, large data processing

### Example: ReadTool Deep Dive (tools/read.py)

Simpler tool with **pagination support**:

```python
input = {"file_path": "/workspace/train.csv", "offset": 0, "limit": 2000}

# Execution flow:
1. Normalize path (relative → absolute):
   if not file_path.startswith('/'):
       file_path = os.path.join(workspace_dir, file_path)

2. Read file with pagination:
   with open(file_path, 'r') as f:
       lines = f.readlines()
   selected_lines = lines[offset:offset+limit]

3. Add line numbers (cat -n style):
   numbered_lines = []
   for i, line in enumerate(selected_lines, start=offset+1):
       numbered_lines.append(f"{i:6d}→{line}")

4. Return:
   return {
       "content": "\n".join(numbered_lines),
       "is_error": False,
       "debug_summary": f"{len(selected_lines)} lines, {total_bytes} bytes"
   }
```

**Why pagination?**
- Large files (100K+ lines) overflow context window
- Agent can read first 2000 lines, then request more if needed

### Example: OracleTool Current Implementation (tools/oracle.py)

**Strategic reasoning tool** using OpenAI O3:

```python
input = {"query": "Why is my CV 0.44 but leaderboard score 0.38?"}

# Execution flow:
1. Get full conversation history:
   history = self.get_conversation_history()  # Callable passed in __init__

2. Convert to O3 format:
   messages = [
       {"role": "system", "content": "You are an expert ML engineer Oracle..."},
       # Add all conversation turns (text + tool uses + tool results)
       {"role": "user", "content": "..."},
       {"role": "assistant", "content": "..."},
       ...
       {"role": "user", "content": f"[ORACLE QUERY]: {query}"}
   ]

3. Call O3 with extended thinking:
   response = openai_client.chat.completions.create(
       model="o3",
       messages=messages,
       max_completion_tokens=8192,  # Allows deep reasoning
       temperature=1.0
   )

4. Return analysis:
   return {
       "content": f"🔮 Oracle Analysis:\n\n{response.content}",
       "is_error": False
   }
```

**Why O3?**
- Extended reasoning (finds subtle bugs)
- Analyzes full conversation context
- Identifies patterns agent missed
- Strategic pivots when stuck

---

## Memory & Context Management

### 1. Conversation History Structure

```python
self.conversation_history: List[Dict] = [
    # Turn 1: User starts
    {
        "role": "user",
        "content": "Train a gold-medal model for this competition"
    },

    # Turn 1: Assistant responds with text + tool uses
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me explore the data first..."},
            {"type": "tool_use", "id": "toolu_001", "name": "Read", "input": {...}},
            {"type": "tool_use", "id": "toolu_002", "name": "Bash", "input": {...}}
        ]
    },

    # Turn 1: Tool results (formatted as "user" for Claude API)
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_001", "content": "...", "is_error": false},
            {"type": "tool_result", "tool_use_id": "toolu_002", "content": "...", "is_error": false}
        ]
    },

    # Turn 2: Assistant continues reasoning
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "I see we have 100K rows. Let me consult Oracle..."},
            {"type": "tool_use", "id": "toolu_003", "name": "Oracle", "input": {"query": "..."}}
        ]
    },

    # Turn 2: Oracle result
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_003", "content": "🔮 Oracle Analysis:\n...", "is_error": false}
        ]
    },

    # ... continues for entire session
]
```

### 2. Context Window Management

**Problem:** Conversation history grows unbounded → overflow context window (200K tokens)

**Current strategy:** No truncation (relies on session ending before overflow)

**Future strategies:**
- **Summarization**: Compress old turns into summaries
- **Sliding window**: Keep only last N turns + critical turns (Oracle consultations, RunSummary logs)
- **Hierarchical**: Move old turns to long-term memory, retrieve on-demand

### 3. Memory Persistence

**Workspace as persistent memory:**

```
/workspace/
├── .pty_logs/              # Background process logs
│   └── bash_abc123.typescript
├── .run_summary.jsonl      # Experiment log (RunSummaryTool)
├── .todos.json             # Task list (TodoWriteTool)
├── train.py                # Generated training script
├── predict.py              # Generated inference script
├── data/                   # Competition data (read-only)
│   ├── train.csv
│   └── test.csv
└── submissions/            # Model predictions
    └── submission.csv
```

**Why this matters:**
- Agent can reference past experiments via `.run_summary.jsonl`
- Tasks persist across interruptions via `.todos.json`
- Background processes survive agent crashes (can reconnect via shell_id)

### 4. Context Injection

**System Prompt** (258 lines in kaggle_agent.py:19-215):
- Competition instructions (verbatim)
- Available tools (descriptions + usage patterns)
- R&D loop guardrails (11-step workflow)
- GPU optimization rules
- Oracle usage guidelines

**Dynamic context** (via tools):
- Data exploration results (Read tool)
- Past experiment results (RunSummaryTool)
- Current task status (ReadTodoList)
- Background process status (ReadBashOutput)

**Oracle's full context access** (oracle.py:73-176):
- Entire conversation history passed to O3
- Includes all tool uses + results
- O3 can identify patterns across 50+ turns
- Self-awareness: Oracle checks if its own prior advice failed

---

## Oracle Tool: Current vs Upgraded

### Current Implementation (Single O3 Call)

```
User → Agent → OracleTool → O3 (8K tokens) → Response → Agent → User
```

**Limitations:**
- Single perspective (only O3)
- No model comparison
- No self-critique
- Fixed token budget

### Upgraded Implementation (Multi-Model Ensemble + Critic)

```
User → Agent → OracleTool
              ↓
         ┌────┴────┐
         ↓         ↓
     O3 (8K)   DeepSeek-R1 (max thinking)
         ↓         ↓
    Plan A    Plan B
         ↓         ↓
         └────┬────┘
              ↓
         O3 Critic (16K tokens)
         - Compare Plan A vs Plan B
         - Identify strengths/weaknesses
         - Synthesize best elements
         - Return unified optimal plan
              ↓
         Agent → User
```

**Benefits:**
1. **Diverse perspectives**: O3 (precise) + DeepSeek-R1 (exploratory)
2. **Cross-validation**: Two models → catch each other's blind spots
3. **Self-critique**: Final O3 call evaluates both plans
4. **Higher quality**: Best of both → better than either alone

---

## Next: Implementing Upgraded Oracle

I'll now implement the multi-model Oracle with parallel consultation + critic synthesis.
