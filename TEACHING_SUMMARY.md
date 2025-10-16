# Agent V5 Framework: Teaching Summary

> **Complete explanation of how the autonomous agent framework works, tool system, memory management, and Oracle upgrade**

---

## 📚 What You Learned

### 1. **Agent Architecture** (3-Layer Design)

```
┌─────────────────────────────────────────────┐
│         Layer 1: Interface Layer             │
│  (CLI, API, Web UI - handles user I/O)      │
└─────────────────┬────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────┐
│         Layer 2: Agent Layer                 │
│  (Agentic loop, tool orchestration)          │
│  - ResearchAgent (agent.py)                  │
│  - KaggleAgent (kaggle_agent.py)             │
└─────────────────┬────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────┐
│         Layer 3: Tool Layer                  │
│  (Individual capabilities)                   │
│  - BashTool, ReadTool, OracleTool, etc.      │
└──────────────────────────────────────────────┘
```

**Key insight**: Separation of concerns - each layer has clear responsibilities.

---

## 🔄 The Agentic Loop (How It Works)

### Step-by-Step Breakdown

**Located in**: `agent_v5/agent.py:67-144`

```python
async def run(self, user_message: str):
    # Step 1: Add user message to history
    self.conversation_history.append({
        "role": "user",
        "content": user_message
    })

    while True:  # Main loop
        # Step 2: Call Claude with full context
        response = anthropic_client.messages.stream(
            system=self.system_prompt,        # Competition instructions
            messages=self.conversation_history, # All past turns
            tools=self.tools.get_schemas(),   # Available tools
            temperature=0                     # Deterministic
        )

        # Step 3: Stream text to user (real-time)
        for event in response:
            if event.type == "text_delta":
                yield {"type": "text_delta", "text": event.delta.text}

        # Step 4: Check if Claude wants to use tools
        tool_uses = response.tool_uses
        if not tool_uses:
            break  # Done! No more tools needed

        # Step 5: Execute each tool
        tool_results = []
        for tool_use in tool_uses:
            result = await self.tools.execute(
                tool_use.name,    # e.g., "Bash"
                tool_use.input    # e.g., {"command": "ls", "background": false}
            )
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result["content"],
                "is_error": result.get("is_error", False)
            })

        # Step 6: Add tool results to history
        self.conversation_history.append({
            "role": "user",
            "content": tool_results
        })

        # Step 7: Loop back to Step 2
        # Claude sees tool results → decides next action
```

### Why This Works

**Multi-step reasoning**: Claude can:
1. Read a file
2. Analyze its contents
3. Write a script based on analysis
4. Execute script
5. See results
6. Decide next steps

**Error recovery**: If tool fails:
1. Claude sees error in tool result
2. Reasons about what went wrong
3. Tries different approach

**Adaptive**: No hardcoded workflow - Claude decides dynamically based on context

---

## 🛠️ Tool System Deep Dive

### BaseTool Abstract Class

**Every tool inherits from this** (`agent_v5/tools/base.py`):

```python
class BaseTool(ABC):
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self._custom_prehook = None  # For validation

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier (e.g., "Bash", "Read")"""
        pass

    @property
    @abstractmethod
    def schema(self) -> Dict:
        """
        Anthropic API schema - tells Claude:
        - What this tool does
        - What parameters it takes
        - When to use it
        """
        pass

    async def prehook(self, input: Dict) -> None:
        """
        Validation before execution
        - Path validation (security)
        - Input normalization
        - Schema checking
        """
        if self._custom_prehook:
            await self._custom_prehook(input)

    @abstractmethod
    async def execute(self, input: Dict) -> Dict:
        """
        Do the actual work
        Returns: {"content": str, "is_error": bool}
        """
        pass
```

### Tool Categories

#### 1. **Execution Tools** (Run code/commands)

**BashTool** - Two modes:

```python
# Mode 1: Foreground (blocking, max 2 min)
input = {"command": "ls -la", "background": false}
# Waits until completion, returns full output

# Mode 2: Background (non-blocking, no timeout)
input = {"command": "python train.py", "background": true}
# Returns shell_id immediately
# Monitor with ReadBashOutput(shell_id)
```

**Why two modes?**
- Quick commands: foreground (get result immediately)
- Long tasks (training): background (don't block agent)

**How background works:**
1. Start subprocess
2. Register in `BashProcessRegistry`
3. Launch async output collector
4. Return shell_id to agent
5. Agent polls with `ReadBashOutput` every 30s

#### 2. **File Tools** (Read/Write/Edit)

**ReadTool** - Pagination support:

```python
# Read first 2000 lines
{"file_path": "/workspace/train.csv", "offset": 0, "limit": 2000}

# Read next 2000 lines
{"file_path": "/workspace/train.csv", "offset": 2000, "limit": 2000}
```

**Why pagination?** Large files (100K+ lines) overflow context window

**WriteTool** - Create new files:
```python
{"file_path": "/workspace/train.py", "content": "import pandas as pd\n..."}
```

**EditTool** - Modify existing files (exact string replacement):
```python
{
    "file_path": "/workspace/train.py",
    "old_string": "batch_size = 32",
    "new_string": "batch_size = 256"
}
```

#### 3. **Search Tools** (Find files/content)

**GlobTool** - Find files by pattern:
```python
{"pattern": "**/*.py"}  # All Python files
{"pattern": "data/*.csv"}  # CSV files in data/
```

**GrepTool** - Search file contents:
```python
{
    "pattern": "def train",
    "path": "/workspace",
    "output_mode": "content"  # Show matching lines
}
```

#### 4. **Meta Tools** (Planning/Strategy)

**TodoWriteTool** - Task management:
```python
{
    "todos": [
        {"content": "Explore data", "status": "completed", "activeForm": "Exploring data"},
        {"content": "Train baseline", "status": "in_progress", "activeForm": "Training baseline"},
        {"content": "Optimize hyperparameters", "status": "pending", "activeForm": "Optimizing..."}
    ]
}
```

**RunSummaryTool** - Experiment logging:
```python
{
    "run_id": "exp_001",
    "phase": "train",
    "model": "XGBoost",
    "hyperparameters": {"n_estimators": 1000, "max_depth": 8},
    "metrics": {"cv_score": 0.8234, "train_time": 45.2},
    "notes": "Baseline with default params"
}
```

Saved to `/workspace/.run_summary.jsonl` - agent can review past experiments

**OracleTool** - Strategic reasoning (NOW UPGRADED):
```python
{"query": "Why is my CV 0.44 but leaderboard 0.38?"}

# Returns multi-model analysis:
# - O3's diagnosis
# - DeepSeek-R1's alternative explanation
# - O3 Critic's synthesized optimal plan
```

---

## 🧠 Memory & Context Management

### Conversation History Structure

```python
self.conversation_history = [
    # Turn 1
    {"role": "user", "content": "Train a model"},
    {"role": "assistant", "content": [
        {"type": "text", "text": "Let me explore the data..."},
        {"type": "tool_use", "name": "Read", "input": {...}}
    ]},
    {"role": "user", "content": [
        {"type": "tool_result", "content": "...", "is_error": false}
    ]},

    # Turn 2
    {"role": "assistant", "content": [
        {"type": "text", "text": "I see we have 100K rows..."},
        {"type": "tool_use", "name": "Oracle", "input": {...}}
    ]},
    {"role": "user", "content": [
        {"type": "tool_result", "content": "Oracle analysis...", "is_error": false}
    ]},

    # ... continues for entire session
]
```

### Why This Format?

**Claude API requirement**:
- Must alternate: user → assistant → user → assistant
- Tool results are "user" messages (agent providing info to Claude)
- Tool uses are part of "assistant" message (Claude requesting actions)

### Context Window Management

**Current**: No truncation (relies on session ending before 200K token limit)

**Problem**: Long sessions → overflow

**Solutions** (not yet implemented):
1. **Summarization**: Compress old turns
2. **Sliding window**: Keep last N turns
3. **Importance sampling**: Keep critical turns (Oracle consultations, errors)

### Workspace as Persistent Memory

```
/workspace/session_abc123/
├── .pty_logs/              # Background process outputs
├── .run_summary.jsonl      # Experiment log (searchable)
├── .todos.json             # Task list (persists across turns)
├── train.py                # Generated code
├── predict.py              # Generated code
├── results/                # Outputs
│   ├── model.pkl
│   └── submission.csv
```

**Benefits**:
- Agent can review past experiments
- Tasks survive interruptions
- Outputs accessible after session
- Debugging via file inspection

---

## 🔮 Oracle Upgrade Explained

### Before: Single-Model

```
Agent: "Why is my CV/LB mismatched?"
   ↓
Oracle: [calls O3]
   ↓
O3: "Probably label encoding bug"
   ↓
Agent: Fixes bug
```

**Risk**: O3 might be wrong or miss something

### After: Multi-Model Ensemble + Critic

```
Agent: "Why is my CV/LB mismatched?"
   ↓
Oracle: [calls O3 and DeepSeek-R1 in parallel]
   ↓                               ↓
O3: "Label encoding bug         DeepSeek-R1: "Could be data
     in column order"                        leakage or metric
                                             calculation error"
   ↓                               ↓
   └───────────┬───────────────────┘
               ↓
    O3 Critic: "After analyzing both:
                - Plan A (label encoding) is most likely
                - Plan B (leakage check) is good validation
                - Unified approach: fix encoding + add validation"
               ↓
           Agent: Executes synthesized plan
```

**Benefits**:
1. **Cross-validation**: Two models check each other
2. **Diverse perspectives**: Catch blind spots
3. **Self-critique**: Critic validates both approaches
4. **Higher quality**: Best of both worlds

### Implementation Flow

```python
async def execute(self, input: Dict):
    # 1. Build messages from conversation history
    messages = self._build_messages(conversation_history, query)

    # 2. Query both models in parallel
    o3_plan, deepseek_plan = await asyncio.gather(
        self._query_o3(client, messages),
        self._query_deepseek_r1(client, messages)
    )

    # 3. O3 Critic synthesizes
    final_plan = await self._critic_synthesis(
        client, messages, o3_plan, deepseek_plan, query
    )

    # 4. Format response with all three outputs
    return {
        "content": self._format_response(o3_plan, deepseek_plan, final_plan),
        "is_error": False
    }
```

### Token Budget

**Phase 1 (Parallel)**:
- O3: 8K tokens (thinking)
- DeepSeek-R1: 8K tokens (thinking)
- Total: 16K tokens

**Phase 2 (Critic)**:
- O3 Critic: 16K tokens (synthesis)

**Total**: 32K tokens vs 8K before (4x more reasoning)

---

## 🎯 How Agent Uses Tools (Example Workflow)

### Scenario: Train gold-medal Kaggle model

**Turn 1: System check** (mandatory first step)

```
Agent: "Let me check available compute"
Tool: Bash(command="nproc", background=false)
Result: "8"
Tool: Bash(command="nvidia-smi --query-gpu=name,memory.total --format=csv", background=false)
Result: "NVIDIA A10, 24576 MiB"
Agent: "We have 8 CPU cores, A10 GPU with 24GB VRAM"
```

**Turn 2: Data exploration**

```
Agent: "Let me check the data"
Tool: Read(file_path="/data/train.csv", limit=100)
Result: [shows first 100 rows]
Tool: Bash(command="wc -l /data/train.csv", background=false)
Result: "100000 /data/train.csv"
Agent: "100K rows, 50 features, binary classification"
```

**Turn 3: Oracle strategic planning** (MANDATORY after exploration)

```
Agent: "I need a gold-medal strategy"
Tool: Oracle(query="Competition: [name]. Data: 100K rows, 50 features, binary classification, imbalanced (90/10). Metric: AUC. Resources: 8 cores, A10 GPU. What's the optimal gold-medal strategy?")

Oracle executes:
├─> O3 analyzes → "Use LightGBM with GPU, focus on feature engineering"
├─> DeepSeek-R1 analyzes → "Try XGBoost + neural network ensemble"
└─> O3 Critic synthesizes → "Unified plan: Start with LightGBM (faster iteration), then ensemble with NN if time permits. Key: handle imbalance with focal loss."

Agent receives synthesized plan
```

**Turn 4: Create task list**

```
Agent: "Breaking down the strategy"
Tool: TodoWrite(todos=[
    {"content": "Write LightGBM training script", "status": "in_progress", ...},
    {"content": "Train baseline model", "status": "pending", ...},
    {"content": "Add feature engineering", "status": "pending", ...},
    ...
])
```

**Turn 5: Write training script**

```
Agent: "Writing LightGBM script with GPU"
Tool: Write(file_path="/workspace/train.py", content="import lightgbm as lgb\n...")
Agent: "Script created. Let me consult Oracle for code review"
Tool: Oracle(query="Review this training script for bugs before I run it: [code]")
Oracle: [both models review, critic synthesizes feedback]
Agent: "Oracle found issue with eval_metric - fixing"
Tool: Edit(file_path="/workspace/train.py", old_string="metric='auc'", new_string="metric='binary_logloss'")
```

**Turn 6: Launch training (background)**

```
Agent: "Launching training in background"
Tool: Bash(command="python -u train.py", background=true)
Result: "Started background process: bash_abc123"
Agent: "Training started. I'll monitor it"
```

**Turn 7-N: Monitor training** (every 30s)

```
Agent: "Checking training progress"
Tool: ReadBashOutput(shell_id="bash_abc123")
Result: "Epoch 50/100 - train_auc: 0.85 - val_auc: 0.82\n[RUNNING - 45s elapsed]"
Agent: "Training progressing well. Waiting..."
```

**Turn N+1: Training complete**

```
Tool: ReadBashOutput(shell_id="bash_abc123")
Result: "Epoch 100/100 - train_auc: 0.92 - val_auc: 0.84\n[COMPLETED - 120s elapsed]"
Agent: "Training done! CV AUC: 0.84"
Tool: RunSummary(run_id="baseline_lgb", metrics={"cv_auc": 0.84}, ...)
```

**Turn N+2: Oracle post-analysis**

```
Agent: "Reviewing results with Oracle"
Tool: Oracle(query="Completed baseline: CV AUC 0.84. Should I continue tuning or pivot?")
Oracle: [analyzes full conversation, synthesizes recommendation]
Agent: Executes Oracle's next steps
```

---

## 🔑 Key Design Patterns

### 1. **Registry Pattern** (Tool Management)

```python
class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self.tools[tool.name] = tool

    async def execute(self, tool_name: str, input: Dict):
        tool = self.tools[tool_name]
        await tool.prehook(input)  # Validate
        return await tool.execute(input)  # Execute
```

**Benefits**:
- Centralized tool management
- Easy to add new tools
- Validation at single point
- Debug logging at registry level

### 2. **Prehook Pattern** (Validation)

```python
class ReadTool(BaseTool):
    async def prehook(self, input: Dict):
        # Security: validate path is within workspace
        file_path = input["file_path"]
        if not file_path.startswith(self.workspace_dir):
            raise SecurityError("Path outside workspace")

    async def execute(self, input: Dict):
        # Path already validated - safe to read
        with open(input["file_path"]) as f:
            return {"content": f.read(), "is_error": False}
```

**Benefits**:
- Security checks before execution
- Input normalization (e.g., relative → absolute paths)
- Consistent error handling
- Reusable validation logic

### 3. **Async Generator Pattern** (Streaming)

```python
async def run(self, user_message: str) -> AsyncGenerator[Dict, None]:
    while True:
        # Stream text as it arrives
        for chunk in response.text:
            yield {"type": "text_delta", "text": chunk}

        # Execute tools
        for tool_use in tool_uses:
            result = await self.tools.execute(...)
            yield {"type": "tool_execution", "tool_name": ..., "result": result}

        if done:
            yield {"type": "done"}
            break
```

**Benefits**:
- Real-time feedback to user
- Progress visibility during long operations
- Can cancel if needed
- Better UX than blocking

---

## 📊 Summary of What You Learned

### Architecture
✅ 3-layer design (Interface → Agent → Tools)
✅ Agentic loop (LLM decides what to do)
✅ Tool registry pattern (centralized management)

### Tools
✅ BaseTool abstract class (consistent interface)
✅ BashTool (foreground vs background modes)
✅ File tools (Read/Write/Edit with pagination)
✅ Search tools (Glob/Grep for navigation)
✅ Meta tools (Todo/RunSummary/Oracle)

### Memory
✅ Conversation history (full context for LLM)
✅ Workspace persistence (files survive session)
✅ Structured logging (experiments, tasks, processes)

### Oracle Upgrade
✅ Multi-model ensemble (O3 + DeepSeek-R1)
✅ Parallel consultation (both think simultaneously)
✅ Critic synthesis (O3 combines best elements)
✅ 4x more reasoning (32K vs 8K tokens)

### Key Insights
✅ **Autonomy**: Agent decides dynamically (no hardcoded workflow)
✅ **Adaptability**: Error recovery through tool results
✅ **Transparency**: Full history visible for debugging
✅ **Scalability**: Easy to add new tools (just inherit BaseTool)
✅ **Security**: Validation at prehook layer
✅ **Quality**: Multi-model validation catches errors

---

## 🚀 Next Steps (For You)

1. **Run existing agent** - see it in action
2. **Add custom tool** - extend for your use case
3. **Test Oracle upgrade** - verify multi-model works
4. **Optimize context** - implement truncation strategy
5. **Add more models** - Claude, Gemini, etc. to ensemble

---

**You now understand:**
- How autonomous agents work (agentic loop)
- How tools are structured (BaseTool pattern)
- How memory is managed (conversation history + workspace)
- How Oracle upgrade improves quality (multi-model ensemble)

**Files to review**:
- `AGENT_ARCHITECTURE_EXPLAINED.md` - Deep dive with diagrams
- `ORACLE_UPGRADE_SUMMARY.md` - Complete upgrade details
- `agent_v5/agent.py` - Agentic loop implementation
- `agent_v5/tools/oracle.py` - Upgraded Oracle tool

**Ready to build your own agent!** 🎉
