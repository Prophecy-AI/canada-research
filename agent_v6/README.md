# agent_v6: Operand Quant-Inspired IDE Agent

**A production-ready autonomous IDE agent with multi-provider ensemble reasoning and first-class Jupyter support.**

Based on the architecture from [Operand Quant](https://arxiv.org/abs/2510.11694), agent_v6 implements a single-agent IDE-based system with advanced capabilities.

## Architecture

```
agent_v6/
├── agent.py              # IDEAgent - main orchestrator
├── ensemble/             # Multi-provider ensemble reasoning
│   └── tool.py          # 4 models + O3 synthesis
├── workspace/            # IDE state tracking
│   └── state.py         # Files, notebooks, processes
├── tools/                # IDE capabilities
│   ├── notebook.py      # Jupyter kernel management
│   ├── execute_script.py # Non-blocking execution
│   ├── check_process.py  # Process monitoring
│   └── interrupt_process.py # Process control
├── memory/               # Memory management
│   └── compactor.py     # Hierarchical compression
├── prompts/              # System prompts
│   └── generic_ide.py   # Generic IDE assistant
└── tests/                # Real API tests (29 tests)
```

## Key Features

### 1. Multi-Provider Ensemble Reasoning

Consult 4 frontier AI models in parallel for complex decisions:
- **GPT-5**: High reasoning, high verbosity
- **Claude Opus 4.1**: Extended thinking (10K budget tokens)
- **Grok-4 Fast Reasoning**: High effort reasoning
- **Gemini 2.5 Pro**: Dynamic thinking with thoughts
- **O3 Synthesizer**: Combines responses into actionable plan

```python
# Agent automatically has access to ConsultEnsemble tool
agent.run("Should I use ResNet-50 or EfficientNet-B0 for image classification?")
# Ensemble returns synthesized expert advice from all 4 models
```

### 2. Jupyter Notebook Support (First-Class)

Execute code cells in persistent kernels:
- Each notebook maintains its own kernel
- Variables persist across cell executions
- Async execution with output capture
- Error handling with tracebacks

```python
# Create notebook
notebook_tool.execute({
    "notebook_path": "analysis.ipynb",
    "cell_index": 0  # or -1 for all cells
})
```

### 3. Non-Blocking Script Execution

Run scripts in background while agent continues reasoning:
- Background execution (doesn't block agent)
- Resource monitoring (CPU, memory)
- Output streaming
- Process control (interrupt, check status)

```python
# Start script (non-blocking)
execute_script_tool.execute({"script_path": "train.py"})

# Check progress while it runs
check_process_tool.execute({"pid": 12345})

# Interrupt if needed
interrupt_process_tool.execute({"pid": 12345})
```

### 4. Hierarchical Memory Compaction

Automatic conversation history compression for long sessions:
- Keeps recent N messages intact (high fidelity)
- Summarizes older messages with O3
- Enables handling very long conversations
- Configurable compression ratio

### 5. IDE Workspace State Tracking

Tracks workspace state across:
- **Files**: created, modified, deleted
- **Notebooks**: kernel status, execution count, cells
- **Processes**: CPU/memory usage, output, status

## Installation

```bash
# Required packages
pip install openai anthropic google-genai xai-sdk jupyter-client psutil
```

## Quick Start

### CLI Interface

```bash
cd agent_v6
python cli.py
```

Example session:
```
You: Create a notebook called test.ipynb
Agent: Created test.ipynb

You: Add a cell that prints hello world
Agent: Added cell to test.ipynb

You: Execute the notebook
Agent: Executed cell 0: "hello world"
```

### Python API

```python
from agent_v6.agent import IDEAgent
from agent_v6.prompts.generic_ide import GENERIC_IDE_PROMPT

# Create agent
agent = IDEAgent(
    session_id="my_session",
    workspace_dir="./workspace",
    system_prompt=GENERIC_IDE_PROMPT,
    enable_memory_compaction=True
)

# Run agent
async for message in agent.run("Analyze data.csv"):
    if message["type"] == "text_delta":
        print(message["text"], end="", flush=True)

# Cleanup
await agent.cleanup()
```

## Testing

**All tests use REAL API calls** (no mocking).

```bash
# Run all tests
pytest agent_v6/tests/ -v

# Specific test suites
pytest agent_v6/tests/test_ensemble.py -v      # Ensemble (4 models + O3)
pytest agent_v6/tests/test_workspace.py -v      # Workspace state
pytest agent_v6/tests/test_notebook.py -v       # Jupyter kernels
pytest agent_v6/tests/test_execute_script.py -v # Process execution
pytest agent_v6/tests/test_memory.py -v         # Memory compaction
pytest agent_v6/tests/test_agent.py -v          # End-to-end agent
```

### Test Results

```
✅ 29 tests passing
- 3 ensemble tests (real 4-model queries + O3)
- 7 workspace state tests
- 6 notebook tests (real Jupyter kernels)
- 6 process execution tests
- 4 memory compaction tests (real O3 API)
- 3 end-to-end agent tests (real GPT-5)
```

## API Keys Required

Set these environment variables:

```bash
export OPENAI_API_KEY="..."      # For GPT-5, O3
export ANTHROPIC_API_KEY="..."   # For Claude Opus 4.1
export XAI_API_KEY="..."          # For Grok-4
export GEMINI_API_KEY="..."       # For Gemini 2.5 Pro
```

## Cost Estimates

Approximate costs per usage:
- **Ensemble consultation**: ~$1-2 per query (4 models + O3)
- **Memory compaction**: ~$0.10-0.20 per compaction (O3)
- **Regular agent turns**: ~$0.01-0.05 per turn (GPT-5)

## Differences from agent_v5

| Feature | agent_v5 | agent_v6 |
|---------|----------|----------|
| Main model | GPT-5 | GPT-5 |
| Ensemble | 2 models (O3 + Gemini) | 4 models + O3 synthesis |
| Notebooks | No | ✅ First-class (jupyter_client) |
| Script execution | Bash only | ✅ Python/Bash with monitoring |
| Process control | Basic | ✅ CheckProcess, InterruptProcess |
| Memory | No compaction | ✅ Hierarchical O3 compression |
| Workspace | Simple dir | ✅ Full state tracking |
| Domain | Healthcare research | Generic IDE |

## Architecture Highlights

### Based on Proven Patterns

agent_v6 extends the battle-tested agent_v5 architecture:
- ✅ Same tool registry pattern
- ✅ Same agentic loop structure
- ✅ Same OpenAI Responses API integration
- ✅ Same streaming pattern
- ✅ Same cleanup mechanisms

### New Innovations

1. **Parallel Ensemble**: 4 models queried simultaneously with `asyncio.gather()`
2. **Persistent Kernels**: Jupyter kernels live across multiple cell executions
3. **Background Monitoring**: Scripts run while agent reasons about next steps
4. **Automatic Compaction**: O3 summarizes old conversation history when needed

## Example Use Cases

### 1. Data Analysis

```python
agent.run("""
Analyze sales_data.csv:
1. Create a notebook
2. Load the data
3. Calculate monthly revenue
4. Create visualization
5. Export results
""")
```

### 2. Model Training

```python
agent.run("""
Train an image classifier:
1. Write training script
2. Start training in background
3. Monitor progress every 30 seconds
4. Save best checkpoint
""")
```

### 3. Complex Decisions

```python
agent.run("""
I need to choose between PostgreSQL and MongoDB for my app.
Use the ensemble to get expert perspectives on:
- Performance trade-offs
- Scaling considerations
- Development complexity
""")
```

## Limitations

1. **Cost**: Ensemble queries are expensive (~$1-2 each)
2. **Latency**: Ensemble takes 30-60s (4 parallel queries + synthesis)
3. **Kernel overhead**: Each notebook uses ~50MB RAM for kernel
4. **Process monitoring**: Requires psutil (may not work in all environments)

## Roadmap

Future improvements:
- [ ] Multi-notebook coordination
- [ ] Convergence detection for training scripts
- [ ] Git integration for version control
- [ ] Terminal emulation for interactive commands
- [ ] Distributed execution (Modal/Ray)

## References

- Paper: [Operand Quant](https://arxiv.org/abs/2510.11694)
- Base: `agent_v5/` framework
- Docs: See `CLAUDE.md` for full framework guide

## License

Same as parent repository.
