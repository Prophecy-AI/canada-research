"""
agent_v6: Operand Quant-inspired IDE agent framework with multi-provider ensemble reasoning

Key Features:
- IDE-centric workspace management
- Jupyter notebook support (first-class citizen)
- Non-blocking script execution with monitoring
- Hierarchical memory compaction
- Multi-provider ensemble reasoning (GPT-5, Claude Opus, Grok-4, Gemini 2.5 Pro, O3 synthesis)

Main exports:
- IDEAgent: Main agent class
- IDEWorkspace: Workspace state tracking
- EnsembleTool: Multi-provider reasoning tool
- MemoryCompactor: Hierarchical memory compression
"""

__version__ = "0.1.0"

__all__ = [
    "IDEAgent",
    "IDEWorkspace",
    "EnsembleTool",
    "MemoryCompactor",
]

from agent_v6.agent import IDEAgent
from agent_v6.workspace import IDEWorkspace
from agent_v6.ensemble.tool import EnsembleTool
from agent_v6.memory import MemoryCompactor
