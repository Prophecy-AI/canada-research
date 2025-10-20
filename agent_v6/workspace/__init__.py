"""
IDE-centric workspace management for agent_v6

Tracks:
- File state (created, modified, deleted)
- Notebook state (cells, outputs, kernel status)
- Process state (running scripts, resource usage, convergence)
"""

__all__ = ["IDEWorkspace"]

from agent_v6.workspace.state import IDEWorkspace
