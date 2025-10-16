"""
ListBashProcessesTool – enumerate active and recently finished background bash jobs.

Returns a human-readable table with shell_id, status (RUNNING/COMPLETED), exit_code, runtime, and command snippet.

Useful for quick overview and resource hygiene.
"""

import time
from typing import Dict, Optional, List

from .base import BaseTool
from .bash_process_registry import BashProcessRegistry


class ListBashProcessesTool(BaseTool):
    """List active background bash processes maintained by BashProcessRegistry."""

    def __init__(self, workspace_dir: str, process_registry: Optional[BashProcessRegistry] = None):
        super().__init__(workspace_dir)
        self.process_registry = process_registry

    @property
    def name(self) -> str:
        return "ListBashProcesses"

    @property
    def schema(self) -> Dict:
        return {
            "name": "ListBashProcesses",
            "description": "List active or recently finished background bash jobs started via Bash(background=true).",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }

    async def execute(self, input: Dict) -> Dict:
        if self.process_registry is None:
            return {
                "content": "Background execution not configured; no process registry available.",
                "is_error": True,
            }

        entries: List[str] = []
        now = time.time()
        for shell_id, bg_proc in self.process_registry.list_all().items():
            status = "RUNNING" if bg_proc.process.returncode is None else f"COMPLETED ({bg_proc.process.returncode})"
            runtime = now - bg_proc.start_time
            cmd_snippet = (bg_proc.command[:60] + "…") if len(bg_proc.command) > 60 else bg_proc.command
            entries.append(f"{shell_id}\t{status}\t{runtime:.1f}s\t{cmd_snippet}")

        if not entries:
            return {"content": "No background bash jobs found.", "is_error": False}

        header = "shell_id\tstatus\truntime\tcommand"
        return {
            "content": header + "\n" + "\n".join(entries),
            "is_error": False,
        }
