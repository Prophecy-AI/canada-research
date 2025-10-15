"""
Kill a background bash process
"""
import asyncio
from typing import Dict, Optional
from .base import BaseTool
from .bash_process_registry import BashProcessRegistry


class KillShellTool(BaseTool):
    """Kill a running background bash process"""

    def __init__(self, workspace_dir: str, process_registry: Optional[BashProcessRegistry] = None):
        """
        Initialize KillShell tool

        Args:
            workspace_dir: Workspace directory (required by BaseTool but not used)
            process_registry: Registry to modify
        """
        super().__init__(workspace_dir)
        self.process_registry = process_registry

    @property
    def name(self) -> str:
        return "KillShell"

    @property
    def schema(self) -> Dict:
        return {
            "name": "KillShell",
            "description": (
                "Kill a running background bash process. Use this to stop a long-running "
                "command that is no longer needed, is consuming too many resources, or "
                "is stuck. The process will be terminated immediately (SIGKILL)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "shell_id": {
                        "type": "string",
                        "description": "Shell ID to kill (e.g., 'bash_a1b2c3d4')"
                    }
                },
                "required": ["shell_id"]
            }
        }

    async def execute(self, input: Dict) -> Dict:
        """Kill background process and remove from registry"""
        if self.process_registry is None:
            return {
                "content": (
                    "KillShell not available: no process registry configured.\n"
                    "Background execution requires a BashProcessRegistry."
                ),
                "is_error": True
            }

        shell_id = input["shell_id"]
        bg_process = self.process_registry.get(shell_id)

        if not bg_process:
            return {
                "content": (
                    f"Shell {shell_id} not found.\n\n"
                    f"The process may have already completed or been killed."
                ),
                "is_error": True
            }

        # Check if process is still running
        was_running = bg_process.process.returncode is None

        # Kill process immediately - most aggressive approach
        try:
            # Step 1: Cancel collector task immediately (don't wait)
            if bg_process.collector_task and not bg_process.collector_task.done():
                bg_process.collector_task.cancel()
            
            # Step 2: Close streams to unblock any pending reads
            try:
                if bg_process.process.stdout:
                    bg_process.process.stdout.close()
                if bg_process.process.stderr:
                    bg_process.process.stderr.close()
            except Exception:
                pass
            
            # Step 3: SIGKILL the process (most brutal)
            if was_running:
                try:
                    bg_process.process.kill()
                except Exception:
                    pass  # Already dead, fine
            
            # Step 4: Remove from registry immediately (don't wait for cleanup)
            self.process_registry.remove(shell_id)

            # Calculate runtime
            import time
            runtime_s = time.time() - bg_process.start_time

            if was_running:
                content = (
                    f"Killed {shell_id}\n"
                    f"Command: {bg_process.command}\n"
                    f"Runtime: {runtime_s:.1f}s"
                )
            else:
                content = (
                    f"Removed {shell_id} (already completed with exit code {bg_process.process.returncode})\n"
                    f"Command: {bg_process.command}\n"
                    f"Runtime: {runtime_s:.1f}s"
                )

            return {
                "content": content,
                "is_error": False,
                "debug_summary": f"killed {shell_id}" if was_running else f"cleaned up {shell_id}"
            }

        except Exception as e:
            return {
                "content": f"Error killing {shell_id}: {str(e)}",
                "is_error": True
            }
