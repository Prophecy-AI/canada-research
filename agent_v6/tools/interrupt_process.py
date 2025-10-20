"""
InterruptProcessTool - Stop background process
"""
import asyncio
from typing import Dict

from agent_v5.tools.base import BaseTool


class InterruptProcessTool(BaseTool):
    """Interrupt (stop) background process"""

    def __init__(self, workspace_dir: str, execute_script_tool, workspace_state=None):
        """
        Initialize interrupt process tool

        Args:
            workspace_dir: Workspace directory
            execute_script_tool: ExecuteScriptTool instance
            workspace_state: Optional IDEWorkspace instance
        """
        super().__init__(workspace_dir)
        self.execute_script_tool = execute_script_tool
        self.workspace_state = workspace_state

    @property
    def name(self) -> str:
        return "InterruptProcess"

    @property
    def schema(self) -> Dict:
        return {
            "name": "InterruptProcess",
            "description": (
                "Stop background process started with ExecuteScript. "
                "Sends SIGTERM first, then SIGKILL if process doesn't stop."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "pid": {
                        "type": "integer",
                        "description": "Process ID to stop"
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Force kill immediately (default: false, tries graceful shutdown first)"
                    }
                },
                "required": ["pid"]
            }
        }

    async def execute(self, input: Dict) -> Dict:
        """Stop process"""
        try:
            pid = input["pid"]
            force = input.get("force", False)

            # Get process info
            proc_info = self.execute_script_tool.get_process_info(pid)

            if not proc_info:
                return {
                    "content": f"Process {pid} not found",
                    "is_error": True
                }

            process = proc_info["process"]
            status = proc_info["status"]

            if status in ["completed", "failed", "error"]:
                return {
                    "content": f"Process {pid} already {status}",
                    "is_error": False
                }

            # Stop process
            try:
                if force:
                    # Force kill
                    process.kill()
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                    message = f"✓ Killed process {pid} (forced)"
                else:
                    # Try graceful shutdown first
                    process.terminate()

                    try:
                        await asyncio.wait_for(process.wait(), timeout=5.0)
                        message = f"✓ Terminated process {pid} (graceful)"
                    except asyncio.TimeoutError:
                        # Graceful shutdown failed, force kill
                        process.kill()
                        await asyncio.wait_for(process.wait(), timeout=5.0)
                        message = f"✓ Killed process {pid} (force after timeout)"
            except ProcessLookupError:
                # Process already dead
                message = f"✓ Process {pid} already stopped"
            except Exception as proc_err:
                # Other process errors (e.g., permissions, zombie)
                err_msg = str(proc_err) if str(proc_err) else type(proc_err).__name__
                return {
                    "content": f"Failed to stop process {pid}: {err_msg}",
                    "is_error": True,
                    "debug_summary": f"Process stop failed: {err_msg}"
                }

            # Update status
            proc_info["status"] = "killed"

            # Track in workspace
            if self.workspace_state:
                self.workspace_state.track_process_killed(pid)

            return {
                "content": message,
                "is_error": False,
                "debug_summary": f"Stopped process {pid}"
            }

        except Exception as e:
            error_msg = str(e) if str(e) else f"{type(e).__name__}"
            return {
                "content": f"Error stopping process: {error_msg}",
                "is_error": True,
                "debug_summary": f"Failed to stop {pid}: {error_msg}"
            }
