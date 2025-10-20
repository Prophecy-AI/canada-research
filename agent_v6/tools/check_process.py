"""
CheckProcessTool - Monitor background process status and output
"""
import time
from typing import Dict

from agent_v5.tools.base import BaseTool


class CheckProcessTool(BaseTool):
    """Check status and read output of background process"""

    def __init__(self, workspace_dir: str, execute_script_tool):
        """
        Initialize check process tool

        Args:
            workspace_dir: Workspace directory
            execute_script_tool: ExecuteScriptTool instance to query
        """
        super().__init__(workspace_dir)
        self.execute_script_tool = execute_script_tool

    @property
    def name(self) -> str:
        return "CheckProcess"

    @property
    def schema(self) -> Dict:
        return {
            "name": "CheckProcess",
            "description": (
                "Check status of background process started with ExecuteScript. "
                "Returns process status, resource usage, and recent output."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "pid": {
                        "type": "integer",
                        "description": "Process ID to check"
                    },
                    "tail_lines": {
                        "type": "integer",
                        "description": "Number of recent output lines to show (default: 50)"
                    }
                },
                "required": ["pid"]
            }
        }

    async def execute(self, input: Dict) -> Dict:
        """Check process status"""
        try:
            pid = input["pid"]
            tail_lines = input.get("tail_lines", 50)

            # Get process info
            proc_info = self.execute_script_tool.get_process_info(pid)

            if not proc_info:
                return {
                    "content": f"Process {pid} not found. It may have completed or never existed.",
                    "is_error": True
                }

            # Format status
            status = proc_info["status"]
            command = proc_info["command"]
            started_at = proc_info["started_at"]
            elapsed = time.time() - started_at

            # Get recent output
            output_buffer = proc_info.get("output_buffer", [])
            recent_output = "".join(output_buffer[-tail_lines:]) if output_buffer else "(no output yet)"

            # Format response
            response_parts = [
                f"Process {pid}: {status.upper()}",
                f"Command: {command}",
                f"Elapsed: {elapsed:.1f}s"
            ]

            # Add resource stats if available
            if "cpu_percent" in proc_info:
                response_parts.append(f"CPU: {proc_info['cpu_percent']:.1f}%")
            if "memory_mb" in proc_info:
                response_parts.append(f"Memory: {proc_info['memory_mb']:.1f}MB")

            # Add output count
            response_parts.append(f"Output lines: {proc_info['total_output_lines']}")

            # Add exit code if completed
            if "exit_code" in proc_info:
                response_parts.append(f"Exit code: {proc_info['exit_code']}")

            # Add recent output
            response_parts.append(f"\n{'='*60}")
            response_parts.append(f"Recent output (last {len(output_buffer[-tail_lines:])} lines):")
            response_parts.append(f"{'='*60}")
            response_parts.append(recent_output)

            return {
                "content": "\n".join(response_parts),
                "is_error": False,
                "debug_summary": f"Checked process {pid} ({status})"
            }

        except Exception as e:
            return {
                "content": f"Error checking process: {str(e)}",
                "is_error": True
            }
