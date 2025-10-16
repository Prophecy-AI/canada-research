"""
Read incremental output from background bash process
"""
from typing import Dict, Optional
from .base import BaseTool
from .bash_process_registry import BashProcessRegistry

# Maximum output size to return to agent (to prevent context window overflow)
# Training logs can be megabytes - we only need recent output
MAX_OUTPUT_SIZE = 20 * 1024  # 20KB (~5K tokens, ~200 lines)


class ReadBashOutputTool(BaseTool):
    """Read new output from background bash process (cursor-based incremental reading)"""

    def __init__(self, workspace_dir: str, process_registry: Optional[BashProcessRegistry] = None):
        """
        Initialize ReadBashOutput tool

        Args:
            workspace_dir: Workspace directory (required by BaseTool but not used)
            process_registry: Registry to query for background processes
        """
        super().__init__(workspace_dir)
        self.process_registry = process_registry

    @property
    def name(self) -> str:
        return "ReadBashOutput"

    @property
    def schema(self) -> Dict:
        return {
            "name": "ReadBashOutput",
            "description": (
                "Read new output from a background bash process. Returns only NEW output "
                "since the last read (cursor-based incremental reading). Use this to monitor "
                "long-running commands like model training. Call periodically to see progress. "
                "Each call advances the cursor, so you never see the same output twice."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "shell_id": {
                        "type": "string",
                        "description": "Shell ID from background Bash execution (e.g., 'bash_a1b2c3d4')"
                    }
                },
                "required": ["shell_id"]
            }
        }

    async def execute(self, input: Dict) -> Dict:
        """Read incremental output from background process"""
        if self.process_registry is None:
            return {
                "content": (
                    "ReadBashOutput not available: no process registry configured.\n"
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
                    f"Possible reasons:\n"
                    f"- Shell ID is incorrect\n"
                    f"- Process was already killed with KillShell\n"
                    f"- Process completed and was cleaned up\n\n"
                    f"Use Bash(background=true) to start a new background process."
                ),
                "is_error": True
            }

        # Get NEW output only (from cursor to current end)
        new_stdout = bg_process.stdout_data[bg_process.stdout_cursor:].decode('utf-8', errors='replace')
        new_stderr = bg_process.stderr_data[bg_process.stderr_cursor:].decode('utf-8', errors='replace')

        # Combine stdout and stderr (preserve chronological order as much as possible)
        # Note: Perfect chronological ordering isn't possible with separate streams
        new_output = new_stdout + new_stderr
        
        # Truncate output if too large (keep most recent output only)
        total_new_bytes = len(new_stdout) + len(new_stderr)
        truncated = False
        if len(new_output) > MAX_OUTPUT_SIZE:
            truncated = True
            # Keep last MAX_OUTPUT_SIZE characters (most recent output)
            new_output = new_output[-MAX_OUTPUT_SIZE:]

        # Update cursors (mark this output as read)
        bg_process.stdout_cursor = len(bg_process.stdout_data)
        bg_process.stderr_cursor = len(bg_process.stderr_data)

        # Check process status
        if bg_process.process.returncode is not None:
            status = f"COMPLETED (exit code: {bg_process.process.returncode})"
        else:
            status = "RUNNING"

        # Calculate runtime
        import time
        runtime_s = time.time() - bg_process.start_time

        # Check if process is stalled (no output for 60+ seconds)
        time_since_last_output = time.time() - bg_process.last_output_time
        stalled_hint = ""
        if time_since_last_output > 60 and bg_process.process.returncode is None:
            stalled_hint = (
                f"\n⚠️  WARNING: No output for {time_since_last_output:.0f} seconds. "
                f"Process may be stalled/hanging.\n"
                f"💡 Consider: KillShell(shell_id='{shell_id}') if not making progress\n"
            )
        
        # Format output
        if new_output.strip():
            truncation_notice = ""
            if truncated:
                truncation_notice = (
                    f"\n⚠️  Output truncated: showing last {MAX_OUTPUT_SIZE:,} chars of {total_new_bytes:,} total chars\n"
                    f"(Full output stored in memory, only recent output shown to save context)\n\n"
                )
            
            # Add hint to wait for more progress (only if still running)
            progress_hint = ""
            if bg_process.process.returncode is None:
                progress_hint = (
                    f"\n\n💡 Consider: Bash(command='sleep 30', background=false) to let the process make more progress before polling again"
                )
            
            content = (
                f"[{status}] {shell_id} (runtime: {runtime_s:.1f}s)\n"
                f"Command: {bg_process.command}\n"
                f"{stalled_hint}"
                f"{truncation_notice}\n"
                f"{new_output}"
                f"{progress_hint}"
            )
            debug_summary = f"{status}, {len(new_output)} bytes" + (" (truncated)" if truncated else "")
        else:
            # No new output - give agent options: wait or kill
            wait_time = 30 if time_since_last_output < 60 else 60
            content = (
                f"[{status}] {shell_id} (runtime: {runtime_s:.1f}s)\n"
                f"Command: {bg_process.command}\n"
                f"{stalled_hint}\n"
                f"(no new output since last read)\n\n"
                f"💡 Options:\n"
                f"  - Wait before polling again: Bash(command='sleep {wait_time}', background=false)\n"
                f"  - Stop if not needed: KillShell(shell_id='{shell_id}')"
            )
            debug_summary = f"{status}, no new output"

        return {
            "content": content,
            "is_error": False,
            "debug_summary": debug_summary
        }
