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
                f"\nWARNING: No output for {time_since_last_output:.0f} seconds - process may be stalled/hanging\n"
                f"Consider: KillShell(shell_id='{shell_id}') if not making progress\n"
            )

        # Format output
        if new_output.strip():
            truncation_notice = ""
            if truncated:
                truncation_notice = (
                    f"\nOutput truncated: showing last {MAX_OUTPUT_SIZE:,} chars of {total_new_bytes:,} total chars\n"
                    f"(Full output stored in memory, only recent output shown to save context)\n\n"
                )

            # AGGRESSIVE ANOMALY DETECTION - kill immediately if anything looks wrong
            kill_warnings = []
            output_lower = new_output.lower()

            # Check for errors/exceptions
            if any(pattern in new_output for pattern in ['Error:', 'ERROR', 'Exception', 'Traceback', 'FAILED']):
                kill_warnings.append("ERRORS/EXCEPTIONS detected - training likely failing")

            # Check for GPU issues
            if 'cpu' in output_lower and 'gpu' not in output_lower and ('train' in output_lower or 'model' in output_lower):
                kill_warnings.append("Process using CPU instead of GPU - wasting time")

            # Check for NaN/Inf values
            if 'nan' in output_lower or 'inf' in output_lower:
                kill_warnings.append("NaN/Inf values detected - model unstable, training will fail")

            # Check for memory issues
            if any(pattern in output_lower for pattern in ['out of memory', 'oom', 'cuda out of memory', 'memory error']):
                kill_warnings.append("OUT OF MEMORY - batch size too large, training will crash")

            # Check for warnings
            if 'warning:' in output_lower or 'warn:' in output_lower:
                kill_warnings.append("Warnings detected - review carefully, may indicate issues")

            # Check for zero accuracy/terrible metrics (only if running)
            if bg_process.process.returncode is None:
                if any(pattern in output_lower for pattern in ['accuracy: 0.', 'auc: 0.', 'f1: 0.', 'loss: nan']):
                    kill_warnings.append("Terrible metrics (0.0 accuracy/AUC or NaN loss) - something is broken")

            # Build kill recommendation
            kill_recommendation = ""
            if kill_warnings and bg_process.process.returncode is None:
                kill_recommendation = (
                    f"\n\n{'='*60}\n"
                    f"IMMEDIATE ACTION REQUIRED\n"
                    f"{'='*60}\n"
                )
                for warning in kill_warnings:
                    kill_recommendation += f"  - {warning}\n"
                kill_recommendation += (
                    f"\nSTRONGLY RECOMMENDED: KillShell(shell_id='{shell_id}') NOW\n"
                    f"Every minute of wasted training = wasted compute\n"
                    f"Kill now, fix the issue, validate with Oracle, then re-run\n"
                    f"Don't wait hours for a training run you KNOW is broken\n"
                    f"{'='*60}\n"
                )

            # Add hint to wait for more progress (only if still running and no kill warnings)
            progress_hint = ""
            if bg_process.process.returncode is None and not kill_warnings:
                progress_hint = (
                    f"\n\nConsider: Bash(command='sleep 30', background=false) to let the process make more progress before polling again"
                )

            content = (
                f"[{status}] {shell_id} (runtime: {runtime_s:.1f}s)\n"
                f"Command: {bg_process.command}\n"
                f"{stalled_hint}"
                f"{truncation_notice}\n"
                f"{new_output}"
                f"{kill_recommendation}"
                f"{progress_hint}"
            )
            debug_summary = f"{status}, {len(new_output)} bytes" + (" (truncated)" if truncated else "") + (f", {len(kill_warnings)} warnings" if kill_warnings else "")
        else:
            # No new output - give agent options: wait or kill
            wait_time = 30 if time_since_last_output < 60 else 60
            content = (
                f"[{status}] {shell_id} (runtime: {runtime_s:.1f}s)\n"
                f"Command: {bg_process.command}\n"
                f"{stalled_hint}\n"
                f"(no new output since last read)\n\n"
                f"Options:\n"
                f"  - Wait before polling again: Bash(command='sleep {wait_time}', background=false)\n"
                f"  - Stop if not needed: KillShell(shell_id='{shell_id}')"
            )
            debug_summary = f"{status}, no new output"

        return {
            "content": content,
            "is_error": False,
            "debug_summary": debug_summary
        }
