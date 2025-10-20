"""
ExecuteScriptTool - Non-blocking script execution with monitoring

Key features from Operand Quant:
- Background execution (doesn't block agent reasoning)
- Resource monitoring (CPU, memory)
- Output streaming
- Convergence detection (optional)
- TTY emulation for unbuffered output
"""
import os
import asyncio
import time
import psutil
import shlex
import uuid
from typing import Dict, Optional
from pathlib import Path

from agent_v5.tools.base import BaseTool


class ExecuteScriptTool(BaseTool):
    """
    Execute Python/Bash scripts in background with monitoring

    Non-blocking: Script runs in background while agent continues reasoning.
    Use CheckProcessTool to monitor progress and InterruptProcessTool to stop.
    """

    def __init__(self, workspace_dir: str, workspace_state=None):
        """
        Initialize script execution tool

        Args:
            workspace_dir: Workspace directory
            workspace_state: Optional IDEWorkspace instance for state tracking
        """
        super().__init__(workspace_dir)
        self.workspace_state = workspace_state
        self.processes: Dict[int, Dict] = {}  # PID -> process info

    @property
    def name(self) -> str:
        return "ExecuteScript"

    @property
    def schema(self) -> Dict:
        return {
            "name": "ExecuteScript",
            "description": (
                "Execute Python or Bash script in background (non-blocking). "
                "The script runs while you continue reasoning. "
                "Use CheckProcessTool to monitor progress and read output. "
                "Use InterruptProcessTool to stop the process."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "script_path": {
                        "type": "string",
                        "description": "Relative path to script file (e.g., 'train.py', 'analysis.sh')"
                    },
                    "interpreter": {
                        "type": "string",
                        "description": "Interpreter to use: 'python', 'bash', 'python3' (default: auto-detect from extension)",
                        "enum": ["python", "python3", "bash", "sh", "auto"]
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Command line arguments for script (optional)"
                    },
                    "env_vars": {
                        "type": "object",
                        "description": "Environment variables to set (optional)"
                    }
                },
                "required": ["script_path"]
            }
        }

    async def execute(self, input: Dict) -> Dict:
        """Start script execution in background"""
        try:
            script_path = input["script_path"]
            interpreter = input.get("interpreter", "auto")
            args = input.get("args", [])
            env_vars = input.get("env_vars", {})

            # Resolve path
            abs_path = os.path.join(self.workspace_dir, script_path)

            if not os.path.exists(abs_path):
                return {
                    "content": f"Script not found: {script_path}",
                    "is_error": True
                }

            # Auto-detect interpreter
            if interpreter == "auto":
                ext = Path(script_path).suffix
                if ext == ".py":
                    interpreter = "python"
                elif ext in [".sh", ".bash"]:
                    interpreter = "bash"
                else:
                    return {
                        "content": f"Cannot auto-detect interpreter for {ext}. Specify interpreter explicitly.",
                        "is_error": True
                    }

            # Build command
            if interpreter in ["python", "python3"]:
                command_parts = [interpreter, abs_path] + args
            elif interpreter in ["bash", "sh"]:
                command_parts = [interpreter, abs_path] + args
            else:
                return {
                    "content": f"Unknown interpreter: {interpreter}",
                    "is_error": True
                }

            # Join command for script wrapper
            command_str = " ".join(shlex.quote(part) for part in command_parts)

            # Prepare environment
            env = os.environ.copy()
            env.update(env_vars)
            env["PYTHONUNBUFFERED"] = "1"  # Ensure immediate output

            # Create log directory for typescript files
            log_dir = os.path.join(self.workspace_dir, ".pty_logs")
            os.makedirs(log_dir, exist_ok=True)

            # Generate unique shell ID for this execution
            shell_id = f"script_{uuid.uuid4().hex[:8]}"
            typescript_path = os.path.join(log_dir, f"{shell_id}.typescript")

            # Wrap with script -q -c for TTY emulation (unbuffered output, progress bars work)
            # This matches agent_v5's BashTool behavior
            wrapped_cmd = f'script -q -c {shlex.quote(command_str)} {shlex.quote(typescript_path)}'

            # Start process with TTY emulation
            process = await asyncio.create_subprocess_shell(
                wrapped_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
                cwd=self.workspace_dir,
                env=env
            )

            # Track process
            self.processes[process.pid] = {
                "process": process,
                "command": command_str,
                "script_path": script_path,
                "shell_id": shell_id,
                "typescript_path": typescript_path,
                "started_at": time.time(),
                "output_buffer": [],
                "total_output_lines": 0,
                "status": "running"
            }

            # Start background monitoring
            asyncio.create_task(self._monitor_process(process.pid))

            # Track in workspace
            if self.workspace_state:
                self.workspace_state.track_process_started(process.pid, command_str)

            return {
                "content": (
                    f"✓ Started background process (PID: {process.pid})\n"
                    f"Script: {script_path}\n"
                    f"Command: {command_str}\n"
                    f"Shell ID: {shell_id}\n\n"
                    f"✓ TTY emulation enabled (unbuffered output, progress bars work)\n\n"
                    f"Use CheckProcessTool(pid={process.pid}) to monitor progress.\n"
                    f"Use InterruptProcessTool(pid={process.pid}) to stop."
                ),
                "is_error": False,
                "debug_summary": f"Started {script_path} (PID {process.pid}, TTY enabled)"
            }

        except Exception as e:
            return {
                "content": f"Failed to start script: {str(e)}",
                "is_error": True
            }

    async def _monitor_process(self, pid: int) -> None:
        """Monitor process in background"""
        if pid not in self.processes:
            return

        proc_info = self.processes[pid]
        process = proc_info["process"]

        try:
            # Read output in background
            async for line in process.stdout:
                decoded = line.decode('utf-8', errors='replace')

                # Store in buffer (keep last 1000 lines)
                proc_info["output_buffer"].append(decoded)
                if len(proc_info["output_buffer"]) > 1000:
                    proc_info["output_buffer"].pop(0)

                proc_info["total_output_lines"] += 1

                # Update resource stats every 10 lines
                if proc_info["total_output_lines"] % 10 == 0:
                    await self._update_resource_stats(pid)

            # Wait for process to complete
            exit_code = await process.wait()

            # Mark completed
            proc_info["status"] = "completed" if exit_code == 0 else "failed"
            proc_info["exit_code"] = exit_code
            proc_info["completed_at"] = time.time()

            # Track in workspace
            if self.workspace_state:
                self.workspace_state.track_process_completed(pid, exit_code)

        except Exception as e:
            proc_info["status"] = "error"
            proc_info["error"] = str(e)

    async def _update_resource_stats(self, pid: int) -> None:
        """Update CPU and memory usage stats"""
        try:
            proc = psutil.Process(pid)
            cpu_percent = proc.cpu_percent(interval=0.1)
            memory_mb = proc.memory_info().rss / (1024 * 1024)

            proc_info = self.processes.get(pid)
            if proc_info:
                proc_info["cpu_percent"] = cpu_percent
                proc_info["memory_mb"] = memory_mb

                # Track in workspace
                if self.workspace_state:
                    self.workspace_state.track_process_stats(
                        pid,
                        cpu_percent,
                        memory_mb,
                        proc_info["total_output_lines"]
                    )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def get_process_info(self, pid: int) -> Optional[Dict]:
        """Get process information"""
        return self.processes.get(pid)

    def get_all_processes(self) -> Dict[int, Dict]:
        """Get all tracked processes"""
        return self.processes.copy()

    async def cleanup(self) -> None:
        """Kill all running processes"""
        for pid, proc_info in list(self.processes.items()):
            if proc_info["status"] == "running":
                try:
                    process = proc_info["process"]
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except:
                    try:
                        process.kill()
                    except:
                        pass
