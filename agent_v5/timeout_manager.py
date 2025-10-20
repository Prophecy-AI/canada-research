"""
TimeoutManager - Intelligent timeout and stall detection for agent runs

Uses EstimateTaskDuration tool to set dynamic timeouts and detect stalls.
"""
import time
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TaskTracker:
    """Track a running task with its estimated duration"""
    task_name: str
    start_time: float
    estimated_duration: float  # seconds (typical)
    max_duration: float  # seconds (worst case)
    last_output_time: float
    completed: bool = False


class TimeoutManager:
    """
    Manages timeouts and stall detection for agent runs.

    Features:
    - Global timeout (max total runtime)
    - Per-task timeouts (based on estimates)
    - Stall detection (no progress for N minutes)
    - Turn limit (max agentic loop iterations)
    """

    def __init__(
        self,
        max_runtime_seconds: Optional[int] = 7200,  # 2 hours default
        max_turns: Optional[int] = 100,  # 100 turns default
        stall_timeout_seconds: int = 600,  # 10 minutes no output = stall
    ):
        self.max_runtime_seconds = max_runtime_seconds
        self.max_turns = max_turns
        self.stall_timeout_seconds = stall_timeout_seconds

        self.start_time = time.time()
        self.turn_count = 0
        self.current_task: Optional[TaskTracker] = None
        self.completed_tasks: List[TaskTracker] = []

        # Track last activity for stall detection
        self.last_activity_time = time.time()

    def check_timeout(self) -> Dict[str, any]:
        """
        Check if any timeout conditions are met.

        Returns:
            {
                "timed_out": bool,
                "reason": str,
                "elapsed_seconds": float,
                "remaining_seconds": float
            }
        """
        elapsed = time.time() - self.start_time
        result = {
            "timed_out": False,
            "reason": None,
            "elapsed_seconds": elapsed,
            "remaining_seconds": None
        }

        # Check global timeout
        if self.max_runtime_seconds and elapsed > self.max_runtime_seconds:
            result["timed_out"] = True
            result["reason"] = f"Global timeout: exceeded {self._format_duration(self.max_runtime_seconds)}"
            result["remaining_seconds"] = 0
            return result

        # Check turn limit
        if self.max_turns and self.turn_count >= self.max_turns:
            result["timed_out"] = True
            result["reason"] = f"Turn limit: exceeded {self.max_turns} turns"
            result["remaining_seconds"] = 0
            return result

        # Check stall (no activity)
        time_since_activity = time.time() - self.last_activity_time
        if time_since_activity > self.stall_timeout_seconds:
            result["timed_out"] = True
            result["reason"] = f"Stalled: no activity for {self._format_duration(time_since_activity)}"
            result["remaining_seconds"] = 0
            return result

        # Check current task timeout
        if self.current_task and not self.current_task.completed:
            task_elapsed = time.time() - self.current_task.start_time
            if task_elapsed > self.current_task.max_duration * 1.5:  # 1.5x buffer
                result["timed_out"] = True
                result["reason"] = (
                    f"Task timeout: '{self.current_task.task_name}' exceeded "
                    f"{self._format_duration(self.current_task.max_duration * 1.5)} "
                    f"(estimated {self._format_duration(self.current_task.estimated_duration)})"
                )
                result["remaining_seconds"] = 0
                return result

        # No timeout - calculate remaining time
        if self.max_runtime_seconds:
            result["remaining_seconds"] = self.max_runtime_seconds - elapsed

        return result

    def start_turn(self):
        """Mark the start of a new turn in the agentic loop"""
        self.turn_count += 1
        self.last_activity_time = time.time()

    def register_activity(self):
        """Mark that activity occurred (output received, tool executed, etc.)"""
        self.last_activity_time = time.time()

    def start_task(self, task_name: str, estimated_duration: float, max_duration: float):
        """
        Start tracking a new task with estimated duration.

        Args:
            task_name: Name of the task (e.g., "train_complex_model")
            estimated_duration: Typical duration in seconds
            max_duration: Worst-case duration in seconds
        """
        # Complete previous task if any
        if self.current_task and not self.current_task.completed:
            self.complete_task()

        self.current_task = TaskTracker(
            task_name=task_name,
            start_time=time.time(),
            estimated_duration=estimated_duration,
            max_duration=max_duration,
            last_output_time=time.time()
        )
        self.register_activity()

        # Log task start for debugging
        from debug import log
        log(f"⏱️  Started task: {task_name} (est: {self._format_duration(estimated_duration)}, max: {self._format_duration(max_duration)})")

    def complete_task(self):
        """Mark the current task as completed"""
        if self.current_task:
            self.current_task.completed = True
            self.current_task.last_output_time = time.time()
            self.completed_tasks.append(self.current_task)
            self.current_task = None
        self.register_activity()

    def get_task_status(self) -> Optional[Dict]:
        """
        Get status of the current task.

        Returns:
            {
                "task_name": str,
                "elapsed_seconds": float,
                "estimated_duration": float,
                "max_duration": float,
                "progress_percent": float,  # based on estimated duration
                "is_overdue": bool  # exceeded estimated duration
            }
        """
        if not self.current_task or self.current_task.completed:
            return None

        elapsed = time.time() - self.current_task.start_time
        progress = min(100, (elapsed / self.current_task.estimated_duration) * 100)

        return {
            "task_name": self.current_task.task_name,
            "elapsed_seconds": elapsed,
            "estimated_duration": self.current_task.estimated_duration,
            "max_duration": self.current_task.max_duration,
            "progress_percent": progress,
            "is_overdue": elapsed > self.current_task.estimated_duration,
            "formatted_elapsed": self._format_duration(elapsed),
            "formatted_estimated": self._format_duration(self.current_task.estimated_duration)
        }

    def get_summary(self) -> Dict:
        """
        Get summary of entire run.

        Returns:
            {
                "total_runtime": float,
                "turn_count": int,
                "tasks_completed": int,
                "current_task": Optional[str],
                "timeout_check": Dict  # from check_timeout()
            }
        """
        return {
            "total_runtime": time.time() - self.start_time,
            "formatted_runtime": self._format_duration(time.time() - self.start_time),
            "turn_count": self.turn_count,
            "tasks_completed": len(self.completed_tasks),
            "current_task": self.current_task.task_name if self.current_task else None,
            "timeout_check": self.check_timeout()
        }

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def should_warn_user(self) -> Optional[str]:
        """
        Check if user should be warned about task taking too long.

        Returns warning message if task is overdue, None otherwise.
        """
        if not self.current_task or self.current_task.completed:
            return None

        elapsed = time.time() - self.current_task.start_time

        # Warn if exceeded estimated duration
        if elapsed > self.current_task.estimated_duration:
            overrun = elapsed - self.current_task.estimated_duration
            return (
                f"⚠️  Task '{self.current_task.task_name}' is taking longer than expected:\n"
                f"   Expected: {self._format_duration(self.current_task.estimated_duration)}\n"
                f"   Actual: {self._format_duration(elapsed)} ({self._format_duration(overrun)} overrun)\n"
                f"   Max allowed: {self._format_duration(self.current_task.max_duration)}"
            )

        return None

    def parse_estimate_result(self, estimate_result: str) -> Optional[Dict]:
        """
        Parse EstimateTaskDuration tool output to extract durations.

        Args:
            estimate_result: The text output from EstimateTaskDuration tool

        Returns:
            {"task_name": str, "min": float, "typical": float, "max": float} or None
        """
        import re

        # Look for duration estimates in format "12m 30s" or "1h 30m"
        # Example output:
        # ⏱️ Task Duration Estimate: train_complex_model
        # 📊 Estimated Duration:
        #    ⚡ Best case:  12m 0s
        #    📈 Typical:    30m 0s
        #    ⚠️  Worst case: 1h 30m

        # Extract task name
        task_match = re.search(r'Task Duration Estimate:\s*(\w+)', estimate_result)
        if not task_match:
            return None

        task_name = task_match.group(1)

        # Extract durations
        def parse_duration(text: str) -> Optional[float]:
            """Parse '1h 30m 45s' format to seconds"""
            hours = minutes = seconds = 0

            h_match = re.search(r'(\d+)h', text)
            m_match = re.search(r'(\d+)m', text)
            s_match = re.search(r'(\d+)s', text)

            if h_match:
                hours = int(h_match.group(1))
            if m_match:
                minutes = int(m_match.group(1))
            if s_match:
                seconds = int(s_match.group(1))

            if hours == 0 and minutes == 0 and seconds == 0:
                return None

            return hours * 3600 + minutes * 60 + seconds

        typical_match = re.search(r'Typical:\s*(.+?)(?:\n|$)', estimate_result)
        max_match = re.search(r'Worst case:\s*(.+?)(?:\n|$)', estimate_result)

        if not typical_match or not max_match:
            return None

        typical_duration = parse_duration(typical_match.group(1))
        max_duration = parse_duration(max_match.group(1))

        if typical_duration is None or max_duration is None:
            return None

        return {
            "task_name": task_name,
            "typical": typical_duration,
            "max": max_duration
        }
