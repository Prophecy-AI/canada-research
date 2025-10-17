"""
ElapsedTime tool - Track how long agent has been working on the task
"""
import time
from typing import Dict
from .base import BaseTool


class ElapsedTimeTool(BaseTool):
    """
    Tool to check elapsed time since agent started working

    Helps agent:
    - Know how much time has passed
    - Plan remaining work within time budget
    - Decide when to consult Oracle
    - Trigger early stopping if time running out
    """

    def __init__(self, workspace_dir: str, start_time: float = None):
        """
        Initialize ElapsedTime tool

        Args:
            workspace_dir: Workspace directory (required by BaseTool)
            start_time: Unix timestamp when agent started (default: current time)
        """
        super().__init__(workspace_dir)
        self.start_time = start_time if start_time is not None else time.time()

    @property
    def name(self) -> str:
        return "ElapsedTime"

    @property
    def schema(self) -> Dict:
        return {
            "name": "ElapsedTime",
            "description": (
                "Check how much time has elapsed since the agent started working on this task. "
                "Returns elapsed time in minutes and seconds. Use this to:\n"
                "- Track progress against the 20±10 minute time budget\n"
                "- Decide if you need to speed up (e.g., reduce folds, kill slow training)\n"
                "- Plan when to consult Oracle with progress update\n"
                "- Trigger early stopping if time running out\n"
                "Example: After 15 minutes, if training not done, consider killing and using partial models."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }

    async def execute(self, input: Dict) -> Dict:
        """
        Calculate and return elapsed time

        Returns:
            Dict with elapsed time in various formats
        """
        try:
            current_time = time.time()
            elapsed_seconds = current_time - self.start_time

            # Calculate components
            elapsed_minutes = int(elapsed_seconds // 60)
            remaining_seconds = int(elapsed_seconds % 60)
            elapsed_hours = elapsed_minutes // 60
            remaining_minutes = elapsed_minutes % 60

            # Format output
            if elapsed_hours > 0:
                time_str = f"{elapsed_hours}h {remaining_minutes}m {remaining_seconds}s"
            else:
                time_str = f"{elapsed_minutes}m {remaining_seconds}s"

            # Calculate percentage of time budget used (assuming 20±10 min target)
            # Use 30 min as max budget
            max_budget_seconds = 30 * 60
            percent_used = (elapsed_seconds / max_budget_seconds) * 100

            # Determine status
            if elapsed_seconds < 10 * 60:  # < 10 min
                status = "On track - plenty of time"
                urgency = "low"
            elif elapsed_seconds < 20 * 60:  # < 20 min
                status = "On track - target time"
                urgency = "medium"
            elif elapsed_seconds < 25 * 60:  # < 25 min
                status = "Approaching time limit - reserve time for inference"
                urgency = "high"
            elif elapsed_seconds < 30 * 60:  # < 30 min
                status = "Near time limit - complete ASAP"
                urgency = "critical"
            else:  # > 30 min
                status = "Exceeded time budget - finish immediately"
                urgency = "critical"

            # Build response
            response = (
                f"⏱️  ELAPSED TIME: {time_str} ({elapsed_minutes} minutes)\n"
                f"📊 Time Budget: {percent_used:.1f}% of 30-minute budget used\n"
                f"🎯 Status: {status}\n"
                f"⚡ Urgency: {urgency.upper()}\n\n"
                f"GUIDANCE:\n"
            )

            # Add guidance based on time
            if elapsed_seconds < 10 * 60:
                response += (
                    "• You're early in the process - take time for good planning\n"
                    "• Consult Oracle for strategy validation\n"
                    "• Aim to finish in 15-20 minutes total\n"
                )
            elif elapsed_seconds < 20 * 60:
                response += (
                    "• You're in the target window - maintain pace\n"
                    "• If training still running, monitor progress\n"
                    "• Start predict.py soon if not already done\n"
                )
            elif elapsed_seconds < 25 * 60:
                response += (
                    "• You're approaching the limit - speed up\n"
                    "• If training not done, consider killing and using partial models\n"
                    "• Reserve at least 5 minutes for inference\n"
                    "• Consult Oracle if stuck\n"
                )
            elif elapsed_seconds < 30 * 60:
                response += (
                    "• TIME CRITICAL - finish ASAP\n"
                    "• Kill training if not done, use available models\n"
                    "• Run predict.py immediately\n"
                    "• Generate submission now\n"
                )
            else:
                response += (
                    "• EXCEEDED BUDGET - emergency mode\n"
                    "• Generate submission with whatever you have\n"
                    "• Don't start new training\n"
                )

            return {
                "content": response,
                "is_error": False,
                "debug_summary": f"Elapsed: {elapsed_minutes}m ({percent_used:.1f}% of budget)"
            }

        except Exception as e:
            return {
                "content": f"Error calculating elapsed time: {str(e)}",
                "is_error": True
            }
