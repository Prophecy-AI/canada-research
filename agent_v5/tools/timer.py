"""
TimerTool - Check elapsed time since agent started
"""
import time
from typing import Dict, Callable
from agent_v5.tools.base import BaseTool


class TimerTool(BaseTool):
    """Check elapsed time since agent started"""

    def __init__(self, workspace_dir: str, get_start_time: Callable[[], float]):
        """
        Initialize timer tool

        Args:
            workspace_dir: Workspace directory (unused but required by base)
            get_start_time: Callable that returns the agent start time (Unix timestamp)
        """
        super().__init__(workspace_dir)
        self.get_start_time = get_start_time

    @property
    def name(self) -> str:
        return "Timer"

    @property
    def schema(self) -> Dict:
        return {
            "name": "Timer",
            "description": (
                "Check how much time has elapsed since you started this Kaggle competition run. "
                "Use this to manage your time budget and prioritize remaining work."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }

    async def execute(self, input: Dict) -> Dict:
        """Get elapsed time"""
        try:
            start_time = self.get_start_time()
            current_time = time.time()
            elapsed = current_time - start_time

            # Format as hours, minutes, seconds
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)

            time_str = f"{hours}h {minutes}m {seconds}s"

            output = (
                f"⏱️ Elapsed time: {time_str}\n"
                f"   Total seconds: {elapsed:.1f}s\n"
                f"   Started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
            )

            return {
                "content": output,
                "is_error": False,
                "debug_summary": f"Elapsed: {time_str}"
            }
        except Exception as e:
            return {
                "content": f"Error getting elapsed time: {str(e)}",
                "is_error": True
            }
