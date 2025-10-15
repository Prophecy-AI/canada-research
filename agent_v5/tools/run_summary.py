"""
RunSummary tool for appending structured run logs
"""
import os
import json
import time
from typing import Dict, Any
from .base import BaseTool


class RunSummaryTool(BaseTool):
    """Append a structured run summary entry to a JSONL log in the workspace."""

    @property
    def name(self) -> str:
        return "RunSummary"

    @property
    def schema(self) -> Dict:
        return {
            "name": "RunSummary",
            "description": (
                "Append a structured run summary entry (JSONL) to the workspace .runs/run_log.jsonl. "
                "Use after training/inference/evaluation to persist hypothesis, action, params, metrics, and artifact paths."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string", "description": "Unique run identifier; generated if omitted"},
                    "phase": {"type": "string", "description": "Phase label (e.g., plan, execute, evaluate, finalize)"},
                    "hypothesis": {"type": "string", "description": "Current hypothesis text"},
                    "action": {"type": "string", "description": "Chosen action (Feature engineering/processing/selection/tuning)"},
                    "model": {"type": "string", "description": "Primary model name/type"},
                    "hyperparameters": {"type": "object", "description": "Hyperparameters dict"},
                    "metrics": {"type": "object", "description": "Metrics dict (e.g., CV scores)"},
                    "artifacts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of artifact file paths saved in workspace"
                    },
                    "notes": {"type": "string", "description": "Free-form notes or decisions"}
                },
                "required": []
            }
        }

    async def execute(self, input: Dict) -> Dict:
        """Append a single JSON line to .runs/run_log.jsonl and also write the latest entry as .runs/latest.json."""
        try:
            # Ensure directories
            runs_dir = os.path.join(self.workspace_dir, ".runs")
            os.makedirs(runs_dir, exist_ok=True)

            # Prepare entry
            now_ts = int(time.time())
            run_id = input.get("run_id") or f"run_{now_ts}"

            entry: Dict[str, Any] = {
                "timestamp": now_ts,
                "run_id": run_id,
                "phase": input.get("phase"),
                "hypothesis": input.get("hypothesis"),
                "action": input.get("action"),
                "model": input.get("model"),
                "hyperparameters": input.get("hyperparameters"),
                "metrics": input.get("metrics"),
                "artifacts": input.get("artifacts"),
                "notes": input.get("notes"),
            }

            # Append to JSONL
            log_path = os.path.join(runs_dir, "run_log.jsonl")
            with open(log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # Also save latest
            latest_path = os.path.join(runs_dir, "latest.json")
            with open(latest_path, "w") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)

            # Truncate summary for logs
            summary = json.dumps({k: entry.get(k) for k in ["run_id", "phase", "model"]}, ensure_ascii=False)
            
            # Remind agent to consult Oracle
            oracle_reminder = (
                "\n\n⚠️  NEXT STEP REQUIRED: Consult Oracle now.\n"
                "Call: Oracle(query=\"I just completed [describe what you did]. "
                "Results: [key metrics]. Should I continue this direction or pivot? Any bugs or issues?\")\n"
                "Oracle will review your full conversation history and catch problems you might have missed."
            )
            
            return {
                "content": f"RunSummary appended to {log_path}. Latest saved to {latest_path}.\n{summary}{oracle_reminder}",
                "is_error": False,
                "debug_summary": summary,
            }
        except Exception as e:
            return {"content": f"Error writing run summary: {str(e)}", "is_error": True}


