import asyncio
import re
from pathlib import Path
from typing import Dict
from agent_v6.agent import Agent
from agent_v6.tools import ToolRegistry, BashTool, ReadTool, WriteTool
from agent_v6.prompts import format_worker_prompt


class Worker:
    def __init__(self, experiment_spec: Dict, workspace_dir: str, data_dir: str):
        self.experiment_spec = experiment_spec
        self.workspace_dir = workspace_dir
        self.data_dir = data_dir
        
        Path(workspace_dir).mkdir(parents=True, exist_ok=True)
        
        tools = ToolRegistry(workspace_dir)
        tools.register(BashTool(workspace_dir))
        tools.register(ReadTool(workspace_dir))
        tools.register(WriteTool(workspace_dir))
        
        system_prompt = format_worker_prompt(
            spec=experiment_spec,
            data_dir=data_dir,
            workspace_dir=workspace_dir
        )
        
        self.agent = Agent(workspace_dir, system_prompt, tools)

    async def run(self) -> Dict:
        exp_id = self.experiment_spec['id']
        
        print(f"  → {exp_id}: Starting...")
        
        try:
            output = await self.agent.run("Execute the experiment as specified.")
            
            score = self._extract_score(output)
            
            if score is None:
                return {
                    "id": self.experiment_spec['id'],
                    "status": "completed_no_score",
                    "score": None,
                    "output": output[:500],
                    "model": self.experiment_spec.get('model'),
                    "hypothesis": self.experiment_spec.get('hypothesis')
                }
            
            return {
                "id": self.experiment_spec['id'],
                "status": "success",
                "score": score,
                "output": output[:500],
                "model": self.experiment_spec.get('model'),
                "hypothesis": self.experiment_spec.get('hypothesis')
            }
        
        except Exception as e:
            return {
                "id": self.experiment_spec['id'],
                "status": "error",
                "score": None,
                "output": str(e)[:500],
                "model": self.experiment_spec.get('model'),
                "hypothesis": self.experiment_spec.get('hypothesis')
            }

    def _extract_score(self, output: str) -> float:
        patterns = [
            r'VALIDATION_SCORE:\s*([0-9.]+)',
            r'validation score:\s*([0-9.]+)',
            r'val_score:\s*([0-9.]+)',
            r'score:\s*([0-9.]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None

