import asyncio
from pathlib import Path
from typing import Dict
from agent_v6.agent import Agent
from agent_v6.tools import ToolRegistry, BashTool, ReadTool, WriteTool
from agent_v6.prompts import format_worker_prompt


class Worker:
    def __init__(self, experiment_spec: Dict, workspace_dir: str, data_dir: str, eda_context: str, data_facts: Dict = None):
        self.experiment_spec = experiment_spec
        self.workspace_dir = Path(workspace_dir)
        self.data_dir = Path(data_dir)
        self.eda_context = eda_context
        self.data_facts = data_facts or {}
        
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        tools = ToolRegistry(str(self.workspace_dir))
        tools.register(BashTool(str(self.workspace_dir)))
        tools.register(ReadTool(str(self.workspace_dir)))
        tools.register(WriteTool(str(self.workspace_dir)))
        
        system_prompt = format_worker_prompt(
            spec=experiment_spec,
            data_dir=data_dir,
            workspace_dir=workspace_dir,
            eda_context=eda_context,
            data_facts=self.data_facts
        )
        
        self.agent = Agent(workspace_dir, system_prompt, tools)

    async def write_script(self) -> bool:
        exp_id = self.experiment_spec['id']
        
        print(f"  → {exp_id}: Writing train.py...")
        
        try:
            output = await self.agent.run("Write train.py for this experiment.")
            
            train_py = self.workspace_dir / "train.py"
            if train_py.exists():
                print(f"  ✓ {exp_id}: train.py created ({train_py.stat().st_size} bytes)")
                return True
            else:
                print(f"  ❌ {exp_id}: train.py not created")
                return False
        
        except Exception as e:
            print(f"  ❌ {exp_id}: Error writing script - {str(e)}")
            import traceback
            traceback.print_exc()
            return False
