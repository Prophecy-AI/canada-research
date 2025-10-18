import asyncio
from pathlib import Path
from typing import Dict
from agent_v6.agent import Agent
from agent_v6.tools import ToolRegistry, BashTool, ReadTool, WriteTool
from agent_v6.prompts import format_worker_prompt


class Worker:
    def __init__(self, experiment_spec: Dict, workspace_dir: str, data_dir: str, eda_context: str):
        self.experiment_spec = experiment_spec
        self.workspace_dir = Path(workspace_dir)
        self.data_dir = Path(data_dir)
        self.eda_context = eda_context
        
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        tools = ToolRegistry(str(self.workspace_dir))
        tools.register(BashTool(str(self.workspace_dir)))
        tools.register(ReadTool(str(self.workspace_dir)))
        tools.register(WriteTool(str(self.workspace_dir)))
        
        system_prompt = format_worker_prompt(
            spec=experiment_spec,
            data_dir=data_dir,
            workspace_dir=workspace_dir,
            eda_context=eda_context
        )
        
        self.agent = Agent(workspace_dir, system_prompt, tools)

    async def write_script(self) -> bool:
        exp_id = self.experiment_spec['id']
        
        print(f"  → {exp_id}: Writing train.py...")
        
        try:
            output = await self.agent.run("Write train.py for this experiment.")
            
            train_py = self.workspace_dir / "train.py"
            if train_py.exists():
                # SOTA: Basic code verification before execution
                if not self._verify_code(train_py):
                    print(f"  ⚠️  {exp_id}: Code verification failed - attempting fix...")
                    # Try to fix common issues
                    output = await self.agent.run("Fix the issues in train.py (syntax, imports, VALIDATION_SCORE print)")
                    if not train_py.exists() or not self._verify_code(train_py):
                        print(f"  ❌ {exp_id}: Code verification still failing")
                        return False
                
                print(f"  ✓ {exp_id}: train.py created and verified ({train_py.stat().st_size} bytes)")
                return True
            else:
                print(f"  ❌ {exp_id}: train.py not created")
                return False
        
        except Exception as e:
            print(f"  ❌ {exp_id}: Error writing script - {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _verify_code(self, train_py: Path) -> bool:
        """SOTA: Verify generated code has basic requirements"""
        try:
            code = train_py.read_text()
            
            # Check 1: Syntax check
            try:
                compile(code, str(train_py), 'exec')
            except SyntaxError as e:
                print(f"    ❌ Syntax error: {e}")
                return False
            
            # Check 2: Must print VALIDATION_SCORE (required by orchestrator)
            if 'VALIDATION_SCORE' not in code:
                print(f"    ❌ Missing VALIDATION_SCORE output")
                return False
            
            # That's it - don't check imports or anything else
            # Let the agent decide what libraries to use
            return True
            
        except Exception as e:
            print(f"    ❌ Verification error: {e}")
            return False
