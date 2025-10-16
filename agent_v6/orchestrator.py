import asyncio
import json
import re
from pathlib import Path
from typing import List, Dict
from agent_v6.agent import Agent
from agent_v6.worker import Worker
from agent_v6.tools import ToolRegistry, BashTool, ReadTool, WriteTool
from agent_v6.prompts import (
    format_eda_prompt,
    format_planning_prompt,
    format_analysis_prompt,
    format_submission_prompt
)


class Orchestrator:
    def __init__(
        self,
        competition_id: str,
        data_dir: str,
        submission_dir: str,
        workspace_dir: str,
        instructions_path: str
    ):
        self.competition_id = competition_id
        self.data_dir = data_dir
        self.submission_dir = submission_dir
        self.workspace_dir = workspace_dir
        self.instructions_path = instructions_path
        
        self.best_score = 0.0
        self.best_experiment = None
        self.eda_summary = ""
        self.round_num = 0
        
        Path(workspace_dir).mkdir(parents=True, exist_ok=True)
        Path(submission_dir).mkdir(parents=True, exist_ok=True)

    async def run(self, max_rounds: int = 5):
        print("\n" + "="*60)
        print("PHASE 1: EDA")
        print("="*60)
        await self._run_eda()
        
        for round_num in range(1, max_rounds + 1):
            self.round_num = round_num
            
            print("\n" + "="*60)
            print(f"ROUND {round_num}: PLANNING")
            print("="*60)
            experiments = await self._run_planning()
            if not experiments:
                print("⚠️  No experiments generated, stopping")
                break
            
            print("\n" + "="*60)
            print(f"ROUND {round_num}: PARALLEL EXECUTION ({len(experiments)} experiments)")
            print("="*60)
            results = await self._run_experiments(experiments)
            
            print("\n" + "="*60)
            print(f"ROUND {round_num}: ANALYSIS")
            print("="*60)
            decision = await self._run_analysis(results)
            
            if "SUBMIT" in decision:
                print("\n✓ Decision: SUBMIT")
                break
            else:
                print("\n→ Decision: CONTINUE to next round")
        
        print("\n" + "="*60)
        print("FINAL: SUBMISSION")
        print("="*60)
        await self._run_submission()

    async def _run_eda(self):
        eda_workspace = Path(self.workspace_dir) / "eda"
        eda_workspace.mkdir(exist_ok=True)
        
        tools = ToolRegistry(str(eda_workspace))
        tools.register(BashTool(str(eda_workspace)))
        tools.register(ReadTool(str(eda_workspace)))
        tools.register(WriteTool(str(eda_workspace)))
        
        prompt = format_eda_prompt(
            competition_id=self.competition_id,
            data_dir=self.data_dir,
            instructions_path=self.instructions_path
        )
        
        agent = Agent(str(eda_workspace), prompt, tools)
        self.eda_summary = await agent.run(
            f"Analyze the competition data in {self.data_dir}"
        )

    async def _run_planning(self) -> List[Dict]:
        plan_workspace = Path(self.workspace_dir) / "planning"
        plan_workspace.mkdir(exist_ok=True)
        
        tools = ToolRegistry(str(plan_workspace))
        tools.register(BashTool(str(plan_workspace)))
        tools.register(ReadTool(str(plan_workspace)))
        tools.register(WriteTool(str(plan_workspace)))
        
        context = f"EDA Summary:\n{self.eda_summary}\n"
        if self.best_experiment:
            context += f"\nPrevious best: {self.best_experiment['model']} scored {self.best_score}"
        
        prompt = format_planning_prompt(
            competition_id=self.competition_id,
            context=context,
            round_num=self.round_num,
            best_score=self.best_score,
            num_experiments=3
        )
        
        agent = Agent(str(plan_workspace), prompt, tools)
        output = await agent.run("Plan experiments for this round")
        
        return self._parse_experiments(output)

    def _parse_experiments(self, output: str) -> List[Dict]:
        json_match = re.search(r'\[[\s\S]*\]', output)
        if not json_match:
            return []
        
        try:
            experiments = json.loads(json_match.group(0))
            if not isinstance(experiments, list):
                return []
            
            print(f"\n✓ Generated {len(experiments)} experiments:")
            for exp in experiments:
                print(f"  • {exp.get('id', '?')}: {exp.get('model', '?')} - {exp.get('hypothesis', '?')[:60]}...")
            
            return experiments
        except json.JSONDecodeError:
            return []

    async def _run_experiments(self, experiments: List[Dict]) -> List[Dict]:
        workers = []
        for exp in experiments:
            exp_workspace = Path(self.workspace_dir) / "experiments" / exp['id']
            worker = Worker(exp, str(exp_workspace), self.data_dir)
            workers.append(worker)
        
        print(f"\n→ Starting {len(workers)} workers in parallel...")
        results = await asyncio.gather(
            *[w.run() for w in workers],
            return_exceptions=True
        )
        
        print(f"\n✓ All workers completed")
        
        formatted_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                formatted_results.append({
                    "id": experiments[i]['id'],
                    "status": "error",
                    "score": None,
                    "output": str(result)[:500],
                    "model": experiments[i].get('model'),
                    "hypothesis": experiments[i].get('hypothesis')
                })
                print(f"  ❌ {experiments[i]['id']}: ERROR - {str(result)[:100]}")
            else:
                formatted_results.append(result)
                
                status_icon = "✓" if result['status'] == 'success' else "⚠️"
                score_str = f"{result['score']:.4f}" if result.get('score') is not None else "N/A"
                print(f"  {status_icon} {result['id']}: {result['status'].upper()} - Score: {score_str}")
                
                if result['status'] == 'success' and result['score'] is not None:
                    if result['score'] > self.best_score:
                        self.best_score = result['score']
                        self.best_experiment = {
                            "id": result['id'],
                            "model": result['model'],
                            "workspace": str(Path(self.workspace_dir) / "experiments" / result['id'])
                        }
                        print(f"    🏆 New best score!")
        
        return formatted_results

    async def _run_analysis(self, results: List[Dict]) -> str:
        plan_workspace = Path(self.workspace_dir) / "planning"
        
        tools = ToolRegistry(str(plan_workspace))
        tools.register(BashTool(str(plan_workspace)))
        tools.register(ReadTool(str(plan_workspace)))
        tools.register(WriteTool(str(plan_workspace)))
        
        results_str = "\n".join([
            f"Experiment {r['id']} ({r.get('model', 'unknown')}): "
            f"Status={r['status']}, Score={r.get('score', 'N/A')}, "
            f"Hypothesis={r.get('hypothesis', 'N/A')}"
            for r in results
        ])
        
        prompt = format_analysis_prompt(
            competition_id=self.competition_id,
            round_num=self.round_num,
            results=results_str,
            best_score=self.best_score,
            metric="accuracy",
            submit_threshold=0.85
        )
        
        agent = Agent(str(plan_workspace), prompt, tools)
        decision = await agent.run("Analyze results and decide next action")
        
        return decision

    async def _run_submission(self):
        if not self.best_experiment:
            return
        
        sub_workspace = Path(self.workspace_dir) / "submission"
        sub_workspace.mkdir(exist_ok=True)
        
        tools = ToolRegistry(str(sub_workspace))
        tools.register(BashTool(str(sub_workspace)))
        tools.register(ReadTool(str(sub_workspace)))
        tools.register(WriteTool(str(sub_workspace)))
        
        prompt = format_submission_prompt(
            competition_id=self.competition_id,
            best_model=self.best_experiment['model'],
            best_workspace=self.best_experiment['workspace'],
            data_dir=self.data_dir,
            submission_dir=self.submission_dir
        )
        
        agent = Agent(str(sub_workspace), prompt, tools)
        await agent.run("Generate final submission")

