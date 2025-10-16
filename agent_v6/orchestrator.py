import asyncio
import json
import re
import os
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
        self.round_history = []
        
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
        
        context = f"{self.eda_summary}"
        if self.round_history:
            context += f"\n\n**PREVIOUS EXPERIMENTS (DO NOT REPEAT):**"
            for round_data in self.round_history:
                for exp in round_data['experiments']:
                    context += f"\n- {exp['model']} (score: {exp.get('score', 'N/A')}): {exp.get('hypothesis', '')}"
        if self.best_experiment:
            context += f"\n\n**Current best:** {self.best_experiment['model']} scored {self.best_score}"
        
        prompt = format_planning_prompt(
            competition_id=self.competition_id,
            context=context,
            round_num=self.round_num,
            best_score=self.best_score
        )
        
        agent = Agent(str(plan_workspace), prompt, tools)
        output = await agent.run("Output JSON array of experiments")
        
        return self._parse_experiments(output)

    def _parse_experiments(self, output: str) -> List[Dict]:
        json_match = re.search(r'\[[\s\S]*\]', output)
        if not json_match:
            print(f"\n⚠️  No JSON array found in planning output")
            print(f"Output preview: {output[:500]}")
            return []
        
        try:
            experiments = json.loads(json_match.group(0))
            if not isinstance(experiments, list):
                print(f"\n⚠️  JSON is not a list")
                return []
            
            if not experiments:
                print(f"\n⚠️  Empty experiment list")
                return []
            
            print(f"\n✓ Generated {len(experiments)} experiments:")
            for exp in experiments:
                print(f"  • {exp.get('id', '?')}: {exp.get('model', '?')} - {exp.get('hypothesis', '?')[:60]}...")
            
            return experiments
        except json.JSONDecodeError as e:
            print(f"\n⚠️  JSON parse error: {e}")
            json_str = json_match.group(0)
            print(f"JSON preview: {json_str[:500]}")
            return []

    async def _run_experiments(self, experiments: List[Dict]) -> List[Dict]:
        import time
        start_time = time.time()
        print(f"\n→ Starting {len(experiments)} experiments in parallel...")
        
        tasks = []
        for exp in experiments:
            exp_workspace = Path(self.workspace_dir) / "experiments" / exp['id']
            tasks.append(self._run_single_experiment(exp, exp_workspace))
        
        training_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        elapsed = time.time() - start_time
        print(f"\n✓ All {len(experiments)} experiments completed in {elapsed:.1f}s")
        
        formatted_results = []
        for i, exp in enumerate(experiments):
            if isinstance(training_results[i], dict):
                result = training_results[i]
                formatted_results.append(result)
                
                status_icon = "✓" if result['status'] == 'success' else "⚠️"
                score_str = f"{result['score']:.4f}" if result.get('score') is not None else "N/A"
                print(f"  {status_icon} {result['id']}: {result['status'].upper()} - Score: {score_str}")
                
                if result['status'] == 'success' and result['score'] is not None:
                    if result['score'] > self.best_score:
                        self.best_score = result['score']
                        exp_workspace = Path(self.workspace_dir) / "experiments" / result['id']
                        self.best_experiment = {
                            "id": result['id'],
                            "model": result['model'],
                            "workspace": str(exp_workspace)
                        }
                        print(f"    🏆 New best score!")
            else:
                formatted_results.append({
                    "id": exp['id'],
                    "status": "error",
                    "score": None,
                    "output": str(training_results[i])[:500],
                    "model": exp.get('model'),
                    "hypothesis": exp.get('hypothesis')
                })
                print(f"  ❌ {exp['id']}: ERROR")
        
        return formatted_results
    
    async def _run_single_experiment(self, exp: Dict, workspace: Path) -> Dict:
        exp_id = exp['id']
        workspace.mkdir(parents=True, exist_ok=True)
        
        try:
            worker = Worker(exp, str(workspace), self.data_dir, self.eda_summary)
            success = await worker.write_script()
            
            if not success:
                return {
                    "id": exp_id,
                    "status": "error",
                    "score": None,
                    "output": "Failed to write train.py",
                    "model": exp.get('model'),
                    "hypothesis": exp.get('hypothesis')
                }
            
            return await self._run_training(exp, workspace)
        
        except Exception as e:
            return {
                "id": exp_id,
                "status": "error",
                "score": None,
                "output": str(e)[:500],
                "model": exp.get('model'),
                "hypothesis": exp.get('hypothesis')
            }
    
    async def _run_training(self, exp: Dict, workspace: Path) -> Dict:
        exp_id = exp['id']
        train_py = workspace / "train.py"
        
        if not train_py.exists():
            return {
                "id": exp_id,
                "status": "error",
                "score": None,
                "output": "train.py not found",
                "model": exp.get('model'),
                "hypothesis": exp.get('hypothesis')
            }
        
        try:
            print(f"  → {exp_id}: Starting training...")
            
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            process = await asyncio.create_subprocess_shell(
                f"cd {workspace} && python train.py 2>&1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env
            )
            
            output_lines = []
            
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                line_str = line.decode().strip()
                output_lines.append(line_str)
                
                if line_str:
                    print(f"  [{exp_id}] {line_str}")
            
            await process.wait()
            output = "\n".join(output_lines)
            
            error_log = workspace / "train_error.log"
            error_log.write_text(output)
            
            if process.returncode != 0:
                print(f"\n  ❌ {exp_id}: FAILED (exit code {process.returncode})")
                print(f"  Last 10 lines:")
                for line in output_lines[-10:]:
                    print(f"     {line}")
                return {
                    "id": exp_id,
                    "status": "error",
                    "score": None,
                    "output": output[-2000:] if len(output) > 2000 else output,
                    "model": exp.get('model'),
                    "hypothesis": exp.get('hypothesis')
                }
            
            import re
            matches = re.findall(r"(?:VALIDATION_SCORE|Validation Score|Val Score|Best Validation|Final Validation|Validation Accuracy|Val Acc|Val AUC|Val Kappa|Val LogLoss|Validation Loss)[:=\s]+(\d+\.?\d*)", output, re.IGNORECASE)
            score = float(matches[-1]) if matches else None
            
            if score is None:
                print(f"\n  ⚠️ {exp_id}: No VALIDATION_SCORE found")
                print(f"  Last 10 lines of output:")
                for line in output_lines[-10:]:
                    print(f"     {line}")
                return {
                    "id": exp_id,
                    "status": "no_score",
                    "score": None,
                    "output": output[-2000:] if len(output) > 2000 else output,
                    "model": exp.get('model'),
                    "hypothesis": exp.get('hypothesis')
                }
            
            print(f"\n  ✓ {exp_id}: VALIDATION_SCORE: {score:.6f}")
            return {
                "id": exp_id,
                "status": "success",
                "score": score,
                "output": output[-1000:] if len(output) > 1000 else output,
                "model": exp.get('model'),
                "hypothesis": exp.get('hypothesis'),
                "workspace": str(workspace)
            }
        
        except Exception as e:
            print(f"\n  ❌ {exp_id}: Exception - {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "id": exp_id,
                "status": "error",
                "score": None,
                "output": str(e)[:500],
                "model": exp.get('model'),
                "hypothesis": exp.get('hypothesis')
            }

    async def _run_analysis(self, results: List[Dict]) -> str:
        self.round_history.append({
            'round': self.round_num,
            'experiments': results
        })
        
        for r in results:
            if r.get('status') == 'success' and r.get('score'):
                score = r['score']
                if score > self.best_score:
                    self.best_score = score
                    self.best_experiment = r
                    print(f"\n    🏆 New best score!")
        
        plan_workspace = Path(self.workspace_dir) / "planning"
        
        tools = ToolRegistry(str(plan_workspace))
        
        results_str = "\n".join([
            f"{r['id']} ({r.get('model', '?')}): Score={r.get('score', 'N/A')}"
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
        decision = await agent.run("Output decision")
        
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

