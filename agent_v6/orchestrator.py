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
        
        self.best_score = None
        self.best_experiment = None
        self.eda_summary = ""
        self.data_facts = {}
        self.round_num = 0
        self.round_history = []
        self.lower_is_better = False
        self.round_elapsed_time = 0
        self.total_start_time = None
        
        Path(workspace_dir).mkdir(parents=True, exist_ok=True)
        Path(submission_dir).mkdir(parents=True, exist_ok=True)

    async def run(self, max_rounds: int = 5):
        import time
        self.total_start_time = time.time()
        
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
        
        import re
        import json
        
        json_match = re.search(r'DATA_STRUCTURE_FACTS:\s*\n?\s*(\{[^}]+\})', self.eda_summary, re.DOTALL)
        if json_match:
            try:
                self.data_facts = json.loads(json_match.group(1))
                print(f"\n✓ Parsed data structure facts: {self.data_facts}")
                
                metric_dir = self.data_facts.get('metric_direction', 'HIGHER')
                self.lower_is_better = (metric_dir == 'LOWER')
                self.best_score = float('inf') if self.lower_is_better else 0.0
            except:
                print("\n⚠️  Failed to parse DATA_STRUCTURE_FACTS JSON")
                self.data_facts = {}
                self.lower_is_better = False
                self.best_score = 0.0
        else:
            print("\n⚠️  No DATA_STRUCTURE_FACTS found in EDA")
            self.data_facts = {}
            
            if "LOWER is better" in self.eda_summary:
                self.lower_is_better = True
                self.best_score = float('inf')
            elif "HIGHER is better" in self.eda_summary:
                self.lower_is_better = False
                self.best_score = 0.0
            else:
                self.lower_is_better = False
                self.best_score = 0.0
        
        print(f"→ Metric direction: {'LOWER is better' if self.lower_is_better else 'HIGHER is better'}")

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
            print(f"Output length: {len(output)} chars")
            print(f"Output preview: {output[:500]}")
            print(f"Output suffix: ...{output[-200:]}")
            return []
        
        json_str = json_match.group(0)
        
        json_str = re.sub(r'(?<!\\)\n', ' ', json_str)
        json_str = re.sub(r'(?<!\\)\r', ' ', json_str)
        json_str = re.sub(r'(?<!\\)\t', ' ', json_str)
        
        try:
            experiments = json.loads(json_str)
            if not isinstance(experiments, list):
                print(f"\n⚠️  JSON is not a list")
                return []
            
            if not experiments:
                print(f"\n⚠️  Empty experiment list")
                return []
            
            for exp in experiments:
                old_id = exp.get('id', 'unknown')
                exp['id'] = f"r{self.round_num}_{old_id}"
            
            print(f"\n✓ Generated {len(experiments)} experiments:")
            for exp in experiments:
                print(f"  • {exp.get('id', '?')}: {exp.get('model', '?')} - {exp.get('hypothesis', '?')[:60]}...")
            
            return experiments
        except json.JSONDecodeError as e:
            print(f"\n⚠️  JSON parse error: {e}")
            print(f"JSON length: {len(json_str)} chars")
            print(f"JSON preview (first 500): {json_str[:500]}")
            print(f"JSON suffix (last 200): ...{json_str[-200:]}")
            print(f"\n💡 This usually means the LLM response was cut off mid-generation.")
            print(f"   Check if the response hit max_tokens limit or if there was an API error.")
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
        
        self.round_elapsed_time = time.time() - start_time
        print(f"\n✓ All {len(experiments)} experiments completed in {self.round_elapsed_time:.1f}s")
        
        formatted_results = []
        for i, exp in enumerate(experiments):
            if isinstance(training_results[i], dict):
                result = training_results[i]
                formatted_results.append(result)
                
                status_icon = "✓" if result['status'] == 'success' else "⚠️"
                score_str = f"{result['score']:.8f}" if result.get('score') is not None else "N/A"
                print(f"  {status_icon} {result['id']}: {result['status'].upper()} - Score: {score_str}")
                
                if result['status'] == 'success' and result['score'] is not None:
                    is_better = (result['score'] < self.best_score if self.lower_is_better 
                                else result['score'] > self.best_score)
                    if is_better:
                        self.best_score = result['score']
                        exp_workspace = Path(self.workspace_dir) / "experiments" / result['id']
                        self.best_experiment = {
                            "id": result['id'],
                            "model": result['model'],
                            "workspace": str(exp_workspace)
                        }
                        print(f"    🏆 New best score!")
            elif isinstance(training_results[i], Exception):
                print(f"\n  ❌ {exp['id']}: Task raised exception: {training_results[i]}")
                import traceback
                traceback.print_exception(type(training_results[i]), training_results[i], training_results[i].__traceback__)
                formatted_results.append({
                    "id": exp['id'],
                    "status": "error",
                    "score": None,
                    "output": f"Task exception: {str(training_results[i])}",
                    "model": exp.get('model'),
                    "hypothesis": exp.get('hypothesis')
                })
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
            worker = Worker(exp, str(workspace), self.data_dir, self.eda_summary, self.data_facts)
            success = await worker.write_script()
            
            if not success:
                print(f"  ❌ {exp_id}: Worker failed to create train.py (returned False)")
                return {
                    "id": exp_id,
                    "status": "error",
                    "score": None,
                    "output": "Worker.write_script() returned False - check worker output above",
                    "model": exp.get('model'),
                    "hypothesis": exp.get('hypothesis')
                }
            
            return await self._run_training(exp, workspace)
        
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(f"\n  ❌ {exp_id}: Exception during worker setup - {str(e)}")
            traceback.print_exc()
            return {
                "id": exp_id,
                "status": "error",
                "score": None,
                "output": error_msg[:1000],
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
            matches = re.findall(r"VALIDATION_SCORE:\s*(\d+\.?\d*)", output)
            score = float(matches[-1]) if matches else None
            
            val_loss_matches = re.findall(r"VAL_LOSS:\s*(\d+\.?\d*)", output)
            val_loss = float(val_loss_matches[-1]) if val_loss_matches else None
            
            train_loss_matches = re.findall(r"TRAIN_LOSS:\s*(\d+\.?\d*)", output)
            train_loss = float(train_loss_matches[-1]) if train_loss_matches else None
            
            train_time_matches = re.findall(r"TRAIN_TIME:\s*(\d+\.?\d*)", output)
            train_time = float(train_time_matches[-1]) if train_time_matches else None
            
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
            
            metrics_str = f"VALIDATION_SCORE: {score:.8f}"
            if val_loss is not None:
                metrics_str += f", VAL_LOSS: {val_loss:.8f}"
            if train_time is not None:
                metrics_str += f", TIME: {train_time:.1f}s"
            
            print(f"\n  ✓ {exp_id}: {metrics_str}")
            return {
                "id": exp_id,
                "status": "success",
                "score": score,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "train_time": train_time,
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
                is_better = (score < self.best_score if self.lower_is_better 
                            else score > self.best_score)
                if is_better:
                    self.best_score = score
                    self.best_experiment = r
                    print(f"\n    🏆 New best score!")
        
        plan_workspace = Path(self.workspace_dir) / "planning"
        
        tools = ToolRegistry(str(plan_workspace))
        
        results_str = "\n".join([
            f"{r['id']} ({r.get('model', '?')}): Score={r.get('score', 'N/A')}, "
            f"VAL_LOSS={r.get('val_loss', 'N/A')}, TRAIN_TIME={r.get('train_time', 'N/A')}s"
            for r in results
        ])
        
        metric_direction = "LOWER is better" if self.lower_is_better else "HIGHER is better"
        round_time_minutes = self.round_elapsed_time / 60
        
        import time
        cumulative_time_minutes = (time.time() - self.total_start_time) / 60
        
        best_exp_id = self.best_experiment.get('id', 'N/A') if self.best_experiment else 'N/A'
        
        prompt = format_analysis_prompt(
            competition_id=self.competition_id,
            round_num=self.round_num,
            results=results_str,
            best_score=self.best_score,
            best_experiment_id=best_exp_id,
            metric_direction=metric_direction,
            round_time_minutes=round_time_minutes,
            cumulative_time_minutes=cumulative_time_minutes
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

