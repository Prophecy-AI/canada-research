import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent_v6.worker import Worker
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


class WorkerTester:
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def test_all_workers(self):
        with open(self.input_file, 'r') as f:
            experiments_data = json.load(f)
        
        total_competitions = len(experiments_data)
        
        print(f"\n{'='*60}")
        print(f"Starting worker test for {total_competitions} competitions")
        print(f"{'='*60}\n")
        
        results = {}
        
        for idx, (competition_id, data) in enumerate(experiments_data.items(), 1):
            print(f"\n{'='*60}")
            print(f"[{idx}/{total_competitions}] Testing: {competition_id}")
            print(f"{'='*60}")
            
            eda_text = data.get('EDA', '')
            plan = data.get('Plan', [])
            
            if not eda_text:
                print(f"⚠️  No EDA found for {competition_id}, skipping...")
                continue
            
            if isinstance(plan, str):
                print(f"⚠️  Plan is a string (error), skipping...")
                continue
            
            if not plan:
                print(f"⚠️  No experiments in plan, skipping...")
                continue
            
            competition_results = await self.test_competition(
                competition_id, eda_text, plan
            )
            results[competition_id] = competition_results
        
        summary_file = self.output_dir / "worker_test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Results saved to {self.output_dir}")
        print(f"Summary: {summary_file}")
        print(f"{'='*60}")
    
    async def test_competition(self, competition_id: str, eda_text: str, experiments: list) -> dict:
        comp_dir = self.output_dir / competition_id
        comp_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "total_experiments": len(experiments),
            "successful": 0,
            "failed": 0,
            "experiments": []
        }
        
        for exp in experiments:
            exp_id = exp.get('id', 'unknown')
            print(f"\n  Generating code for {exp_id}...")
            
            try:
                success = await self.generate_worker_code(
                    competition_id, exp_id, exp, eda_text, comp_dir
                )
                
                if success:
                    results["successful"] += 1
                    print(f"  ✓ Successfully generated train.py for {exp_id}")
                else:
                    results["failed"] += 1
                    print(f"  ⚠️  Worker returned False for {exp_id}")
                
                results["experiments"].append({
                    "id": exp_id,
                    "success": success,
                    "strategy": exp.get("strategy"),
                    "model": exp.get("model", exp.get("models"))
                })
                
            except Exception as e:
                results["failed"] += 1
                print(f"  ❌ Error generating code for {exp_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                results["experiments"].append({
                    "id": exp_id,
                    "success": False,
                    "error": str(e),
                    "strategy": exp.get("strategy"),
                    "model": exp.get("model", exp.get("models"))
                })
        
        print(f"\n  Summary: {results['successful']}/{results['total_experiments']} successful")
        return results
    
    async def generate_worker_code(
        self, 
        competition_id: str, 
        exp_id: str, 
        experiment: dict, 
        eda_text: str, 
        comp_dir: Path
    ) -> bool:
        exp_dir = comp_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        fake_data_dir = f"/home/data"
        
        worker = Worker(
            experiment_spec=experiment,
            workspace_dir=str(exp_dir),
            data_dir=fake_data_dir,
            eda_context=eda_text
        )
        
        success = await worker.write_script()
        
        if success:
            train_py = exp_dir / "train.py"
            if train_py.exists():
                print(f"    → Generated {train_py.stat().st_size} bytes")
            else:
                print(f"    ⚠️  train.py not found after worker.write_script()")
                return False
        
        return success


async def main():
    script_dir = Path(__file__).parent
    input_file = script_dir / "experiments_output.json"
    output_dir = script_dir / "worker_outputs"
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print(f"   Please run test_planner.py first to generate experiments_output.json")
        return
    
    tester = WorkerTester(str(input_file), str(output_dir))
    await tester.test_all_workers()


if __name__ == "__main__":
    asyncio.run(main())

