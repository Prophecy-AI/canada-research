import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent_v6.worker import Worker
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


async def test_single_competition(competition_id: str):
    script_dir = Path(__file__).parent
    input_file = script_dir / "experiments_output.json"
    output_dir = script_dir / "worker_outputs" / competition_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'r') as f:
        experiments_data = json.load(f)
    
    if competition_id not in experiments_data:
        print(f"❌ Competition '{competition_id}' not found in experiments_output.json")
        return
    
    data = experiments_data[competition_id]
    eda_text = data.get('EDA', '')
    plan = data.get('Plan', [])
    
    if not eda_text:
        print(f"❌ No EDA found for {competition_id}")
        return
    
    if isinstance(plan, str):
        print(f"❌ Plan is a string (error): {plan}")
        return
    
    if not plan:
        print(f"❌ No experiments in plan")
        return
    
    print(f"\n{'='*60}")
    print(f"Testing Worker for: {competition_id}")
    print(f"Experiments: {len(plan)}")
    print(f"{'='*60}\n")
    
    for idx, exp in enumerate(plan, 1):
        exp_id = exp.get('id', f'exp_{idx}')
        strategy = exp.get('strategy', 'unknown')
        model = exp.get('model', exp.get('models', 'unknown'))
        
        print(f"\n[{idx}/{len(plan)}] Experiment: {exp_id}")
        print(f"  Strategy: {strategy}")
        print(f"  Model: {model}")
        
        exp_dir = output_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        fake_data_dir = "/home/data"
        
        try:
            worker = Worker(
                experiment_spec=exp,
                workspace_dir=str(exp_dir),
                data_dir=fake_data_dir,
                eda_context=eda_text
            )
            
            print(f"  Generating train.py...")
            success = await worker.write_script()
            
            if success:
                train_py = exp_dir / "train.py"
                if train_py.exists():
                    size = train_py.stat().st_size
                    lines = len(train_py.read_text().splitlines())
                    print(f"  ✓ Generated train.py: {size} bytes, {lines} lines")
                    print(f"  Location: {train_py}")
                else:
                    print(f"  ⚠️  train.py not found after worker.write_script()")
            else:
                print(f"  ❌ Worker returned False")
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"All train.py files saved to: {output_dir}")
    print(f"{'='*60}")


async def main():
    competition_id = "aerial-cactus-identification"
    await test_single_competition(competition_id)


if __name__ == "__main__":
    asyncio.run(main())

