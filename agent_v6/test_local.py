import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from agent_v6.orchestrator import Orchestrator

load_dotenv()


async def main():
    test_dir = Path("test_competition")
    test_dir.mkdir(exist_ok=True)
    
    data_dir = test_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    submission_dir = test_dir / "submission"
    submission_dir.mkdir(exist_ok=True)
    
    workspace_dir = test_dir / "workspace"
    workspace_dir.mkdir(exist_ok=True)
    
    instructions = """This is a simple binary classification competition.

Dataset:
- train.csv: Training data with features and target column 'Transported'
- test.csv: Test data (same features, no target)
- sample_submission.csv: Format for submission

Evaluation: Accuracy

Goal: Predict whether passengers were transported to another dimension."""
    
    (data_dir / "train.csv").write_text("""PassengerId,Age,Fare,Transported
1,25,50.0,True
2,30,100.0,False
3,22,75.0,True
4,35,120.0,False
5,28,80.0,True""")
    
    (data_dir / "test.csv").write_text("""PassengerId,Age,Fare
6,26,60.0
7,32,90.0""")
    
    (data_dir / "sample_submission.csv").write_text("""PassengerId,Transported
6,False
7,False""")
    
    print("=" * 60)
    print("AGENT V6 LOCAL TEST")
    print("=" * 60)
    print(f"Data: {data_dir}")
    print(f"Workspace: {workspace_dir}")
    print(f"Submission: {submission_dir}")
    print("=" * 60)
    
    orchestrator = Orchestrator(
        competition_id="test-competition",
        data_dir=str(data_dir),
        submission_dir=str(submission_dir),
        workspace_dir=str(workspace_dir),
        instructions=instructions
    )
    
    await orchestrator.run(max_rounds=2)
    
    submission_path = submission_dir / "submission.csv"
    if submission_path.exists():
        print("\n" + "=" * 60)
        print("✓ SUCCESS: Submission created!")
        print("=" * 60)
        print(f"Location: {submission_path}")
        print(f"Contents:\n{submission_path.read_text()}")
        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ FAILURE: No submission created")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Run: export ANTHROPIC_API_KEY='your-key-here'")
        exit(1)
    
    exit_code = asyncio.run(main())
    exit(exit_code)

