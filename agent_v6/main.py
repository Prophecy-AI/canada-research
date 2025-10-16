import asyncio
import os
import sys
from pathlib import Path

AGENT_DIR = os.environ.get('AGENT_DIR', '/home/agent')
sys.path.insert(0, AGENT_DIR)

from agent_v6.orchestrator import Orchestrator


async def main():
    competition_id = os.environ.get('COMPETITION_ID', 'unknown')
    data_dir = "/home/data"
    submission_dir = os.environ.get('SUBMISSION_DIR', '/home/submission')
    code_dir = os.environ.get('CODE_DIR', '/home/code')
    instructions_path = "/home/instructions.txt"
    
    try:
        instructions = Path(instructions_path).read_text()
    except Exception as e:
        print(f"Error reading instructions: {e}")
        instructions = "Competition instructions not available"
    
    print(f"Starting Agent V6 for competition: {competition_id}")
    print(f"Data: {data_dir}")
    print(f"Workspace: {code_dir}")
    print(f"Submission: {submission_dir}")
    
    orchestrator = Orchestrator(
        competition_id=competition_id,
        data_dir=data_dir,
        submission_dir=submission_dir,
        workspace_dir=code_dir,
        instructions=instructions
    )
    
    await orchestrator.run()
    
    submission_path = Path(submission_dir) / "submission.csv"
    if submission_path.exists():
        print(f"✓ Submission created: {submission_path}")
        print(f"  Size: {submission_path.stat().st_size} bytes")
        return 0
    else:
        print(f"❌ No submission file found at {submission_path}")
        print(f"Contents of {submission_dir}:")
        for item in Path(submission_dir).iterdir():
            print(f"  - {item.name}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
