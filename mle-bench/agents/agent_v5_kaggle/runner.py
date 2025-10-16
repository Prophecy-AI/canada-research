"""
Bridge between mle-bench environment and agent_v5 KaggleAgent
"""
import asyncio
import json
import json
import os
from pathlib import Path
import sys
import time

# Add agent_v5 to path
AGENT_DIR = os.environ.get('AGENT_DIR', '/home/agent')
sys.path.insert(0, AGENT_DIR)

from kaggle_agent import KaggleAgent
from debug import log, log_tool_call


async def main():
    """Run Kaggle competition agent"""

    # Get environment variables from mle-bench
    data_dir = "/home/data"
    code_dir = os.environ.get('CODE_DIR', '/home/code')
    submission_dir = os.environ.get('SUBMISSION_DIR', '/home/submission')
    logs_dir = os.environ.get('LOGS_DIR', '/home/logs')
    instructions_path = "/home/instructions.txt"
    competition_id = os.environ.get('COMPETITION_ID', 'unknown')

    # Start global timer
    start_time = time.time()

    log(f"🏆 Starting Kaggle Agent for competition: {competition_id}")
    log(f"📊 Data: {data_dir}")
    log(f"💻 Workspace: {code_dir}")
    log(f"📤 Submission: {submission_dir}")
    log(f"⏱️  Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    # Create directories
    Path(code_dir).mkdir(exist_ok=True, parents=True)
    Path(submission_dir).mkdir(exist_ok=True, parents=True)
    Path(logs_dir).mkdir(exist_ok=True, parents=True)

    # Create agent
    agent = KaggleAgent(
        session_id=competition_id,
        workspace_dir=code_dir,
        data_dir=data_dir,
        submission_dir=submission_dir,
        instructions_path=instructions_path
    )

    # Store start time for agent to access
    agent.start_time = start_time

    # Security handled by Docker container isolation

    # Initial message to agent
    initial_message = (
        f"You are competing in the Kaggle competition: {competition_id}\n\n"
        f"Your goal: Analyze the data in {data_dir}/, build a machine learning model, "
        f"and create a valid submission file at {submission_dir}/submission.csv\n\n"
        f"Start by reading the competition instructions at {instructions_path} and "
        f"exploring the data files."
    )

    log("→ Starting agent run")

    # Run agent (blocking, stream to console)
    full_response = []
    try:
        async for message in agent.run(initial_message):
            if message.get("type") == "text_delta":
                text = message["text"]
                print(text, end="", flush=True)
                full_response.append(text)
            elif message.get("type") == "tool_execution":
                log_tool_call(message)
    except Exception as e:
        log(f"❌ Agent error: {e}", 2)
        print(f"\n\nERROR: {e}\n", flush=True)
        await agent.cleanup()  # Cleanup on error
        sys.exit(1)
    finally:
        # CRITICAL: Cleanup background processes to prevent leaks
        await agent.cleanup()

    print("\n")

    # Calculate and log total runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    hours = int(total_runtime // 3600)
    minutes = int((total_runtime % 3600) // 60)
    seconds = int(total_runtime % 60)

    log("✓ Agent run complete")
    log(f"⏱️  Total runtime: {hours}h {minutes}m {seconds}s ({total_runtime:.1f}s)", 1)

    # Check if submission was created
    submission_path = Path(submission_dir) / "submission.csv"
    if submission_path.exists():
        log(f"✅ Submission created: {submission_path}")
        log(f"   Size: {submission_path.stat().st_size} bytes")

        # Read first few lines
        try:
            with open(submission_path, 'r') as f:
                lines = f.readlines()[:5]
                log(f"   Preview:\n{''.join(lines)}")
        except Exception as e:
            log(f"   Could not preview: {e}")

        return 0
    else:
        log("❌ WARNING: No submission file found!", 2)
        log(f"   Expected at: {submission_path}")
        log(f"   Contents of {submission_dir}:")
        for item in Path(submission_dir).iterdir():
            log(f"     - {item.name}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
