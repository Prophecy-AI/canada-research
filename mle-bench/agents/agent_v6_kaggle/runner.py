"""
Bridge between mle-bench environment and agent_v6 KaggleAgent

Based on Operand Quant architecture with Deep-Thinking Ensemble
"""
import asyncio
import os
from pathlib import Path
import sys

# Add agent_v6 to path
AGENT_DIR = os.environ.get('AGENT_DIR', '/home/agent')
sys.path.insert(0, AGENT_DIR)

from kaggle_agent import KaggleAgent
from debug import log, log_tool_call


async def main():
    """Run Kaggle competition agent using Operand Quant architecture"""

    # Get environment variables from mle-bench
    data_dir = "/home/data"
    code_dir = os.environ.get('CODE_DIR', '/home/code')
    submission_dir = os.environ.get('SUBMISSION_DIR', '/home/submission')
    logs_dir = os.environ.get('LOGS_DIR', '/home/logs')
    instructions_path = "/home/instructions.txt"
    competition_id = os.environ.get('COMPETITION_ID', 'unknown')

    log(f"🏆 Starting agent_v6 (Operand Quant) for: {competition_id}")
    log(f"📊 Data: {data_dir}")
    log(f"💻 Workspace: {code_dir}")
    log(f"📤 Submission: {submission_dir}")
    log(f"🔮 Deep-Thinking Ensemble: GPT-5, Claude Opus, Grok-4, Gemini + O3")

    # Create directories
    Path(code_dir).mkdir(exist_ok=True, parents=True)
    Path(submission_dir).mkdir(exist_ok=True, parents=True)
    Path(logs_dir).mkdir(exist_ok=True, parents=True)

    # Create agent with Operand Quant architecture
    # - Single-agent, IDE-based
    # - Deep-Thinking Ensemble (4 models + O3 synthesis)
    # - Non-blocking execution
    # - Jupyter notebook support
    # - Memory compaction
    agent = KaggleAgent(
        session_id=competition_id,
        workspace_dir=code_dir,
        data_dir=data_dir,
        submission_dir=submission_dir,
        instructions_path=instructions_path,
        enable_memory_compaction=True  # Hierarchical compaction for long sessions
    )

    # Security handled by Docker container isolation

    # Initial message to agent
    initial_message = (
        f"You are competing in the Kaggle competition: {competition_id}\n\n"
        f"Your goal: Analyze the data in {data_dir}/, build a machine learning model, "
        f"and create a valid submission file at {submission_dir}/submission.csv\n\n"
        f"Start by reading the competition instructions at {instructions_path} and "
        f"exploring the data files.\n\n"
        f"IMPORTANT: Before writing any training script, read /home/kaggle_competition_strategy.txt "
        f"which contains Grandmaster strategies to avoid common errors and optimize for medals.\n\n"
        f"You have access to the Deep-Thinking Ensemble (ConsultEnsemble tool) - use it for:\n"
        f"1. Initial strategy after data exploration (MANDATORY)\n"
        f"2. Code review before training (MANDATORY)\n"
        f"3. During training for progress checks\n"
        f"4. After training for improvement suggestions"
    )

    log("→ Starting agent run with Deep-Thinking Ensemble enabled")

    # Open log file for writing agent output
    agent_log_path = Path(logs_dir) / "agentic.log"

    # Run agent (blocking, stream to console and log file)
    full_response = []
    try:
        with open(agent_log_path, 'w', buffering=1) as log_file:  # Line buffered
            log_file.write(f"=== agent_v6 (Operand Quant) Run Started: {competition_id} ===\n")
            log_file.write(f"Data: {data_dir}\n")
            log_file.write(f"Workspace: {code_dir}\n")
            log_file.write(f"Submission: {submission_dir}\n")
            log_file.write(f"Ensemble: GPT-5, Claude Opus 4.1, Grok-4, Gemini 2.5 Pro + O3\n")
            log_file.write(f"{'='*60}\n\n")

            async for message in agent.run(initial_message):
                if message.get("type") == "text_delta":
                    text = message["text"]
                    print(text, end="", flush=True)
                    log_file.write(text)  # Write to log file
                    full_response.append(text)
                elif message.get("type") == "tool_execution":
                    tool_log = f"\n[TOOL] {message.get('tool_name', 'unknown')} - {message.get('status', 'unknown')}\n"
                    log_file.write(tool_log)  # Log tool executions
                    log_tool_call(message)

            log_file.write(f"\n\n{'='*60}\n")
            log_file.write(f"=== Agent Run Complete ===\n")
    except Exception as e:
        log(f"❌ Agent error: {e}", 2)
        print(f"\n\nERROR: {e}\n", flush=True)

        # Write error to log file
        try:
            with open(agent_log_path, 'a') as log_file:
                log_file.write(f"\n\nERROR: {e}\n")
        except:
            pass

        await agent.cleanup()  # Cleanup on error
        sys.exit(1)
    finally:
        # CRITICAL: Cleanup background processes and notebook kernels
        await agent.cleanup()

    print("\n")
    log("✓ Agent run complete")

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
