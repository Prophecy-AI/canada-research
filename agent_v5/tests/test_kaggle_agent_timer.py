"""
Integration test for KaggleAgent with Timer tool
"""
import pytest
import time
import tempfile
from pathlib import Path
import sys
import os

# Add mle-bench agent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../mle-bench/agents/agent_v5_kaggle'))

from kaggle_agent import KaggleAgent


@pytest.mark.asyncio
async def test_kaggle_agent_has_timer_tool():
    """Test that KaggleAgent registers timer tool"""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        data_dir = Path(tmpdir) / "data"
        submission_dir = Path(tmpdir) / "submission"
        instructions = Path(tmpdir) / "instructions.txt"

        workspace.mkdir()
        data_dir.mkdir()
        submission_dir.mkdir()
        instructions.write_text("Test competition instructions")

        agent = KaggleAgent(
            session_id="test",
            workspace_dir=str(workspace),
            data_dir=str(data_dir),
            submission_dir=str(submission_dir),
            instructions_path=str(instructions)
        )

        # Check timer tool is registered
        assert "Timer" in agent.tools.tools
        timer_tool = agent.tools.tools["Timer"]
        assert timer_tool.name == "Timer"


@pytest.mark.asyncio
async def test_kaggle_agent_timer_tracks_elapsed_time():
    """Test that timer tool tracks elapsed time correctly"""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        data_dir = Path(tmpdir) / "data"
        submission_dir = Path(tmpdir) / "submission"
        instructions = Path(tmpdir) / "instructions.txt"

        workspace.mkdir()
        data_dir.mkdir()
        submission_dir.mkdir()
        instructions.write_text("Test competition instructions")

        # Set start time to 5 seconds ago
        start_time = time.time() - 5.0

        agent = KaggleAgent(
            session_id="test",
            workspace_dir=str(workspace),
            data_dir=str(data_dir),
            submission_dir=str(submission_dir),
            instructions_path=str(instructions)
        )

        # Override start time
        agent.start_time = start_time

        # Execute timer tool
        timer_tool = agent.tools.tools["Timer"]
        result = await timer_tool.execute({})

        assert result["is_error"] is False
        assert "Elapsed time:" in result["content"]

        # Should show at least 5 seconds elapsed
        seconds_line = [line for line in result["content"].split('\n') if "Total seconds:" in line][0]
        seconds_str = seconds_line.split(": ")[1].replace("s", "")
        elapsed = float(seconds_str)

        assert elapsed >= 5.0, f"Expected at least 5.0 seconds, got {elapsed}"
