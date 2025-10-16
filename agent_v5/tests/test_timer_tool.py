"""
Tests for TimerTool
"""
import pytest
import time
from pathlib import Path
from agent_v5.tools.timer import TimerTool


@pytest.mark.asyncio
async def test_timer_tool_basic(tmp_path):
    """Test timer tool shows elapsed time"""
    start_time = time.time()

    tool = TimerTool(
        workspace_dir=str(tmp_path),
        get_start_time=lambda: start_time
    )

    # Wait a bit
    time.sleep(0.1)

    result = await tool.execute({})

    assert result["is_error"] is False
    assert "Elapsed time:" in result["content"]
    assert "Total seconds:" in result["content"]
    assert "Started at:" in result["content"]


@pytest.mark.asyncio
async def test_timer_tool_elapsed_time(tmp_path):
    """Test timer tool calculates elapsed time correctly"""
    start_time = time.time()

    tool = TimerTool(
        workspace_dir=str(tmp_path),
        get_start_time=lambda: start_time
    )

    # Wait 1 second
    time.sleep(1.0)

    result = await tool.execute({})

    # Should show at least 1 second elapsed
    assert "Total seconds: " in result["content"]
    seconds_line = [line for line in result["content"].split('\n') if "Total seconds:" in line][0]
    seconds_str = seconds_line.split(": ")[1].replace("s", "")
    elapsed = float(seconds_str)

    assert elapsed >= 1.0, f"Expected at least 1.0 seconds, got {elapsed}"


@pytest.mark.asyncio
async def test_timer_tool_format(tmp_path):
    """Test timer tool formats time as hours/minutes/seconds"""
    # Simulate start time 2 hours ago
    start_time = time.time() - 7260  # 2h 1m ago

    tool = TimerTool(
        workspace_dir=str(tmp_path),
        get_start_time=lambda: start_time
    )

    result = await tool.execute({})

    assert result["is_error"] is False
    # Should show 2h 1m in the output
    assert "2h" in result["content"]
    assert "1m" in result["content"]


@pytest.mark.asyncio
async def test_timer_tool_schema(tmp_path):
    """Test timer tool schema is correct"""
    tool = TimerTool(
        workspace_dir=str(tmp_path),
        get_start_time=lambda: time.time()
    )

    schema = tool.schema
    assert schema["name"] == "Timer"
    assert "elapsed" in schema["description"].lower()
    assert "time" in schema["description"].lower()
    assert schema["input_schema"]["type"] == "object"
    assert schema["input_schema"]["required"] == []
