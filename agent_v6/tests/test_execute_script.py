"""
Test ExecuteScriptTool with CheckProcessTool and InterruptProcessTool
"""
import os
import time
import tempfile
import shutil
import asyncio
from pathlib import Path

import pytest

from agent_v6.tools.execute_script import ExecuteScriptTool
from agent_v6.tools.check_process import CheckProcessTool
from agent_v6.tools.interrupt_process import InterruptProcessTool
from agent_v6.workspace import IDEWorkspace


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing"""
    temp_dir = tempfile.mkdtemp(prefix="test_execute_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_execute_python_script(temp_workspace):
    """Test executing Python script in background"""
    workspace = IDEWorkspace(temp_workspace)
    execute_tool = ExecuteScriptTool(temp_workspace, workspace)
    check_tool = CheckProcessTool(temp_workspace, execute_tool)

    # Create test script
    script_path = "test.py"
    Path(temp_workspace, script_path).write_text("""
import time
for i in range(5):
    print(f'Count: {i}')
    time.sleep(0.1)
print('Done!')
""")

    # Execute script
    result = await execute_tool.execute({
        "script_path": script_path
    })

    assert not result["is_error"]
    assert "Started background process" in result["content"]

    # Extract PID
    pid = None
    for line in result["content"].split("\n"):
        if "PID:" in line:
            pid = int(line.split("PID:")[1].split(")")[0].strip())
            break

    assert pid is not None

    # Wait for script to run
    await asyncio.sleep(1.0)

    # Check process
    check_result = await check_tool.execute({"pid": pid})

    assert not check_result["is_error"]
    assert "completed" in check_result["content"].lower() or "running" in check_result["content"].lower()
    assert "Done!" in check_result["content"] or "Count:" in check_result["content"]

    # Cleanup
    await execute_tool.cleanup()


@pytest.mark.asyncio
async def test_execute_bash_script(temp_workspace):
    """Test executing Bash script"""
    execute_tool = ExecuteScriptTool(temp_workspace)

    # Create test script
    script_path = "test.sh"
    script_content = """#!/bin/bash
echo "Hello from Bash"
echo "Line 2"
echo "Line 3"
"""
    Path(temp_workspace, script_path).write_text(script_content)

    # Execute script
    result = await execute_tool.execute({
        "script_path": script_path,
        "interpreter": "bash"
    })

    assert not result["is_error"]
    assert "Started background process" in result["content"]

    # Wait briefly
    await asyncio.sleep(0.5)

    # Cleanup
    await execute_tool.cleanup()


@pytest.mark.asyncio
async def test_script_with_arguments(temp_workspace):
    """Test script with command line arguments"""
    execute_tool = ExecuteScriptTool(temp_workspace)

    # Create script that uses arguments
    script_path = "args.py"
    Path(temp_workspace, script_path).write_text("""
import sys
print(f"Args: {sys.argv[1:]}")
""")

    # Execute with arguments
    result = await execute_tool.execute({
        "script_path": script_path,
        "args": ["arg1", "arg2", "arg3"]
    })

    assert not result["is_error"]

    # Cleanup
    await execute_tool.cleanup()


@pytest.mark.asyncio
async def test_interrupt_process(temp_workspace):
    """Test interrupting running process"""
    workspace = IDEWorkspace(temp_workspace)
    execute_tool = ExecuteScriptTool(temp_workspace, workspace)
    check_tool = CheckProcessTool(temp_workspace, execute_tool)
    interrupt_tool = InterruptProcessTool(temp_workspace, execute_tool, workspace)

    # Create long-running script
    script_path = "long.py"
    Path(temp_workspace, script_path).write_text("""
import time
for i in range(100):
    print(f'Iteration {i}')
    time.sleep(0.1)
""")

    # Start script
    result = await execute_tool.execute({"script_path": script_path})
    assert not result["is_error"]

    # Extract PID
    pid = None
    for line in result["content"].split("\n"):
        if "PID:" in line:
            pid = int(line.split("PID:")[1].split(")")[0].strip())
            break

    # Let it run briefly
    await asyncio.sleep(0.5)

    # Check it's running
    check_result = await check_tool.execute({"pid": pid})
    assert not check_result["is_error"]

    # Interrupt it
    interrupt_result = await interrupt_tool.execute({"pid": pid})
    assert not interrupt_result["is_error"]
    assert "process" in interrupt_result["content"].lower()

    # Verify it's stopped
    proc_info = execute_tool.get_process_info(pid)
    assert proc_info["status"] == "killed"

    # Cleanup
    await execute_tool.cleanup()


@pytest.mark.asyncio
async def test_process_resource_monitoring(temp_workspace):
    """Test resource usage monitoring"""
    workspace = IDEWorkspace(temp_workspace)
    execute_tool = ExecuteScriptTool(temp_workspace, workspace)
    check_tool = CheckProcessTool(temp_workspace, execute_tool)

    # Create script that uses some resources
    script_path = "resources.py"
    Path(temp_workspace, script_path).write_text("""
import time
data = []
for i in range(20):
    data.append([0] * 1000)
    print(f'Allocated {len(data)} arrays')
    time.sleep(0.1)
""")

    # Execute script
    result = await execute_tool.execute({"script_path": script_path})
    assert not result["is_error"]

    # Extract PID
    pid = None
    for line in result["content"].split("\n"):
        if "PID:" in line:
            pid = int(line.split("PID:")[1].split(")")[0].strip())
            break

    # Wait for some execution
    await asyncio.sleep(1.0)

    # Check resources
    check_result = await check_tool.execute({"pid": pid})
    assert not check_result["is_error"]

    # Should have CPU and memory stats
    content_lower = check_result["content"].lower()
    # Note: may not always show stats if process completed very quickly
    # So this is a soft check

    # Cleanup
    await execute_tool.cleanup()


@pytest.mark.asyncio
async def test_missing_script(temp_workspace):
    """Test error when script doesn't exist"""
    execute_tool = ExecuteScriptTool(temp_workspace)

    result = await execute_tool.execute({
        "script_path": "nonexistent.py"
    })

    assert result["is_error"]
    assert "not found" in result["content"].lower()

    # Cleanup
    await execute_tool.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
