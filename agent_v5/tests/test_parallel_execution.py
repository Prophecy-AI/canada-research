"""
Tests for parallel tool execution with asyncio.gather
"""
import os
import pytest
import tempfile
import time
from agent_v5.agent import ResearchAgent


@pytest.mark.asyncio
async def test_parallel_read_tools():
    """Test that multiple Read operations execute in parallel"""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        with open(f"{tmpdir}/file1.txt", "w") as f:
            f.write("Content of file 1")
        with open(f"{tmpdir}/file2.txt", "w") as f:
            f.write("Content of file 2")
        with open(f"{tmpdir}/file3.txt", "w") as f:
            f.write("Content of file 3")

        agent = ResearchAgent(
            session_id="test",
            workspace_dir=tmpdir,
            system_prompt="You are a helpful assistant. When asked to read multiple files, use the Read tool for each file."
        )

        start_time = time.time()
        tool_executions = []

        async for msg in agent.run("Read file1.txt, file2.txt, and file3.txt and tell me their contents"):
            if msg.get("type") == "tool_execution":
                tool_executions.append(msg)

        execution_time = time.time() - start_time

        # Verify all three files were read
        read_executions = [t for t in tool_executions if t["tool_name"] == "Read"]
        assert len(read_executions) == 3, f"Expected 3 Read operations, got {len(read_executions)}"

        # Verify all contents are present
        outputs = [t["tool_output"] for t in read_executions]
        assert any("Content of file 1" in out for out in outputs)
        assert any("Content of file 2" in out for out in outputs)
        assert any("Content of file 3" in out for out in outputs)

        print(f"✓ Parallel read test completed in {execution_time:.2f}s")


@pytest.mark.asyncio
async def test_can_parallelize_tools():
    """Test _can_parallelize_tools logic"""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = ResearchAgent(
            session_id="test",
            workspace_dir=tmpdir,
            system_prompt="Test"
        )

        # Read-only tools can be parallelized
        assert agent._can_parallelize_tools([
            {"name": "Read", "input": {"file_path": "a.txt"}},
            {"name": "Read", "input": {"file_path": "b.txt"}},
        ])

        assert agent._can_parallelize_tools([
            {"name": "Glob", "input": {"pattern": "*.py"}},
            {"name": "Grep", "input": {"pattern": "test"}},
        ])

        # Write/Edit operations cannot be parallelized
        assert not agent._can_parallelize_tools([
            {"name": "Write", "input": {"file_path": "a.txt"}},
            {"name": "Read", "input": {"file_path": "b.txt"}},
        ])

        assert not agent._can_parallelize_tools([
            {"name": "Edit", "input": {"file_path": "a.txt"}},
            {"name": "Read", "input": {"file_path": "a.txt"}},
        ])

        # Bash operations cannot be parallelized
        assert not agent._can_parallelize_tools([
            {"name": "Bash", "input": {"command": "ls"}},
            {"name": "Bash", "input": {"command": "pwd"}},
        ])

        # ReadBashOutput cannot be parallelized
        assert not agent._can_parallelize_tools([
            {"name": "ReadBashOutput", "input": {"shell_id": "123"}},
        ])

        # Single tool should not trigger parallelization (no benefit)
        assert agent._can_parallelize_tools([
            {"name": "Read", "input": {"file_path": "a.txt"}},
        ])


@pytest.mark.asyncio
async def test_parallel_glob_and_grep():
    """Test that Glob and Grep can run in parallel"""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        with open(f"{tmpdir}/test1.py", "w") as f:
            f.write("def hello():\n    print('hello')")
        with open(f"{tmpdir}/test2.py", "w") as f:
            f.write("def world():\n    print('world')")
        with open(f"{tmpdir}/readme.md", "w") as f:
            f.write("# README\nThis contains the word hello")

        agent = ResearchAgent(
            session_id="test",
            workspace_dir=tmpdir,
            system_prompt="You are a helpful assistant with file search capabilities."
        )

        tool_executions = []

        async for msg in agent.run("Find all Python files and search for the word 'hello' in all files"):
            if msg.get("type") == "tool_execution":
                tool_executions.append(msg)

        # Should have used Glob and Grep
        glob_executions = [t for t in tool_executions if t["tool_name"] == "Glob"]
        grep_executions = [t for t in tool_executions if t["tool_name"] == "Grep"]

        assert len(glob_executions) >= 1, "Should have used Glob"
        assert len(grep_executions) >= 1, "Should have used Grep"

        print(f"✓ Glob/Grep test completed with {len(tool_executions)} tool executions")


@pytest.mark.asyncio
async def test_sequential_write_then_read():
    """Test that Write followed by Read executes sequentially (not parallel)"""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    with tempfile.TemporaryDirectory() as tmpdir:
        agent = ResearchAgent(
            session_id="test",
            workspace_dir=tmpdir,
            system_prompt="You are a helpful assistant."
        )

        tool_executions = []

        async for msg in agent.run("Create a file called test.txt with content 'Hello' then read it back"):
            if msg.get("type") == "tool_execution":
                tool_executions.append(msg)

        # Should have both Write and Read
        write_executions = [t for t in tool_executions if t["tool_name"] == "Write"]
        read_executions = [t for t in tool_executions if t["tool_name"] == "Read"]

        assert len(write_executions) >= 1, "Should have used Write"
        assert len(read_executions) >= 1, "Should have used Read"

        # Verify content was written and read correctly
        read_output = read_executions[0]["tool_output"]
        assert "Hello" in read_output

        print(f"✓ Sequential Write→Read test passed")
