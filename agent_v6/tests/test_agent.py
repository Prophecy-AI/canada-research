"""
Test IDEAgent with real GPT-5 API calls

WARNING: These tests make real API calls and will cost money.
"""
import os
import tempfile
import shutil
from pathlib import Path

import pytest

from agent_v6.agent import IDEAgent


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing"""
    temp_dir = tempfile.mkdtemp(prefix="test_agent_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_agent_basic_interaction(temp_workspace):
    """Test basic agent interaction with real GPT-5 API"""
    # Skip if no API key
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    system_prompt = """You are a helpful IDE agent. You can:
- Read and write files
- Execute code in Jupyter notebooks
- Run scripts in background

Be concise and helpful."""

    agent = IDEAgent(
        session_id="test_basic",
        workspace_dir=temp_workspace,
        system_prompt=system_prompt,
        enable_memory_compaction=False  # Disable for short test
    )

    # Create a file for the agent to read
    test_file = Path(temp_workspace, "data.txt")
    test_file.write_text("Hello from test file!")

    # Test: Ask agent to read the file
    messages = []
    async for msg in agent.run("Read the file data.txt and tell me what it says"):
        messages.append(msg)

    # Verify agent responded
    text_parts = [m["text"] for m in messages if m.get("type") == "text_delta"]
    full_text = "".join(text_parts)

    assert "Hello from test file" in full_text or "test file" in full_text.lower()

    print("\n" + "="*70)
    print("AGENT RESPONSE:")
    print("="*70)
    print(full_text)
    print("="*70)

    # Cleanup
    await agent.cleanup()


@pytest.mark.asyncio
async def test_agent_file_operations(temp_workspace):
    """Test agent can create and modify files"""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    system_prompt = "You are a helpful IDE agent. You can read, write, and edit files."

    agent = IDEAgent(
        session_id="test_files",
        workspace_dir=temp_workspace,
        system_prompt=system_prompt,
        enable_memory_compaction=False
    )

    # Ask agent to create a file
    async for msg in agent.run("Create a file called hello.txt with the content 'Hello World!'"):
        pass

    # Verify file exists
    hello_file = Path(temp_workspace, "hello.txt")
    assert hello_file.exists()
    assert "Hello World" in hello_file.read_text()

    print("\n✓ Agent successfully created file")

    # Cleanup
    await agent.cleanup()


@pytest.mark.asyncio
async def test_agent_multi_turn(temp_workspace):
    """Test multi-turn conversation"""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    system_prompt = "You are a helpful IDE agent."

    agent = IDEAgent(
        session_id="test_multiturn",
        workspace_dir=temp_workspace,
        system_prompt=system_prompt,
        enable_memory_compaction=False
    )

    # Turn 1: Create file
    async for msg in agent.run("Create a file test.py with: print('hello')"):
        pass

    assert Path(temp_workspace, "test.py").exists()

    # Turn 2: Read the file back
    response_parts = []
    async for msg in agent.run("Read test.py and tell me what it contains"):
        if msg.get("type") == "text_delta":
            response_parts.append(msg["text"])

    response = "".join(response_parts)
    assert "print" in response.lower() or "hello" in response.lower()

    print(f"\n✓ Multi-turn conversation successful")
    print(f"  Conversation history: {len(agent.conversation_history)} messages")

    # Cleanup
    await agent.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
