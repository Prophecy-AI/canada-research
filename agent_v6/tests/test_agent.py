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
    print("\n" + "="*70)
    print("AGENT REQUEST: Create a file called hello.txt with the content 'Hello World!'")
    print("="*70)

    response_text = []
    tool_calls = []

    async for msg in agent.run("Create a file called hello.txt with the content 'Hello World!'"):
        if msg.get("type") == "text_delta":
            response_text.append(msg["text"])
        elif msg.get("type") == "tool_execution":
            tool_calls.append(msg)
            print(f"🔧 Tool: {msg.get('tool_name')}")
            print(f"   Input: {msg.get('tool_input')}")
            print(f"   Result: {msg.get('result', {}).get('content', '')[:100]}")

    full_response = "".join(response_text)
    print("\nAGENT RESPONSE:")
    print(full_response)
    print(f"\nTools called: {len(tool_calls)}")
    print("="*70)

    # Verify file exists
    hello_file = Path(temp_workspace, "hello.txt")

    if not hello_file.exists():
        print(f"\n❌ FILE NOT CREATED")
        print(f"Workspace contents: {list(Path(temp_workspace).iterdir())}")

    assert hello_file.exists(), f"File not created. Agent called {len(tool_calls)} tools. Response: {full_response[:200]}"
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
    print("\n" + "="*70)
    print("TURN 1: Create a file test.py with: print('hello')")
    print("="*70)

    turn1_tools = []
    async for msg in agent.run("Create a file test.py with: print('hello')"):
        if msg.get("type") == "tool_execution":
            turn1_tools.append(msg.get("tool_name"))
            print(f"🔧 Tool: {msg.get('tool_name')}")

    print(f"Turn 1 tools called: {turn1_tools}")

    test_file = Path(temp_workspace, "test.py")
    if not test_file.exists():
        print(f"❌ File not created in Turn 1")
        print(f"Workspace: {list(Path(temp_workspace).iterdir())}")

    assert test_file.exists(), f"Turn 1 failed to create file. Tools used: {turn1_tools}"

    # Turn 2: Read the file back
    print("\n" + "="*70)
    print("TURN 2: Read test.py and tell me what it contains")
    print("="*70)

    response_parts = []
    turn2_tools = []
    async for msg in agent.run("Read test.py and tell me what it contains"):
        if msg.get("type") == "text_delta":
            response_parts.append(msg["text"])
        elif msg.get("type") == "tool_execution":
            turn2_tools.append(msg.get("tool_name"))
            print(f"🔧 Tool: {msg.get('tool_name')}")

    response = "".join(response_parts)
    print(f"\nTurn 2 response: {response}")
    print(f"Turn 2 tools called: {turn2_tools}")

    assert "print" in response.lower() or "hello" in response.lower()

    print(f"\n✓ Multi-turn conversation successful")
    print(f"  Conversation history: {len(agent.conversation_history)} messages")

    # Cleanup
    await agent.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
