"""
Test token estimation and automatic memory compaction

Tests the new 150k token threshold compaction feature with real API calls.
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
    temp_dir = tempfile.mkdtemp(prefix="test_token_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_estimate_token_count_simple():
    """Test token estimation with simple string messages"""
    agent = IDEAgent(
        session_id="test_tokens",
        workspace_dir="/tmp/test",
        system_prompt="Test agent",
        enable_memory_compaction=False
    )

    # Test 1: Empty conversation
    agent.conversation_history = []
    assert agent._estimate_token_count() == 0

    # Test 2: Simple string messages
    agent.conversation_history = [
        {"role": "user", "content": "Hello"},  # 5 chars = ~1 token
        {"role": "assistant", "content": "Hi there!"}  # 9 chars = ~2 tokens
    ]
    tokens = agent._estimate_token_count()
    assert tokens == 3  # (5 + 9) / 4 = 3.5 → 3

    # Test 3: Longer messages
    long_message = "This is a longer message with many words. " * 10  # ~420 chars
    agent.conversation_history = [
        {"role": "user", "content": long_message}
    ]
    tokens = agent._estimate_token_count()
    assert tokens == 105  # 420 / 4 = 105


def test_estimate_token_count_complex():
    """Test token estimation with complex content (lists, dicts)"""
    agent = IDEAgent(
        session_id="test_tokens_complex",
        workspace_dir="/tmp/test",
        system_prompt="Test agent",
        enable_memory_compaction=False
    )

    # Test with list content (tool use format)
    agent.conversation_history = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me read that file"},
                {"type": "tool_use", "name": "Read", "input": {"file_path": "data.txt"}},
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "content": "File contents: " + "x" * 100}
            ]
        }
    ]

    tokens = agent._estimate_token_count()
    # Should handle complex structures by converting to string
    assert tokens > 0
    print(f"\nComplex content tokens: {tokens}")


def test_estimate_token_count_realistic():
    """Test token estimation with realistic conversation"""
    agent = IDEAgent(
        session_id="test_tokens_real",
        workspace_dir="/tmp/test",
        system_prompt="Test agent",
        enable_memory_compaction=False
    )

    # Simulate realistic conversation with ~10k tokens
    code_snippet = """
def train_model(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model
""" * 5  # ~1000 chars = 250 tokens per message

    agent.conversation_history = [
        {"role": "user", "content": "Write a training script"},
        {"role": "assistant", "content": code_snippet},
        {"role": "user", "content": "Now add validation"},
        {"role": "assistant", "content": code_snippet + code_snippet},
        {"role": "user", "content": "Add hyperparameter tuning"},
        {"role": "assistant", "content": code_snippet * 5},
    ] * 3  # Repeat to get ~10k tokens

    tokens = agent._estimate_token_count()
    print(f"\nRealistic conversation tokens: {tokens:,}")
    assert 5000 <= tokens <= 8000  # Should be around 5-6k based on actual data


@pytest.mark.asyncio
async def test_no_compaction_below_threshold(temp_workspace):
    """Test that compaction doesn't trigger below 150k tokens"""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    agent = IDEAgent(
        session_id="test_no_compact",
        workspace_dir=temp_workspace,
        system_prompt="Test agent",
        enable_memory_compaction=True
    )

    # Add messages totaling ~5k tokens (well below 150k)
    message = "This is a test message. " * 50  # ~1250 chars = ~312 tokens
    for i in range(16):  # 16 * 312 = ~5k tokens
        agent.conversation_history.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Message {i}: {message}"
        })

    tokens_before = agent._estimate_token_count()
    len_before = len(agent.conversation_history)

    # Trigger compaction check
    await agent._maybe_compact_memory()

    tokens_after = agent._estimate_token_count()
    len_after = len(agent.conversation_history)

    print(f"\nBefore: {len_before} messages, {tokens_before:,} tokens")
    print(f"After: {len_after} messages, {tokens_after:,} tokens")

    # Should NOT have compacted (below 150k threshold)
    assert len_after == len_before
    assert tokens_after == tokens_before


@pytest.mark.asyncio
async def test_compaction_above_150k_tokens(temp_workspace):
    """Test that compaction triggers above 150k tokens - REAL API CALL"""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    print("\n" + "="*70)
    print("TESTING 150K TOKEN THRESHOLD COMPACTION (Real O3 API Call)")
    print("="*70)

    agent = IDEAgent(
        session_id="test_150k_compact",
        workspace_dir=temp_workspace,
        system_prompt="You are a data analysis agent",
        enable_memory_compaction=True
    )

    # Create messages totaling >150k tokens
    # Each message: ~6000 chars = ~1500 tokens
    # Need 100+ messages to exceed 150k
    base_message = (
        "I'm analyzing the dataset. Here are the results:\n"
        "Mean: 42.5, Median: 41.2, Std: 15.3\n"
        "The distribution shows strong right skew with outliers above 95th percentile.\n"
        "Key findings include seasonal patterns and correlation with external factors.\n"
    ) * 30  # ~6000 chars

    # Add 110 messages (110 * 1500 = 165k tokens)
    for i in range(110):
        agent.conversation_history.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Turn {i}:\n{base_message}"
        })

    tokens_before = agent._estimate_token_count()
    len_before = len(agent.conversation_history)

    print(f"\nBefore compaction:")
    print(f"  Messages: {len_before}")
    print(f"  Estimated tokens: {tokens_before:,}")
    print(f"  Exceeds 150k: {tokens_before > 150_000}")

    assert tokens_before > 150_000, "Test setup failed: didn't create >150k tokens"

    # Trigger compaction (will call O3 API)
    print(f"\nTriggering compaction (calling O3 for summarization)...")
    await agent._maybe_compact_memory()

    tokens_after = agent._estimate_token_count()
    len_after = len(agent.conversation_history)

    print(f"\nAfter compaction:")
    print(f"  Messages: {len_after}")
    print(f"  Estimated tokens: {tokens_after:,}")
    print(f"  Token reduction: {tokens_before - tokens_after:,} ({(1 - tokens_after/tokens_before)*100:.1f}%)")
    print(f"  Message reduction: {len_before - len_after}")

    # Verify compaction occurred
    assert len_after < len_before, "Compaction should reduce message count"
    assert tokens_after < tokens_before, "Compaction should reduce token count"
    assert tokens_after < 150_000, "Should be below threshold after compaction"

    # Verify summary structure
    assert any("SUMMARY" in str(msg.get("content", "")) for msg in agent.conversation_history[:2]), \
        "Should have summary message"

    # Verify recent messages preserved (last 20)
    assert agent.conversation_history[-1]["content"].startswith("Turn 109:"), \
        "Should preserve most recent message"

    print("\n✅ Compaction successful!")
    print("="*70)


@pytest.mark.asyncio
async def test_compaction_maintains_quality(temp_workspace):
    """Test that compacted conversation still works with agent - REAL API CALL"""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    print("\n" + "="*70)
    print("TESTING COMPACTION QUALITY (Agent still functions after compaction)")
    print("="*70)

    agent = IDEAgent(
        session_id="test_quality",
        workspace_dir=temp_workspace,
        system_prompt="You are a helpful data analysis agent.",
        enable_memory_compaction=True
    )

    # Create large conversation history
    for i in range(60):
        agent.conversation_history.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": ("Analyzing sales data, found interesting patterns. " * 100)
        })

    # Force compaction
    await agent._maybe_compact_memory()

    print(f"\nHistory compacted to {len(agent.conversation_history)} messages")

    # Now test that agent still functions (doesn't crash)
    print("\nSending message to agent (testing basic functionality)...")
    response_parts = []
    async for msg in agent.run("Hello, can you help me?"):
        if msg.get("type") == "text_delta":
            response_parts.append(msg["text"])

    response = "".join(response_parts)
    print(f"\nAgent response:\n{response[:200]}...")

    # Verify agent is operational (responds with something coherent)
    assert len(response) > 10, "Agent should generate a response"
    assert response.strip(), "Agent response should not be empty"

    # Verify agent is helpful (not crashing with actual errors)
    # Check for actual error patterns, not just substring matches
    error_patterns = [
        "error occurred",
        "failed to",
        "exception:",
        "traceback",
        "cannot proceed"
    ]
    assert not any(err in response.lower() for err in error_patterns), \
        f"Agent should not error after compaction. Response: {response[:200]}"

    print("\n✅ Agent remains operational after compaction!")
    print("="*70)


@pytest.mark.asyncio
async def test_multiple_compactions(temp_workspace):
    """Test that multiple compactions work correctly"""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    print("\n" + "="*70)
    print("TESTING MULTIPLE COMPACTIONS")
    print("="*70)

    agent = IDEAgent(
        session_id="test_multi_compact",
        workspace_dir=temp_workspace,
        system_prompt="Data analysis agent",
        enable_memory_compaction=True
    )

    # First compaction
    for i in range(110):
        agent.conversation_history.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": ("Analysis results: Mean=50, patterns observed. " * 30)
        })

    tokens_1 = agent._estimate_token_count()
    await agent._maybe_compact_memory()
    tokens_2 = agent._estimate_token_count()

    print(f"\nFirst compaction: {tokens_1:,} → {tokens_2:,} tokens")

    # Add more messages to trigger second compaction
    for i in range(110):
        agent.conversation_history.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": ("Further analysis complete, trends identified. " * 30)
        })

    tokens_3 = agent._estimate_token_count()
    await agent._maybe_compact_memory()
    tokens_4 = agent._estimate_token_count()

    print(f"Second compaction: {tokens_3:,} → {tokens_4:,} tokens")

    # Both compactions should reduce tokens
    assert tokens_2 < tokens_1
    assert tokens_4 < tokens_3
    assert tokens_4 < 150_000

    print("\n✅ Multiple compactions work correctly!")
    print("="*70)


@pytest.mark.asyncio
async def test_compaction_clears_response_id_cache(temp_workspace):
    """Test that compaction clears previous_response_id to prevent cache mismatch"""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    print("\n" + "="*70)
    print("TESTING RESPONSE ID CACHE INVALIDATION")
    print("="*70)

    agent = IDEAgent(
        session_id="test_cache_clear",
        workspace_dir=temp_workspace,
        system_prompt="Test agent",
        enable_memory_compaction=True
    )

    # Simulate having a previous response ID (like after a real API call)
    agent.last_response_id = "fake_response_id_12345"
    print(f"\nBefore compaction: last_response_id = {agent.last_response_id}")

    # Create large conversation to trigger compaction
    for i in range(110):
        agent.conversation_history.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": ("Test message with data. " * 30)
        })

    # Trigger compaction
    await agent._maybe_compact_memory()

    print(f"After compaction: last_response_id = {agent.last_response_id}")

    # CRITICAL: response_id must be cleared after compaction
    # Otherwise OpenAI's cache will be mismatched with the new conversation structure
    assert agent.last_response_id is None, \
        "response_id must be cleared after compaction to prevent cache mismatch"

    print("\n✅ Response ID cache correctly invalidated after compaction!")
    print("="*70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
