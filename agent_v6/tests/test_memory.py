"""
Test MemoryCompactor with real O3 API calls
"""
import os
import pytest

from agent_v6.memory import MemoryCompactor


@pytest.mark.asyncio
async def test_memory_compactor_basic():
    """Test basic memory compaction with real O3 API"""
    # Skip if no API key
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    compactor = MemoryCompactor(keep_recent=5)

    # Create sample conversation history
    conversation_history = [
        {"role": "user", "content": "I want to analyze sales data"},
        {"role": "assistant", "content": "I'll help you analyze the sales data. Let me start by reading the file."},
        {"role": "user", "content": [{"type": "text", "text": "Read data.csv"}]},
        {"role": "assistant", "content": "I've read the file. It contains 1000 rows of sales data."},
        {"role": "user", "content": "Calculate total revenue"},
        {"role": "assistant", "content": "Total revenue is $50,000"},
        {"role": "user", "content": "Create visualization"},
        {"role": "assistant", "content": "Created chart.png showing revenue by month"},
        {"role": "user", "content": "What's the trend?"},
        {"role": "assistant", "content": "Revenue is increasing 10% month-over-month"},
        # Recent messages (will be kept intact)
        {"role": "user", "content": "Export the results"},
        {"role": "assistant", "content": "Exported to results.xlsx"},
    ]

    # Compact the history
    compacted = await compactor.compact(conversation_history)

    # Verify structure
    assert len(compacted) < len(conversation_history)
    assert len(compacted) >= 5  # At least keep_recent messages

    # Verify summary is present
    assert any("SUMMARY" in str(msg.get("content", "")) for msg in compacted[:2])

    # Verify recent messages preserved
    assert compacted[-2]["content"] == "Export the results"
    assert compacted[-1]["content"] == "Exported to results.xlsx"

    print("\n" + "="*70)
    print("COMPACTION TEST RESULT:")
    print("="*70)
    print(f"Original: {len(conversation_history)} messages")
    print(f"Compacted: {len(compacted)} messages")
    print(f"\nSummary message:")
    print(compacted[0]["content"][:500] + "..." if len(compacted[0]["content"]) > 500 else compacted[0]["content"])
    print("="*70)


@pytest.mark.asyncio
async def test_no_compaction_when_short():
    """Test that short conversations aren't compacted"""
    compactor = MemoryCompactor(keep_recent=10)

    short_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm doing well, thanks!"},
    ]

    compacted = await compactor.compact(short_history)

    # Should return unchanged
    assert len(compacted) == len(short_history)
    assert compacted == short_history


def test_should_compact():
    """Test compaction threshold logic"""
    compactor = MemoryCompactor(keep_recent=10)

    short = [{"role": "user", "content": "msg"}] * 10
    assert not compactor.should_compact(short)

    long = [{"role": "user", "content": "msg"}] * 20
    assert compactor.should_compact(long)


def test_estimate_token_savings():
    """Test token savings estimation"""
    compactor = MemoryCompactor(keep_recent=5, compression_ratio=0.25)

    # Short history - no savings
    short = [{"role": "user", "content": "hi"}] * 3
    assert compactor.estimate_token_savings(short) == 0

    # Long history - should have savings
    long = [{"role": "user", "content": "This is a longer message " * 20}] * 20
    savings = compactor.estimate_token_savings(long)
    assert savings > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
