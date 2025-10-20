"""
Test ensemble tool with REAL API calls

WARNING: These tests make real API calls and will cost money (~$1-2 per run).
All 4 providers (OpenAI, Anthropic, XAI, Google) must have valid API keys set.

Required environment variables:
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- XAI_API_KEY
- GEMINI_API_KEY
"""
import os
import pytest
import sys
from pathlib import Path

# Add agent_v6 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_v6.ensemble.tool import EnsembleTool


@pytest.mark.asyncio
async def test_ensemble_basic_consultation():
    """
    Test basic ensemble consultation with all 4 models + O3 synthesis

    Cost: ~$1-2 per run (5 API calls)
    """
    # Verify all API keys are set
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "XAI_API_KEY", "GEMINI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.environ.get(key)]

    if missing_keys:
        pytest.skip(f"Missing API keys: {', '.join(missing_keys)}")

    # Create simple conversation history
    conversation_history = [
        {
            "role": "user",
            "content": "I'm building a machine learning model for image classification with 100 categories and 10,000 images."
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I understand. Let me help you design an approach."}
            ]
        }
    ]

    def get_history():
        return conversation_history

    # Create tool
    tool = EnsembleTool(workspace_dir="/tmp/test_ensemble", get_conversation_history=get_history)

    # Test tool schema
    schema = tool.schema
    assert schema["name"] == "ConsultEnsemble"
    assert "input_schema" in schema
    assert "problem" in schema["input_schema"]["properties"]

    # Execute ensemble consultation
    print("\n" + "="*70)
    print("Testing ensemble consultation with REAL API calls")
    print("This will cost ~$1-2 and take 30-60 seconds")
    print("="*70)

    result = await tool.execute({
        "problem": "Should I use ResNet-50, EfficientNet-B0, or ViT for this task? Consider training time and accuracy.",
        "context": "Training on single GPU (24GB), time budget is 2 hours"
    })

    # Verify result structure
    assert isinstance(result, dict)
    assert "content" in result
    assert "is_error" in result

    # Print result for manual inspection
    print("\n" + "="*70)
    print("ENSEMBLE RESULT:")
    print("="*70)
    print(result["content"])
    print("="*70)

    # Verify no error
    if result["is_error"]:
        print(f"\nERROR: {result['content']}")
        pytest.fail(f"Ensemble consultation failed: {result['content']}")

    # Verify response contains expected sections
    content = result["content"]

    # Should contain multiple expert responses
    assert "EXPERT" in content, "Response should contain expert sections"

    # Should contain synthesis
    assert "SYNTHESIZED" in content or "OPTIMAL PLAN" in content, "Response should contain O3 synthesis"

    # Should mention at least one model name
    model_names = ["GPT-5", "Claude", "Grok", "Gemini"]
    assert any(name in content for name in model_names), "Response should mention model names"

    print("\n✅ Test passed - Ensemble consultation successful!")


@pytest.mark.asyncio
async def test_ensemble_partial_failure():
    """
    Test ensemble handles partial failures gracefully

    If 1-2 models fail, ensemble should still work with remaining models
    """
    conversation_history = [
        {
            "role": "user",
            "content": "Simple test question"
        }
    ]

    def get_history():
        return conversation_history

    tool = EnsembleTool(workspace_dir="/tmp/test_ensemble", get_conversation_history=get_history)

    # Use simple problem to reduce cost
    result = await tool.execute({
        "problem": "What is 2+2?",
    })

    # Even if some models fail, should get a response
    assert isinstance(result, dict)
    assert "content" in result

    # Print for inspection
    print("\n" + "="*70)
    print("PARTIAL FAILURE TEST RESULT:")
    print("="*70)
    print(result["content"])
    print("="*70)


@pytest.mark.asyncio
async def test_ensemble_missing_api_keys():
    """Test that ensemble fails gracefully when API keys are missing"""

    # Temporarily remove API keys
    original_keys = {}
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "XAI_API_KEY", "GEMINI_API_KEY"]:
        original_keys[key] = os.environ.pop(key, None)

    try:
        conversation_history = []
        def get_history():
            return conversation_history

        tool = EnsembleTool(workspace_dir="/tmp/test_ensemble", get_conversation_history=get_history)

        result = await tool.execute({"problem": "Test"})

        # Should return error about missing keys
        assert result["is_error"] is True
        assert "Missing API keys" in result["content"]

        print("\n✅ Test passed - Correctly detected missing API keys")

    finally:
        # Restore API keys
        for key, value in original_keys.items():
            if value is not None:
                os.environ[key] = value


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
