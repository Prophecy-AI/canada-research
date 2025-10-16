"""
Tests for PlanTask tool
"""
import pytest
import os
from pathlib import Path
from agent_v5.tools.plan import PlanTaskTool


@pytest.fixture
def temp_workspace(tmp_path):
    """Create temporary workspace directory"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return str(workspace)


@pytest.mark.asyncio
async def test_plan_basic_task(temp_workspace):
    """Test planning a basic task"""
    tool = PlanTaskTool(temp_workspace)

    result = await tool.execute({
        "task_description": "Analyze a CSV file containing sales data and create a summary report"
    })

    assert not result["is_error"]
    assert "Execution Plan" in result["content"]
    assert "Goal Analysis" in result["content"] or "Step" in result["content"]
    # Plan should mention relevant tools or steps
    assert any(keyword in result["content"].lower() for keyword in ["read", "analyze", "csv", "report"])


@pytest.mark.asyncio
async def test_plan_with_context(temp_workspace):
    """Test planning with additional context"""
    tool = PlanTaskTool(temp_workspace)

    result = await tool.execute({
        "task_description": "Build a machine learning model to predict customer churn",
        "context": "We have a dataset with 50K rows and 20 features. Historical accuracy should be >80%."
    })

    assert not result["is_error"]
    assert "Execution Plan" in result["content"]
    # Should incorporate context about dataset size and accuracy requirement
    content_lower = result["content"].lower()
    assert any(keyword in content_lower for keyword in ["data", "model", "accuracy", "features"])


@pytest.mark.asyncio
async def test_plan_with_conversation_history(temp_workspace):
    """Test planning with conversation history context"""
    conversation_history = [
        {"role": "user", "content": "I need to analyze healthcare prescription data from BigQuery"},
        {"role": "assistant", "content": "I can help with that. What specific analysis do you need?"}
    ]

    tool = PlanTaskTool(temp_workspace, lambda: conversation_history)

    result = await tool.execute({
        "task_description": "Find top 10 prescribers of HUMIRA in California"
    })

    assert not result["is_error"]
    assert "Execution Plan" in result["content"]
    # Plan should be relevant to the healthcare/prescription context


@pytest.mark.asyncio
async def test_plan_complex_multi_step_task(temp_workspace):
    """Test planning a complex multi-step task"""
    tool = PlanTaskTool(temp_workspace)

    result = await tool.execute({
        "task_description": (
            "Create a data pipeline that:\n"
            "1. Fetches data from BigQuery\n"
            "2. Cleans and validates the data\n"
            "3. Performs statistical analysis\n"
            "4. Generates visualizations\n"
            "5. Creates a PDF report"
        ),
        "context": "Data contains patient records with PHI, must be HIPAA compliant"
    })

    assert not result["is_error"]
    content = result["content"]
    assert "Execution Plan" in content

    # Should have structured planning sections
    content_lower = content.lower()
    assert any(keyword in content_lower for keyword in ["step", "stage", "phase"])

    # Should mention data privacy/compliance given context
    assert any(keyword in content_lower for keyword in ["hipaa", "privacy", "compliance", "phi"])


@pytest.mark.asyncio
async def test_plan_tool_schema():
    """Test that tool schema is properly formatted"""
    tool = PlanTaskTool("/tmp/workspace")

    schema = tool.schema

    assert schema["name"] == "PlanTask"
    assert "description" in schema
    assert "input_schema" in schema
    assert schema["input_schema"]["type"] == "object"
    assert "task_description" in schema["input_schema"]["properties"]
    assert "context" in schema["input_schema"]["properties"]
    assert "task_description" in schema["input_schema"]["required"]


@pytest.mark.asyncio
async def test_plan_uses_correct_model(temp_workspace):
    """Test that planning uses the configured reasoning model"""
    # Set custom planning model via environment variable
    original_model = os.getenv("PLANNING_MODEL")
    os.environ["PLANNING_MODEL"] = "claude-opus-4-20250514"

    try:
        tool = PlanTaskTool(temp_workspace)
        assert tool.planning_model == "claude-opus-4-20250514"

        result = await tool.execute({
            "task_description": "Simple test task"
        })

        # Should mention the model used
        assert "claude-opus-4-20250514" in result["content"]
    finally:
        # Restore original environment
        if original_model:
            os.environ["PLANNING_MODEL"] = original_model
        elif "PLANNING_MODEL" in os.environ:
            del os.environ["PLANNING_MODEL"]


@pytest.mark.asyncio
async def test_plan_error_handling(temp_workspace):
    """Test error handling when planning fails"""
    tool = PlanTaskTool(temp_workspace)

    # Save original API key
    original_key = os.environ.get("ANTHROPIC_API_KEY")

    try:
        # Set invalid API key to trigger error
        os.environ["ANTHROPIC_API_KEY"] = "invalid_key"

        # Recreate tool with invalid key
        tool = PlanTaskTool(temp_workspace)

        result = await tool.execute({
            "task_description": "Test task"
        })

        # Should return error gracefully
        assert result["is_error"]
        assert "Planning failed" in result["content"] or "error" in result["content"].lower()
    finally:
        # Restore original API key
        if original_key:
            os.environ["ANTHROPIC_API_KEY"] = original_key


@pytest.mark.asyncio
async def test_plan_includes_available_tools(temp_workspace):
    """Test that plan mentions available tools"""
    tool = PlanTaskTool(temp_workspace)

    result = await tool.execute({
        "task_description": "Search for Python files containing a specific function and modify them"
    })

    assert not result["is_error"]
    content_lower = result["content"].lower()

    # Should suggest using appropriate tools for this task
    # (Glob/Grep for searching, Edit for modifying)
    assert any(keyword in content_lower for keyword in ["glob", "grep", "search", "find"])


@pytest.mark.asyncio
async def test_plan_identifies_risks(temp_workspace):
    """Test that plan identifies potential risks and edge cases"""
    tool = PlanTaskTool(temp_workspace)

    result = await tool.execute({
        "task_description": "Delete all log files older than 30 days from production servers"
    })

    assert not result["is_error"]
    content_lower = result["content"].lower()

    # Should identify risks related to deletion, production, etc.
    assert any(keyword in content_lower for keyword in [
        "risk", "backup", "verify", "validate", "careful", "check", "confirm"
    ])
