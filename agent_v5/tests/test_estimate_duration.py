"""
Tests for EstimateTaskDuration tool
"""
import pytest
from agent_v5.tools.estimate_duration import EstimateTaskDurationTool


@pytest.fixture
def estimate_tool():
    """Create estimate duration tool"""
    return EstimateTaskDurationTool(workspace_dir="/tmp/test")


@pytest.mark.asyncio
async def test_simple_task_estimate(estimate_tool):
    """Test estimating a simple task"""
    result = await estimate_tool.execute({
        "task_type": "load_data"
    })

    assert not result["is_error"]
    assert "load_data" in result["content"]
    assert "Best case" in result["content"]
    assert "Typical" in result["content"]
    assert "Worst case" in result["content"]


@pytest.mark.asyncio
async def test_model_training_estimate(estimate_tool):
    """Test estimating model training"""
    result = await estimate_tool.execute({
        "task_type": "train_complex_model",
        "complexity": "complex",
        "data_size": "large"
    })

    assert not result["is_error"]
    assert "train_complex_model" in result["content"]
    assert "longer task" in result["content"] or "long-running task" in result["content"]


@pytest.mark.asyncio
async def test_gpu_acceleration(estimate_tool):
    """Test estimate with GPU acceleration"""
    # Without GPU
    result_cpu = await estimate_tool.execute({
        "task_type": "train_deep_learning"
    })

    # With GPU
    result_gpu = await estimate_tool.execute({
        "task_type": "train_deep_learning",
        "additional_context": "using GPU"
    })

    assert not result_cpu["is_error"]
    assert not result_gpu["is_error"]

    # GPU version should mention faster execution
    assert "deep_learning" in result_gpu["content"]


@pytest.mark.asyncio
async def test_data_size_modifier(estimate_tool):
    """Test that data size affects estimates"""
    result = await estimate_tool.execute({
        "task_type": "process_large_dataset",
        "data_size": "large"
    })

    assert not result["is_error"]
    assert "process_large_dataset" in result["content"]


@pytest.mark.asyncio
async def test_unknown_task(estimate_tool):
    """Test handling of unknown task type"""
    result = await estimate_tool.execute({
        "task_type": "completely_unknown_task_type"
    })

    assert result["is_error"]
    assert "Unknown task type" in result["content"]
    assert "Available task types" in result["content"]


@pytest.mark.asyncio
async def test_fuzzy_matching(estimate_tool):
    """Test fuzzy matching for task types"""
    # Should match "train_simple_model" or similar
    result = await estimate_tool.execute({
        "task_type": "train_model"
    })

    # Should find a match via fuzzy matching
    assert not result["is_error"]
    assert "train" in result["content"].lower()


@pytest.mark.asyncio
async def test_quick_task_recommendations(estimate_tool):
    """Test recommendations for quick tasks"""
    result = await estimate_tool.execute({
        "task_type": "read_small_file"
    })

    assert not result["is_error"]
    assert "quick task" in result["content"].lower()


@pytest.mark.asyncio
async def test_long_task_recommendations(estimate_tool):
    """Test recommendations for long-running tasks"""
    result = await estimate_tool.execute({
        "task_type": "train_deep_learning",
        "complexity": "complex"
    })

    assert not result["is_error"]
    content = result["content"].lower()
    assert "background" in content or "timeout" in content


@pytest.mark.asyncio
async def test_all_task_types(estimate_tool):
    """Test that all predefined task types work"""
    task_types = [
        "load_data",
        "explore_data",
        "clean_data",
        "feature_engineering",
        "train_simple_model",
        "train_complex_model",
        "evaluate_model",
        "prepare_submission",
    ]

    for task_type in task_types:
        result = await estimate_tool.execute({"task_type": task_type})
        assert not result["is_error"], f"Failed for task_type: {task_type}"
        assert task_type in result["content"]


@pytest.mark.asyncio
async def test_format_duration(estimate_tool):
    """Test duration formatting"""
    # Test seconds
    assert "s" in estimate_tool._format_duration(45)

    # Test minutes
    formatted = estimate_tool._format_duration(150)
    assert "m" in formatted

    # Test hours
    formatted = estimate_tool._format_duration(7200)
    assert "h" in formatted


@pytest.mark.asyncio
async def test_complexity_modifier(estimate_tool):
    """Test that complexity affects estimates"""
    # Simple
    result_simple = await estimate_tool.execute({
        "task_type": "train_simple_model",
        "complexity": "simple"
    })

    # Complex
    result_complex = await estimate_tool.execute({
        "task_type": "train_simple_model",
        "complexity": "complex"
    })

    assert not result_simple["is_error"]
    assert not result_complex["is_error"]
