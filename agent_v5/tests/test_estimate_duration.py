"""
Tests for EstimateDuration tool
"""
import pytest
import time
from agent_v5.tools.estimate_duration import EstimateDurationTool, TaskDurationEstimator


class TestTaskDurationEstimator:
    """Test core estimation logic"""

    def test_base_estimates_exist_for_all_task_types(self):
        """Verify all task types have base estimates"""
        task_types = [
            "image_classification",
            "image_segmentation",
            "object_detection",
            "tabular",
            "nlp_classification",
            "time_series",
            "audio"
        ]

        for task_type in task_types:
            assert task_type in TaskDurationEstimator.BASE_ESTIMATES
            estimates = TaskDurationEstimator.BASE_ESTIMATES[task_type]
            assert "small" in estimates
            assert "medium" in estimates
            assert "large" in estimates

    def test_estimate_base_time_image_classification(self):
        """Test estimation for image classification"""
        min_t, typ_t, max_t = TaskDurationEstimator.estimate_base_time(
            task_type="image_classification",
            dataset_size="medium",
            complexity="moderate",
            num_parallel_models=1
        )

        # Medium image classification: 12-30 min typical
        assert 10 < typ_t < 35
        assert min_t < typ_t < max_t

    def test_estimate_base_time_tabular(self):
        """Test estimation for tabular data"""
        min_t, typ_t, max_t = TaskDurationEstimator.estimate_base_time(
            task_type="tabular",
            dataset_size="small",
            complexity="simple",
            num_parallel_models=1
        )

        # Small tabular: 3-8 min typical, simple: 0.7x
        assert 2 < typ_t < 10
        assert min_t < typ_t < max_t

    def test_complexity_multiplier(self):
        """Test complexity affects duration"""
        # Simple should be faster than complex
        _, simple_t, _ = TaskDurationEstimator.estimate_base_time(
            task_type="tabular",
            dataset_size="medium",
            complexity="simple",
            num_parallel_models=1
        )

        _, complex_t, _ = TaskDurationEstimator.estimate_base_time(
            task_type="tabular",
            dataset_size="medium",
            complexity="complex",
            num_parallel_models=1
        )

        assert simple_t < complex_t
        # Simple is 0.7x, complex is 1.5x
        assert abs(complex_t / simple_t - (1.5 / 0.7)) < 0.1

    def test_parallel_efficiency(self):
        """Test parallel training increases time but not linearly"""
        _, single_t, _ = TaskDurationEstimator.estimate_base_time(
            task_type="image_classification",
            dataset_size="medium",
            complexity="moderate",
            num_parallel_models=1
        )

        _, double_t, _ = TaskDurationEstimator.estimate_base_time(
            task_type="image_classification",
            dataset_size="medium",
            complexity="moderate",
            num_parallel_models=2
        )

        _, triple_t, _ = TaskDurationEstimator.estimate_base_time(
            task_type="image_classification",
            dataset_size="medium",
            complexity="moderate",
            num_parallel_models=3
        )

        # 2 models should be ~1.3x slower (not 2x)
        assert 1.2 < double_t / single_t < 1.4

        # 3 models should be ~1.5x slower (not 3x)
        assert 1.4 < triple_t / single_t < 1.6

    def test_adaptive_strategy_full(self):
        """Test FULL strategy when plenty of time"""
        result = TaskDurationEstimator.adaptive_time_allocation(
            estimated_time=10.0,
            time_remaining=20.0,
            time_total=30.0
        )

        assert result["urgency"] == "low"
        assert result["strategy"] == "full"
        assert result["time_ratio"] <= 0.6
        assert "FULL" in result["guidance"]

    def test_adaptive_strategy_standard(self):
        """Test STANDARD strategy when comfortable"""
        result = TaskDurationEstimator.adaptive_time_allocation(
            estimated_time=15.0,
            time_remaining=16.0,
            time_total=30.0
        )

        assert result["urgency"] == "medium"
        assert result["strategy"] == "standard"
        assert 0.6 < result["time_ratio"] <= 1.0
        assert "STANDARD" in result["guidance"]

    def test_adaptive_strategy_fast(self):
        """Test FAST strategy when tight"""
        result = TaskDurationEstimator.adaptive_time_allocation(
            estimated_time=12.0,
            time_remaining=10.0,
            time_total=30.0
        )

        assert result["urgency"] == "high"
        assert result["strategy"] == "fast"
        assert 1.0 < result["time_ratio"] <= 1.3
        assert "FAST" in result["guidance"]

    def test_adaptive_strategy_emergency(self):
        """Test EMERGENCY strategy when critical"""
        result = TaskDurationEstimator.adaptive_time_allocation(
            estimated_time=15.0,
            time_remaining=5.0,
            time_total=30.0
        )

        assert result["urgency"] == "critical"
        assert result["strategy"] == "emergency"
        assert result["time_ratio"] > 1.3
        assert "EMERGENCY" in result["guidance"]

    def test_speed_modifier_adjustments(self):
        """Test speed modifiers are applied correctly"""
        full = TaskDurationEstimator.adaptive_time_allocation(10, 20, 30)
        standard = TaskDurationEstimator.adaptive_time_allocation(15, 16, 30)
        fast = TaskDurationEstimator.adaptive_time_allocation(12, 10, 30)
        emergency = TaskDurationEstimator.adaptive_time_allocation(15, 5, 30)

        assert full["speed_modifier"] == 1.0
        assert standard["speed_modifier"] == 1.0
        assert fast["speed_modifier"] == 0.7
        assert emergency["speed_modifier"] == 0.5


@pytest.mark.asyncio
class TestEstimateDurationTool:
    """Test EstimateDuration tool"""

    async def test_basic_execution(self):
        """Test basic tool execution"""
        tool = EstimateDurationTool(
            workspace_dir="/tmp/test",
            start_time=time.time() - 300,  # Started 5 min ago
            total_budget_min=30.0
        )

        result = await tool.execute({
            "task_type": "image_classification",
            "dataset_size": "medium"
        })

        assert not result["is_error"]
        assert "TASK DURATION ESTIMATE" in result["content"]
        assert "TIME ESTIMATES" in result["content"]
        assert "ADAPTIVE STRATEGY" in result["content"]
        assert "MODEL RECOMMENDATIONS" in result["content"]

    async def test_with_description(self):
        """Test with optional description"""
        tool = EstimateDurationTool(
            workspace_dir="/tmp/test",
            start_time=time.time() - 600,  # Started 10 min ago
            total_budget_min=30.0
        )

        result = await tool.execute({
            "task_type": "tabular",
            "dataset_size": "large",
            "complexity": "complex",
            "description": "Customer churn prediction"
        })

        assert not result["is_error"]
        assert "Customer churn prediction" in result["content"]

    async def test_all_task_types(self):
        """Test all task types execute without error"""
        tool = EstimateDurationTool(
            workspace_dir="/tmp/test",
            start_time=time.time(),
            total_budget_min=30.0
        )

        task_types = [
            "image_classification",
            "image_segmentation",
            "object_detection",
            "tabular",
            "nlp_classification",
            "time_series",
            "audio"
        ]

        for task_type in task_types:
            result = await tool.execute({
                "task_type": task_type,
                "dataset_size": "medium"
            })
            assert not result["is_error"], f"Failed for {task_type}"
            assert task_type in result["debug_summary"]

    async def test_time_budget_tracking(self):
        """Test time budget is tracked correctly"""
        # Start 20 min ago
        tool = EstimateDurationTool(
            workspace_dir="/tmp/test",
            start_time=time.time() - 1200,
            total_budget_min=30.0
        )

        result = await tool.execute({
            "task_type": "image_classification",
            "dataset_size": "small"
        })

        assert not result["is_error"]
        content = result["content"]

        # Should show ~20 min elapsed, ~10 min remaining
        assert "Elapsed:" in content
        assert "Remaining:" in content

        # Should be in FAST or EMERGENCY strategy
        assert "FAST" in content or "EMERGENCY" in content

    async def test_parallel_models_parameter(self):
        """Test num_parallel_models parameter"""
        tool = EstimateDurationTool(
            workspace_dir="/tmp/test",
            start_time=time.time(),
            total_budget_min=30.0
        )

        result = await tool.execute({
            "task_type": "image_classification",
            "dataset_size": "medium",
            "num_parallel_models": 3
        })

        assert not result["is_error"]
        assert "Parallel models: 3" in result["content"]

    async def test_model_recommendations_present(self):
        """Test model recommendations are included"""
        tool = EstimateDurationTool(
            workspace_dir="/tmp/test",
            start_time=time.time() - 300,  # 5 min ago
            total_budget_min=30.0
        )

        # Image classification
        result = await tool.execute({
            "task_type": "image_classification",
            "dataset_size": "medium"
        })
        assert "EfficientNet" in result["content"]

        # Tabular
        result = await tool.execute({
            "task_type": "tabular",
            "dataset_size": "small"
        })
        assert "LightGBM" in result["content"]

        # NLP
        result = await tool.execute({
            "task_type": "nlp_classification",
            "dataset_size": "medium"
        })
        assert "distilbert" in result["content"] or "DeBERTa" in result["content"]

    async def test_schema_validation(self):
        """Test tool schema is valid"""
        tool = EstimateDurationTool(
            workspace_dir="/tmp/test",
            start_time=time.time(),
            total_budget_min=30.0
        )

        schema = tool.schema
        assert schema["name"] == "EstimateDuration"
        assert "description" in schema
        assert "input_schema" in schema

        input_schema = schema["input_schema"]
        assert "properties" in input_schema
        assert "task_type" in input_schema["properties"]
        assert "dataset_size" in input_schema["properties"]
        assert "complexity" in input_schema["properties"]
        assert "num_parallel_models" in input_schema["properties"]

        # Check required fields
        assert set(input_schema["required"]) == {"task_type", "dataset_size"}

    async def test_error_handling(self):
        """Test error handling for invalid inputs"""
        tool = EstimateDurationTool(
            workspace_dir="/tmp/test",
            start_time=time.time(),
            total_budget_min=30.0
        )

        # Missing required field
        result = await tool.execute({
            "dataset_size": "medium"
        })
        assert result["is_error"]


@pytest.mark.asyncio
class TestIntegrationScenarios:
    """Test realistic usage scenarios"""

    async def test_scenario_early_exploration(self):
        """Scenario: Just started, exploring data (5 min elapsed)"""
        tool = EstimateDurationTool(
            workspace_dir="/tmp/test",
            start_time=time.time() - 300,  # 5 min ago
            total_budget_min=30.0
        )

        result = await tool.execute({
            "task_type": "image_classification",
            "dataset_size": "medium",
            "description": "Cassava leaf disease classification"
        })

        assert not result["is_error"]
        # Should recommend STANDARD or FULL strategy
        assert "STANDARD" in result["content"] or "FULL" in result["content"]

    async def test_scenario_midway(self):
        """Scenario: Midway through (15 min elapsed)"""
        tool = EstimateDurationTool(
            workspace_dir="/tmp/test",
            start_time=time.time() - 900,  # 15 min ago
            total_budget_min=30.0
        )

        result = await tool.execute({
            "task_type": "tabular",
            "dataset_size": "large",
            "complexity": "moderate"
        })

        assert not result["is_error"]
        # Should recommend STANDARD or FAST strategy
        assert "STANDARD" in result["content"] or "FAST" in result["content"]

    async def test_scenario_running_behind(self):
        """Scenario: Running behind schedule (22 min elapsed)"""
        tool = EstimateDurationTool(
            workspace_dir="/tmp/test",
            start_time=time.time() - 1320,  # 22 min ago
            total_budget_min=30.0
        )

        result = await tool.execute({
            "task_type": "nlp_classification",
            "dataset_size": "medium"
        })

        assert not result["is_error"]
        # Should recommend FAST or EMERGENCY strategy
        assert "FAST" in result["content"] or "EMERGENCY" in result["content"]
        # Should suggest speed optimizations
        assert "distilbert" in result["content"] or "TF-IDF" in result["content"]

    async def test_scenario_parallel_training(self):
        """Scenario: Planning parallel training ensemble"""
        tool = EstimateDurationTool(
            workspace_dir="/tmp/test",
            start_time=time.time() - 480,  # 8 min ago
            total_budget_min=30.0
        )

        result = await tool.execute({
            "task_type": "image_classification",
            "dataset_size": "medium",
            "complexity": "moderate",
            "num_parallel_models": 3,
            "description": "Ensemble: EfficientNet-B3 + ResNet-50 + ViT-small"
        })

        assert not result["is_error"]
        # Parallel training should increase estimate
        assert "Parallel models: 3" in result["content"]

    async def test_scenario_quick_tabular(self):
        """Scenario: Quick tabular competition"""
        tool = EstimateDurationTool(
            workspace_dir="/tmp/test",
            start_time=time.time() - 180,  # 3 min ago
            total_budget_min=30.0
        )

        result = await tool.execute({
            "task_type": "tabular",
            "dataset_size": "small",
            "complexity": "simple"
        })

        assert not result["is_error"]
        # Should be quick (< 10 min estimate)
        assert "Optimistic:" in result["content"]
