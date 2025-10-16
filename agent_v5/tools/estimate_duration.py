"""
EstimateTaskDuration - Estimate how long a task should take

Provides heuristic-based estimates for common data science and Kaggle tasks.
Helps agents set expectations and detect when tasks are taking too long.
"""
import time
from typing import Dict, Optional
from agent_v5.tools.base import BaseTool


class EstimateTaskDurationTool(BaseTool):
    """Estimate how long a task should take based on heuristics"""

    def __init__(self, workspace_dir: str):
        super().__init__(workspace_dir)

        # Task duration estimates (in seconds)
        # Format: (min_duration, typical_duration, max_duration)
        self.task_estimates = {
            # Data exploration
            "load_data": (1, 5, 30),
            "explore_data": (10, 60, 300),
            "data_profiling": (30, 120, 600),
            "visualize_data": (20, 60, 180),

            # Data preprocessing
            "clean_data": (30, 180, 900),
            "feature_engineering": (60, 300, 1800),
            "handle_missing_values": (20, 120, 600),
            "encode_categorical": (10, 60, 300),
            "scale_features": (5, 30, 120),

            # Model training
            "train_simple_model": (10, 60, 300),
            "train_complex_model": (120, 600, 3600),
            "train_deep_learning": (300, 1800, 7200),
            "hyperparameter_tuning": (300, 1800, 7200),
            "cross_validation": (60, 300, 1800),

            # Model evaluation
            "evaluate_model": (5, 30, 120),
            "generate_predictions": (5, 30, 180),
            "calculate_metrics": (5, 20, 60),

            # Large data operations
            "process_large_dataset": (120, 600, 3600),
            "merge_large_dataframes": (30, 180, 900),
            "aggregate_data": (10, 60, 300),

            # Code operations
            "write_script": (60, 300, 900),
            "debug_code": (120, 600, 1800),
            "refactor_code": (60, 300, 1200),

            # File operations
            "read_small_file": (1, 2, 10),
            "read_large_file": (5, 30, 120),
            "write_file": (1, 5, 30),

            # Kaggle specific
            "understand_competition": (300, 600, 1800),
            "prepare_submission": (30, 120, 300),
            "ensemble_models": (60, 300, 1200),
        }

    @property
    def name(self) -> str:
        return "EstimateTaskDuration"

    @property
    def schema(self) -> Dict:
        return {
            "name": "EstimateTaskDuration",
            "description": (
                "Get an estimate of how long a task should take. "
                "Useful for planning work, setting timeouts, and detecting stalled processes. "
                "Provides min/typical/max duration estimates based on task type and parameters."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "description": (
                            "Type of task to estimate. Examples: 'load_data', 'train_simple_model', "
                            "'train_complex_model', 'feature_engineering', 'cross_validation', "
                            "'explore_data', 'prepare_submission', etc."
                        )
                    },
                    "data_size_mb": {
                        "type": "number",
                        "description": (
                            "Optional. Size of data being processed in megabytes (MB). "
                            "Examples: 10 for 10MB, 1024 for 1GB, 10240 for 10GB. "
                            "If not provided, assumes medium size (~1GB). "
                            "Used to scale duration estimates for data operations."
                        )
                    },
                    "data_size": {
                        "type": "string",
                        "description": (
                            "DEPRECATED: Use data_size_mb instead. "
                            "Categorical size: 'small' (<1GB), 'medium' (1-10GB), 'large' (>10GB)."
                        ),
                        "enum": ["small", "medium", "large"]
                    },
                    "complexity": {
                        "type": "string",
                        "description": (
                            "Optional. Complexity level: 'simple', 'moderate', 'complex'. "
                            "Adjusts estimates for model training and code tasks."
                        ),
                        "enum": ["simple", "moderate", "complex"]
                    },
                    "additional_context": {
                        "type": "string",
                        "description": (
                            "Optional. Additional context about the task that might affect duration "
                            "(e.g., 'using GPU', 'distributed training', 'large dataset with 100+ columns')"
                        )
                    }
                },
                "required": ["task_type"]
            }
        }

    async def execute(self, input: Dict) -> Dict:
        """Estimate task duration"""
        try:
            task_type = input["task_type"].lower().replace(" ", "_")
            complexity = input.get("complexity", "moderate").lower()
            additional_context = input.get("additional_context", "")

            # Handle data size - prefer data_size_mb, fallback to categorical
            data_size_mb = input.get("data_size_mb")
            if data_size_mb is None:
                # Fallback to categorical if provided
                data_size_cat = input.get("data_size", "medium").lower()
                data_size_mb = self._categorical_to_mb(data_size_cat)

            # Get base estimate
            base_estimate = self._get_base_estimate(task_type)
            if base_estimate is None:
                return self._suggest_similar_tasks(task_type)

            min_dur, typ_dur, max_dur = base_estimate

            # Apply modifiers based on data size (using actual size)
            if any(keyword in task_type for keyword in ["data", "load", "process", "merge", "read", "write"]):
                size_multiplier = self._calculate_size_multiplier(data_size_mb)
                min_dur *= size_multiplier
                typ_dur *= size_multiplier
                max_dur *= size_multiplier

            # Apply modifiers based on complexity
            if complexity == "complex" and any(keyword in task_type for keyword in ["model", "train", "code"]):
                min_dur *= 1.5
                typ_dur *= 2
                max_dur *= 2.5
            elif complexity == "simple":
                min_dur *= 0.6
                typ_dur *= 0.7
                max_dur *= 0.8

            # Check for GPU acceleration in context
            if additional_context and "gpu" in additional_context.lower():
                if "train" in task_type or "deep_learning" in task_type:
                    typ_dur *= 0.3  # GPU training is ~3x faster
                    max_dur *= 0.4

            # Format output
            output = self._format_estimate(
                task_type=task_type,
                min_dur=min_dur,
                typ_dur=typ_dur,
                max_dur=max_dur,
                data_size_mb=data_size_mb,
                complexity=complexity,
                additional_context=additional_context
            )

            return {
                "content": output,
                "is_error": False,
                "debug_summary": f"Estimate for {task_type}: {self._format_duration(typ_dur)}"
            }

        except Exception as e:
            return {
                "content": f"Error estimating task duration: {str(e)}",
                "is_error": True
            }

    def _categorical_to_mb(self, category: str) -> float:
        """Convert categorical size to MB estimate"""
        mapping = {
            "small": 100,    # 100MB
            "medium": 1024,  # 1GB
            "large": 10240   # 10GB
        }
        return mapping.get(category, 1024)

    def _calculate_size_multiplier(self, size_mb: float) -> float:
        """
        Calculate duration multiplier based on data size.

        Uses a logarithmic scale to account for:
        - Sublinear scaling for small files (caching, memory ops)
        - Linear scaling for medium files
        - Superlinear scaling for large files (I/O bottlenecks, memory pressure)

        Baseline: 1GB (1024 MB) = 1.0x multiplier

        Examples:
            10 MB -> 0.2x (5x faster than 1GB)
            100 MB -> 0.5x (2x faster than 1GB)
            1 GB (1024 MB) -> 1.0x (baseline)
            10 GB (10240 MB) -> 3.0x (3x slower than 1GB)
            100 GB (102400 MB) -> 10.0x (10x slower than 1GB)
        """
        import math

        # Baseline: 1GB = 1024 MB
        baseline_mb = 1024

        if size_mb <= 0:
            return 0.2  # Minimum multiplier for tiny files

        # Use logarithmic scaling with different coefficients based on size
        ratio = size_mb / baseline_mb

        if ratio < 1:
            # Sublinear for small files (good caching behavior)
            # 10MB -> 0.2x, 100MB -> 0.5x, 500MB -> 0.8x
            multiplier = 0.2 + (0.8 * math.sqrt(ratio))
        else:
            # Superlinear for large files (I/O becomes bottleneck)
            # 1GB -> 1.0x, 10GB -> 3.0x, 100GB -> 10.0x
            multiplier = math.pow(ratio, 0.7)

        # Cap multipliers at reasonable bounds
        return max(0.1, min(multiplier, 50.0))

    def _get_base_estimate(self, task_type: str) -> Optional[tuple]:
        """Get base estimate for task type"""
        # Direct match
        if task_type in self.task_estimates:
            return self.task_estimates[task_type]

        # Fuzzy match - check if task_type contains any known keywords
        # First try: check if any known task contains the input
        for known_task, estimate in self.task_estimates.items():
            if task_type in known_task:
                return estimate

        # Second try: check if input contains any known task
        for known_task, estimate in self.task_estimates.items():
            if known_task in task_type:
                return estimate

        # Third try: word-level matching (e.g., "train_model" matches "train_simple_model")
        task_words = set(task_type.split("_"))
        for known_task, estimate in self.task_estimates.items():
            known_words = set(known_task.split("_"))
            # If task has significant overlap with known task
            if len(task_words & known_words) >= 2:
                return estimate

        return None

    def _suggest_similar_tasks(self, task_type: str) -> Dict:
        """Suggest similar tasks when exact match not found"""
        # Get all task types
        all_tasks = sorted(self.task_estimates.keys())

        # Group by category
        categories = {
            "Data Exploration": [t for t in all_tasks if any(k in t for k in ["explore", "visual", "profile"])],
            "Data Preprocessing": [t for t in all_tasks if any(k in t for k in ["clean", "feature", "missing", "encode", "scale"])],
            "Model Training": [t for t in all_tasks if any(k in t for k in ["train", "tuning", "validation"])],
            "Model Evaluation": [t for t in all_tasks if any(k in t for k in ["evaluate", "predict", "metrics"])],
            "Data Operations": [t for t in all_tasks if any(k in t for k in ["process", "merge", "aggregate"])],
            "Code Operations": [t for t in all_tasks if any(k in t for k in ["write", "debug", "refactor"])],
            "File Operations": [t for t in all_tasks if any(k in t for k in ["read", "write", "file"])],
            "Kaggle Specific": [t for t in all_tasks if any(k in t for k in ["competition", "submission", "ensemble"])],
        }

        output = f"❌ Unknown task type: '{task_type}'\n\n"
        output += "Available task types by category:\n\n"

        for category, tasks in categories.items():
            if tasks:
                output += f"📁 {category}:\n"
                for task in tasks:
                    output += f"   - {task}\n"
                output += "\n"

        output += "💡 Tip: Use the exact task_type name, or a close variation (e.g., 'train_model' matches 'train_simple_model')"

        return {
            "content": output,
            "is_error": True
        }

    def _format_estimate(
        self,
        task_type: str,
        min_dur: float,
        typ_dur: float,
        max_dur: float,
        data_size_mb: float,
        complexity: str,
        additional_context: str
    ) -> str:
        """Format the estimate output"""
        output = f"⏱️  Task Duration Estimate: {task_type}\n\n"

        output += "📊 Estimated Duration:\n"
        output += f"   ⚡ Best case:  {self._format_duration(min_dur)}\n"
        output += f"   📈 Typical:    {self._format_duration(typ_dur)}\n"
        output += f"   ⚠️  Worst case: {self._format_duration(max_dur)}\n\n"

        output += "Parameters:\n"
        output += f"   • Data size: {self._format_data_size(data_size_mb)}\n"
        output += f"   • Complexity: {complexity}\n"
        if additional_context:
            output += f"   • Context: {additional_context}\n"
        output += "\n"

        output += "💡 Recommendations:\n"

        if typ_dur < 60:
            output += "   • This is a quick task - should complete in under a minute\n"
        elif typ_dur < 300:
            output += "   • This is a medium task - budget 5-10 minutes\n"
        elif typ_dur < 1800:
            output += "   • This is a longer task - budget 10-30 minutes\n"
        else:
            output += "   • This is a long-running task - consider running in background\n"
            output += "   • Use Bash(run_in_background=True) and monitor with ReadBashOutput\n"

        if typ_dur > 300:
            output += f"   • Set timeout to at least {self._format_duration(max_dur * 1.5)}\n"
            output += "   • Consider checkpointing for tasks > 30 minutes\n"

        if "train" in task_type:
            output += "   • Monitor with ReadBashOutput to detect stalls\n"
            output += "   • Consider early stopping if no improvement\n"

        return output

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def _format_data_size(self, size_mb: float) -> str:
        """Format data size in human-readable format"""
        if size_mb < 1:
            return f"{size_mb * 1024:.1f} KB"
        elif size_mb < 1024:
            return f"{size_mb:.1f} MB"
        elif size_mb < 1024 * 1024:
            size_gb = size_mb / 1024
            return f"{size_gb:.2f} GB"
        else:
            size_tb = size_mb / (1024 * 1024)
            return f"{size_tb:.2f} TB"
