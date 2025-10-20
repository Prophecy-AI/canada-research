"""
EstimateDuration tool - Smart task duration estimation with adaptive time control

Estimates optimal time allocation based on:
1. Task type and complexity
2. Dataset size
3. Remaining time budget
4. Hardware capabilities (A100 GPU, 36 cores)

Then adapts recommendations (faster/slower) based on time constraints.
"""
import time
from typing import Dict, Tuple, Optional
from .base import BaseTool


class TaskDurationEstimator:
    """
    Core estimation logic for ML task durations

    Based on Kaggle Grandmaster playbook and empirical A100 performance data
    """

    # Hardware specs (A100 40GB, 36 cores)
    GPU_VRAM_GB = 40
    CPU_CORES = 36
    RAM_GB = 440

    # Base time estimates (minutes) for different task types on A100
    # Format: (min_time, typical_time, max_time)
    BASE_ESTIMATES = {
        # Computer Vision
        "image_classification": {
            "small": (8, 12, 18),      # <10K images, EfficientNet-B3
            "medium": (12, 20, 30),    # 10K-100K images, EfficientNet-B4
            "large": (20, 35, 50),     # >100K images, EfficientNet-B5
        },
        "image_segmentation": {
            "small": (10, 15, 22),     # <5K images, U-Net + B3 backbone
            "medium": (15, 25, 35),    # 5K-50K images, U-Net + B4 backbone
            "large": (25, 40, 60),     # >50K images, larger models
        },
        "object_detection": {
            "small": (8, 12, 18),      # <5K images, YOLOv5s
            "medium": (12, 20, 28),    # 5K-50K images, YOLOv8n
            "large": (20, 30, 45),     # >50K images, larger YOLO
        },

        # Tabular
        "tabular": {
            "small": (3, 5, 8),        # <100K rows, LightGBM
            "medium": (5, 8, 12),      # 100K-1M rows, LightGBM/XGBoost
            "large": (8, 15, 25),      # >1M rows, multiple GBDTs
        },

        # NLP
        "nlp_classification": {
            "small": (5, 8, 12),       # <50K texts, distilbert
            "medium": (8, 15, 22),     # 50K-500K texts, DeBERTa-small
            "large": (15, 25, 40),     # >500K texts, DeBERTa-base
        },

        # Time Series
        "time_series": {
            "small": (3, 5, 8),        # <100K rows, GBDT
            "medium": (5, 10, 15),     # 100K-1M rows, GBDT + features
            "large": (10, 18, 28),     # >1M rows, hybrid models
        },

        # Audio
        "audio": {
            "small": (8, 12, 18),      # <5K samples, mel-spec + CNN
            "medium": (12, 20, 30),    # 5K-50K samples
            "large": (20, 32, 45),     # >50K samples
        },
    }

    # Complexity multipliers
    COMPLEXITY_MULTIPLIERS = {
        "simple": 0.7,      # Single model, basic features
        "moderate": 1.0,    # Standard approach, 2-3 models
        "complex": 1.5,     # Ensemble, extensive features
        "very_complex": 2.0,  # Large ensemble, complex pipelines
    }

    # Parallel training efficiency
    # Running 2-3 models in parallel is ~1.3x slower than 1 model (not 2-3x)
    PARALLEL_EFFICIENCY = {
        1: 1.0,
        2: 1.3,
        3: 1.5,
        4: 1.8,
    }

    @classmethod
    def estimate_base_time(
        cls,
        task_type: str,
        dataset_size: str,
        complexity: str = "moderate",
        num_parallel_models: int = 1
    ) -> Tuple[float, float, float]:
        """
        Estimate base time requirements for a task

        Args:
            task_type: Type of ML task (image_classification, tabular, etc.)
            dataset_size: small/medium/large
            complexity: simple/moderate/complex/very_complex
            num_parallel_models: Number of models to train in parallel

        Returns:
            (min_time, typical_time, max_time) in minutes
        """
        # Get base estimate
        if task_type not in cls.BASE_ESTIMATES:
            # Unknown task type - use moderate estimate
            base = (10, 20, 30)
        else:
            task_estimates = cls.BASE_ESTIMATES[task_type]
            if dataset_size not in task_estimates:
                # Unknown size - use medium
                base = task_estimates.get("medium", (10, 20, 30))
            else:
                base = task_estimates[dataset_size]

        # Apply complexity multiplier
        complexity_mult = cls.COMPLEXITY_MULTIPLIERS.get(complexity, 1.0)

        # Apply parallel efficiency
        parallel_mult = cls.PARALLEL_EFFICIENCY.get(num_parallel_models, 2.0)

        # Calculate final estimates
        min_time = base[0] * complexity_mult * parallel_mult
        typical_time = base[1] * complexity_mult * parallel_mult
        max_time = base[2] * complexity_mult * parallel_mult

        return (min_time, typical_time, max_time)

    @classmethod
    def adaptive_time_allocation(
        cls,
        estimated_time: float,
        time_remaining: float,
        time_total: float = 30.0
    ) -> Dict:
        """
        Adapt strategy based on time remaining vs estimated time needed

        Args:
            estimated_time: Estimated time needed (minutes)
            time_remaining: Time remaining in budget (minutes)
            time_total: Total time budget (minutes)

        Returns:
            Dict with strategy recommendations
        """
        time_ratio = estimated_time / time_remaining if time_remaining > 0 else float('inf')
        time_used = time_total - time_remaining
        percent_used = (time_used / time_total) * 100

        # Determine urgency and strategy
        if time_ratio <= 0.6:
            # Plenty of time - can run full strategy
            urgency = "low"
            strategy = "full"
            guidance = (
                f"You have {time_remaining:.1f} min remaining, need ~{estimated_time:.1f} min.\n"
                f"Time is abundant - use FULL strategy:\n"
                f"• Run complete CV (3-5 folds)\n"
                f"• Use larger models (B4/B5 for CV, DeBERTa for NLP)\n"
                f"• Consider ensemble (2-3 models in parallel)\n"
                f"• Take time for proper validation\n"
                f"• Aim for best possible score"
            )
            speed_modifier = 1.0

        elif time_ratio <= 1.0:
            # Comfortable - run standard strategy
            urgency = "medium"
            strategy = "standard"
            guidance = (
                f"You have {time_remaining:.1f} min remaining, need ~{estimated_time:.1f} min.\n"
                f"Time is comfortable - use STANDARD strategy:\n"
                f"• Run 3-fold CV (standard)\n"
                f"• Use medium models (B3 for CV, distilbert for NLP)\n"
                f"• Single model or small ensemble (2 models)\n"
                f"• Focus on core approach\n"
                f"• Should finish on time"
            )
            speed_modifier = 1.0

        elif time_ratio <= 1.3:
            # Tight but feasible - optimize for speed
            urgency = "high"
            strategy = "fast"
            guidance = (
                f"You have {time_remaining:.1f} min remaining, need ~{estimated_time:.1f} min.\n"
                f"Time is TIGHT - use FAST strategy:\n"
                f"• Reduce to 2-fold CV (faster)\n"
                f"• Use smaller models (B2 for CV, distilbert for NLP)\n"
                f"• Single model only (no ensemble)\n"
                f"• Reduce epochs by 20-30%\n"
                f"• Increase batch size if possible\n"
                f"• May need to cut corners to finish"
            )
            speed_modifier = 0.7

        else:
            # Critical - emergency mode
            urgency = "critical"
            strategy = "emergency"
            guidance = (
                f"You have {time_remaining:.1f} min remaining, need ~{estimated_time:.1f} min.\n"
                f"Time is CRITICAL - use EMERGENCY strategy:\n"
                f"• NO CV - single train/val split\n"
                f"• Smallest viable model (B0 for CV, tiny for NLP)\n"
                f"• Minimal epochs (3-5 max)\n"
                f"• Large batch size (maximize speed)\n"
                f"• Consider simple baseline (LR, small GBDT)\n"
                f"• Accept lower score to finish on time\n"
                f"• OR: Skip training, use pretrained model directly if possible"
            )
            speed_modifier = 0.5

        return {
            "urgency": urgency,
            "strategy": strategy,
            "guidance": guidance,
            "time_ratio": time_ratio,
            "percent_used": percent_used,
            "speed_modifier": speed_modifier,
            "recommended_time": estimated_time * speed_modifier,
        }


class EstimateDurationTool(BaseTool):
    """
    Tool to estimate task duration and get adaptive time control recommendations

    Combines:
    1. Task-specific duration estimates (based on empirical data)
    2. Remaining time budget tracking
    3. Adaptive strategy recommendations (faster/slower)
    """

    def __init__(self, workspace_dir: str, start_time: float = None, total_budget_min: float = 30.0):
        """
        Initialize EstimateDuration tool

        Args:
            workspace_dir: Workspace directory
            start_time: Unix timestamp when agent started
            total_budget_min: Total time budget in minutes (default: 30)
        """
        super().__init__(workspace_dir)
        self.start_time = start_time if start_time is not None else time.time()
        self.total_budget_min = total_budget_min
        self.estimator = TaskDurationEstimator()

    @property
    def name(self) -> str:
        return "EstimateDuration"

    @property
    def schema(self) -> Dict:
        return {
            "name": "EstimateDuration",
            "description": (
                "Estimate how long a task should take and get adaptive time control recommendations.\n\n"
                "This tool combines:\n"
                "1. Task-specific duration estimates based on empirical A100 performance\n"
                "2. Remaining time budget tracking\n"
                "3. Adaptive strategy recommendations (full/standard/fast/emergency)\n\n"
                "Use this BEFORE training to:\n"
                "• Estimate if your planned approach fits in time budget\n"
                "• Get guidance on model size, CV folds, epochs\n"
                "• Decide between full strategy vs fast strategy\n\n"
                "Example inputs:\n"
                "- Image classification, 50K images → task_type='image_classification', dataset_size='medium'\n"
                "- Tabular, 500K rows → task_type='tabular', dataset_size='medium'\n"
                "- NLP, 100K texts → task_type='nlp_classification', dataset_size='medium'\n\n"
                "Returns time estimate AND adaptive recommendations based on time remaining."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "description": (
                            "Type of ML task. Options:\n"
                            "• image_classification: Standard image classification\n"
                            "• image_segmentation: Semantic/instance segmentation\n"
                            "• object_detection: Bounding box detection\n"
                            "• tabular: Tabular data (GBDT models)\n"
                            "• nlp_classification: Text classification\n"
                            "• time_series: Time series forecasting\n"
                            "• audio: Audio classification"
                        ),
                        "enum": [
                            "image_classification",
                            "image_segmentation",
                            "object_detection",
                            "tabular",
                            "nlp_classification",
                            "time_series",
                            "audio"
                        ]
                    },
                    "dataset_size": {
                        "type": "string",
                        "description": (
                            "Dataset size category:\n"
                            "• small: <10K images, <100K rows, <50K texts\n"
                            "• medium: 10K-100K images, 100K-1M rows, 50K-500K texts\n"
                            "• large: >100K images, >1M rows, >500K texts"
                        ),
                        "enum": ["small", "medium", "large"]
                    },
                    "complexity": {
                        "type": "string",
                        "description": (
                            "Task complexity:\n"
                            "• simple: Single model, basic features\n"
                            "• moderate: Standard approach, 2-3 models (DEFAULT)\n"
                            "• complex: Ensemble, extensive features\n"
                            "• very_complex: Large ensemble, complex pipelines"
                        ),
                        "enum": ["simple", "moderate", "complex", "very_complex"]
                    },
                    "num_parallel_models": {
                        "type": "integer",
                        "description": "Number of models to train in parallel (1-4). Default: 1"
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of the task for context (optional)"
                    }
                },
                "required": ["task_type", "dataset_size"]
            }
        }

    async def execute(self, input: Dict) -> Dict:
        """
        Estimate task duration and provide adaptive recommendations

        Args:
            input: Dict with task_type, dataset_size, complexity (optional),
                   num_parallel_models (optional), description (optional)

        Returns:
            Dict with time estimates and adaptive strategy recommendations
        """
        try:
            # Extract parameters
            task_type = input["task_type"]
            dataset_size = input["dataset_size"]
            complexity = input.get("complexity", "moderate")
            num_parallel_models = input.get("num_parallel_models", 1)
            description = input.get("description", "")

            # Calculate elapsed and remaining time
            elapsed_seconds = time.time() - self.start_time
            elapsed_minutes = elapsed_seconds / 60
            time_remaining = self.total_budget_min - elapsed_minutes

            # Get base time estimate
            min_time, typical_time, max_time = self.estimator.estimate_base_time(
                task_type=task_type,
                dataset_size=dataset_size,
                complexity=complexity,
                num_parallel_models=num_parallel_models
            )

            # Get adaptive recommendations
            adaptation = self.estimator.adaptive_time_allocation(
                estimated_time=typical_time,
                time_remaining=time_remaining,
                time_total=self.total_budget_min
            )

            # Build response
            response = (
                f"⏱️  TASK DURATION ESTIMATE\n"
                f"{'=' * 60}\n\n"
            )

            if description:
                response += f"Task: {description}\n\n"

            response += (
                f"📋 TASK DETAILS:\n"
                f"• Type: {task_type}\n"
                f"• Dataset size: {dataset_size}\n"
                f"• Complexity: {complexity}\n"
                f"• Parallel models: {num_parallel_models}\n\n"

                f"⏰ TIME ESTIMATES (A100 GPU):\n"
                f"• Optimistic: {min_time:.1f} minutes\n"
                f"• Typical: {typical_time:.1f} minutes\n"
                f"• Pessimistic: {max_time:.1f} minutes\n\n"

                f"📊 TIME BUDGET STATUS:\n"
                f"• Total budget: {self.total_budget_min:.1f} minutes\n"
                f"• Elapsed: {elapsed_minutes:.1f} minutes ({adaptation['percent_used']:.1f}% used)\n"
                f"• Remaining: {time_remaining:.1f} minutes\n"
                f"• Estimated need: {typical_time:.1f} minutes\n"
                f"• Time ratio: {adaptation['time_ratio']:.2f}x (estimate/remaining)\n\n"

                f"🎯 ADAPTIVE STRATEGY:\n"
                f"• Urgency: {adaptation['urgency'].upper()}\n"
                f"• Recommended strategy: {adaptation['strategy'].upper()}\n"
                f"• Speed modifier: {adaptation['speed_modifier']:.1f}x\n"
                f"• Adjusted target: {adaptation['recommended_time']:.1f} minutes\n\n"

                f"📝 GUIDANCE:\n"
                f"{adaptation['guidance']}\n\n"
            )

            # Add specific model recommendations based on strategy
            response += self._get_model_recommendations(
                task_type=task_type,
                strategy=adaptation['strategy']
            )

            return {
                "content": response,
                "is_error": False,
                "debug_summary": (
                    f"Task: {task_type}/{dataset_size}, "
                    f"Estimate: {typical_time:.1f}min, "
                    f"Remaining: {time_remaining:.1f}min, "
                    f"Strategy: {adaptation['strategy']}"
                )
            }

        except Exception as e:
            return {
                "content": f"Error estimating duration: {str(e)}",
                "is_error": True
            }

    def _get_model_recommendations(self, task_type: str, strategy: str) -> str:
        """Get specific model recommendations based on task type and strategy"""

        recommendations = {
            "image_classification": {
                "emergency": "EfficientNet-B0 or ResNet-34, single train/val split, 3-5 epochs",
                "fast": "EfficientNet-B2, 2-fold CV, 6-8 epochs, batch_size=384",
                "standard": "EfficientNet-B3, 3-fold CV, 8-10 epochs, batch_size=256, MixUp",
                "full": "EfficientNet-B4/B5, 5-fold CV, 10-15 epochs, batch_size=192, MixUp+CutMix, ensemble 2-3 models"
            },
            "image_segmentation": {
                "emergency": "U-Net + ResNet-34 backbone, 256x256 tiles, single split, 5 epochs",
                "fast": "U-Net + EfficientNet-B2, 256x256 tiles, 2-fold CV, 8 epochs",
                "standard": "U-Net + EfficientNet-B3, 512x512 tiles, 3-fold CV, 10 epochs",
                "full": "U-Net + EfficientNet-B4, 512x512 tiles, 5-fold CV, 12 epochs, TTA"
            },
            "object_detection": {
                "emergency": "YOLOv5n pretrained, fine-tune 3 epochs",
                "fast": "YOLOv5s, fine-tune 5 epochs, 512x512 images",
                "standard": "YOLOv8n, fine-tune 8-10 epochs, 640x640 images",
                "full": "YOLOv8s/m, fine-tune 10-15 epochs, TTA, ensemble"
            },
            "tabular": {
                "emergency": "LightGBM single model, default params, no feature engineering",
                "fast": "LightGBM, 2-fold CV, minimal features, early stopping",
                "standard": "LightGBM + XGBoost, 3-fold CV, basic feature engineering",
                "full": "LightGBM + XGBoost + CatBoost ensemble, 5-fold CV, extensive features, stacking"
            },
            "nlp_classification": {
                "emergency": "TF-IDF + LogisticRegression (fastest) OR distilbert 1 epoch single split",
                "fast": "distilbert-base-uncased, max_length=128, 1 epoch, 2-fold CV",
                "standard": "DeBERTa-v3-small, max_length=256, 2 epochs, 3-fold CV",
                "full": "DeBERTa-v3-base, max_length=512, 3 epochs, 5-fold CV, ensemble"
            },
            "time_series": {
                "emergency": "LightGBM with basic lag features, single split",
                "fast": "LightGBM with lag + rolling features, TimeSeriesSplit(n=2)",
                "standard": "LightGBM + XGBoost, lag + rolling + date features, TimeSeriesSplit(n=3)",
                "full": "GBDT ensemble + LSTM/Transformer, extensive features, TimeSeriesSplit(n=5)"
            },
            "audio": {
                "emergency": "Mel-spectrogram + tiny CNN, single split, 5 epochs",
                "fast": "Mel-spectrogram + EfficientNet-B0, 2-fold CV, 8 epochs",
                "standard": "Mel-spectrogram + EfficientNet-B2/ResNet-50, 3-fold CV, 10 epochs",
                "full": "Mel-spectrogram + EfficientNet-B3, 5-fold CV, 12 epochs, ensemble"
            }
        }

        model_rec = recommendations.get(task_type, {}).get(strategy, "Use standard approach for this task type")

        return (
            f"🤖 MODEL RECOMMENDATIONS ({strategy.upper()} strategy):\n"
            f"• {model_rec}\n"
        )
