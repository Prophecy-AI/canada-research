"""
Memory System for Kaggle Agent
Learns from past competitions to improve future performance

Based on Kaggle Grandmaster Playbook (/home/kaggle_competition_strategy.txt)
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import os


class CompetitionMemory:
    """Stores and retrieves knowledge from past competitions"""

    def __init__(self, memory_dir: Optional[str] = None):
        if memory_dir is None:
            # Use environment variable or default
            memory_dir = os.environ.get('KAGGLE_MEMORY_DIR', '/home/.kaggle_memory')

        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.patterns_file = self.memory_dir / "patterns.json"
        self.strategies_file = self.memory_dir / "strategies.pkl"

        self.patterns = self._load_patterns()
        self.strategies = self._load_strategies()

    def _load_patterns(self) -> Dict:
        """Load learned patterns from disk"""
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, 'r') as f:
                    return json.load(f)
            except:
                pass

        # Default patterns based on Kaggle Grandmaster Playbook
        # Source: /home/kaggle_competition_strategy.txt
        return {
            "image_classification": {
                "small_dataset": {  # <10K samples
                    "best_models": ["EfficientNet-B0", "ResNet-34", "MobileNet-V2"],
                    "best_strategies": [
                        "3-fold StratifiedKFold CV",
                        "3-5 epochs with early stopping (patience=3)",
                        "224x224 images (upscale if needed)",
                        "Basic augmentation: HorizontalFlip, VerticalFlip, RandomRotate90",
                        "Transfer learning from pretrained (timm/torchvision)",
                        "Consider bottleneck features approach (fast)"
                    ],
                    "avoid": ["EfficientNet-B4+", "5-fold CV", ">8 epochs", "training from scratch"],
                    "advanced_techniques": ["Simple ensemble (2-3 models)", "TTA (+0.5-1%)"],
                    "typical_time_min": "8-12",
                    "cv_strategy": "StratifiedKFold (k=3)",
                    "common_pitfalls": ["Overfitting due to small data", "Not using pretrained weights"],
                    "expected_medal": "bronze-silver"
                },
                "medium_dataset": {  # 10K-100K samples
                    "best_models": ["EfficientNet-B2", "EfficientNet-B3", "ResNet-50", "DenseNet121"],
                    "best_strategies": [
                        "3-fold StratifiedKFold CV",
                        "6-8 epochs with early stopping",
                        "224x224 or 256x256 images",
                        "Advanced augmentation: MixUp/CutMix (+1-2% accuracy)",
                        "Fine-tuning pretrained models",
                        "Multi-architecture ensemble for diversity"
                    ],
                    "avoid": ["EfficientNet-B4+ (too slow for 30 min)", ">10 epochs", "5-fold CV unless <20K samples"],
                    "advanced_techniques": [
                        "MixUp/CutMix augmentation",
                        "Weighted ensemble (based on CV scores)",
                        "TTA with multiple augmentations",
                        "Progressive unfreezing (if time allows)"
                    ],
                    "typical_time_min": "15-25",
                    "cv_strategy": "StratifiedKFold (k=3)",
                    "common_pitfalls": ["Augmenting validation set", "Overfitting to public LB"],
                    "expected_medal": "silver-gold"
                },
                "large_dataset": {  # >100K samples
                    "best_models": ["EfficientNet-B3", "EfficientNet-B4 (40-60 min budget)", "ResNeXt50"],
                    "best_strategies": [
                        "2-fold CV (speed) or 3-fold (robustness)",
                        "6-8 epochs with early stopping",
                        "256x256 images (balance speed/accuracy)",
                        "MixUp/CutMix augmentation",
                        "Fine-tuning with progressive unfreezing",
                        "Use A10 GPU efficiently: batch_size=128+"
                    ],
                    "avoid": ["5-fold CV", ">300x300 images", "EfficientNet-B5+ (60+ min)"],
                    "advanced_techniques": [
                        "Stacking (meta-model on OOF predictions)",
                        "Pseudo-labeling (if initial model strong)",
                        "TTA with geometric augmentations"
                    ],
                    "typical_time_min": "20-30",
                    "cv_strategy": "StratifiedKFold (k=2-3)",
                    "common_pitfalls": ["Model too large for time budget", "Not using mixed precision"],
                    "expected_medal": "silver-gold"
                },
                "medical_imaging": {  # Special case: DICOM, grayscale, X-rays
                    "best_models": ["ResNet-50", "ResNeXt50", "DenseNet121"],
                    "best_strategies": [
                        "ResNet/ResNeXt better than EfficientNet for grayscale",
                        "CLAHE for contrast enhancement",
                        "Convert DICOM to Hounsfield Units",
                        "3-fold CV, 8-10 epochs",
                        "Geometric augmentations only (avoid color)"
                    ],
                    "preprocessing": ["DICOM → Hounsfield Units", "Resample to isotropic", "Segment ROI", "CLAHE"],
                    "typical_time_min": "15-25",
                    "expected_medal": "silver"
                }
            },
            "image_segmentation": {
                "any_size": {
                    "best_models": ["U-Net + EfficientNet-B0 backbone", "U-Net + ResNet-34", "FPN"],
                    "best_strategies": [
                        "Train on 256x256 or 512x512 tiles",
                        "3-fold CV",
                        "5-10 epochs with early stopping",
                        "Basic geometric augmentations (flips, rotations)",
                        "Overlapping tiles for inference (predict center only)"
                    ],
                    "avoid": ["512x512+ tiles (slow)", "5-fold CV", "Complex augmentations"],
                    "advanced_techniques": ["TTA with flips/rotations", "Edge prediction refinement"],
                    "typical_time_min": "15-25",
                    "cv_strategy": "KFold (k=3)",
                    "common_pitfalls": ["Edge effects from tiles", "Class imbalance in pixels"],
                    "expected_medal": "bronze-silver"
                }
            },
            "object_detection": {
                "2d_detection": {
                    "best_models": ["YOLOv5s", "YOLOv8n", "Faster R-CNN (if time allows)"],
                    "best_strategies": [
                        "Start with pretrained weights",
                        "Fine-tune 5-10 epochs",
                        "512x512 images (balance speed/accuracy)",
                        "3-fold CV",
                        "Use small variants (s/n) not large (x/l)"
                    ],
                    "avoid": ["YOLOv5x/YOLOv8x (too slow)", "Training from scratch"],
                    "typical_time_min": "10-20",
                    "cv_strategy": "KFold (k=3)",
                    "expected_medal": "bronze-silver"
                },
                "3d_detection": {
                    "best_models": ["PointPillars"],
                    "best_strategies": ["Fast 3D detection", "Pretrained weights", "5-10 epochs fine-tune"],
                    "typical_time_min": "15-25",
                    "expected_medal": "bronze"
                }
            },
            "tabular": {
                "any_size": {
                    "best_models": ["LightGBM (fastest)", "XGBoost (stable)", "CatBoost (categoricals)"],
                    "best_strategies": [
                        "LightGBM default for 30-min budget (fastest)",
                        "3-fold CV (TimeSeriesSplit if time-ordered)",
                        "Minimal feature engineering for speed",
                        "Feature interactions: A*B, A/B, polynomials (A²)",
                        "Group aggregations: mean/std by category",
                        "Target encoding for high-cardinality categoricals",
                        "Default hyperparameters + early stopping"
                    ],
                    "avoid": ["5-fold CV", "Heavy feature engineering (time sink)", "Deep neural nets (slower)"],
                    "advanced_techniques": [
                        "GBDT ensemble (LightGBM + XGBoost + CatBoost): +0.5-2%",
                        "GBDT + NN hybrid: +0.5-1% (adds 5-10 min)",
                        "Stacking with meta-model",
                        "Feature selection based on importance"
                    ],
                    "typical_time_min": "5-15",
                    "cv_strategy": "StratifiedKFold (imbalanced) or KFold (balanced) or TimeSeriesSplit (time-ordered)",
                    "common_pitfalls": [
                        "Data leakage (target encoding on full data)",
                        "Train-test contamination (fitting scaler on test)",
                        "Not handling missing values properly"
                    ],
                    "expected_medal": "silver-gold"
                }
            },
            "nlp": {
                "any_size": {
                    "best_models": ["distilbert-base-uncased (fastest)", "DeBERTa-small", "RoBERTa-base"],
                    "model_evolution": "BERT → RoBERTa → DeBERTa (current best)",
                    "best_strategies": [
                        "Fine-tune pretrained from Hugging Face",
                        "1-2 epochs only (diminishing returns after)",
                        "max_length=128 or 256 (not 512, too slow)",
                        "3-fold StratifiedKFold or GroupKFold",
                        "For simple tasks: TF-IDF + LogisticRegression (fast baseline)"
                    ],
                    "avoid": ["Large models (bert-large, >2 epochs)", "max_length>512"],
                    "advanced_techniques": [
                        "Knowledge distillation (teacher→student)",
                        "Longformer for long sequences",
                        "BiLSTM on top of transformer embeddings",
                        "Ensemble different transformer architectures"
                    ],
                    "typical_time_min": "10-20",
                    "cv_strategy": "StratifiedKFold (classification) or GroupKFold (grouped data)",
                    "common_pitfalls": ["Training too many epochs", "Not monitoring forums for new models"],
                    "meta_game": "NLP is architectural meta-game - monitor forums for breakthrough models",
                    "expected_medal": "bronze-silver"
                }
            },
            "time_series": {
                "any_size": {
                    "best_models": ["LightGBM with engineered features"],
                    "best_strategies": [
                        "Transform to tabular + use GBDTs (wins most competitions)",
                        "Time-based features: hour, day, month, day_of_week, holiday",
                        "Lag features: past values at different time steps",
                        "Rolling statistics: mean/std/min/max over windows",
                        "MANDATORY: TimeSeriesSplit CV (prevents future leakage)",
                        "3-fold TimeSeriesSplit"
                    ],
                    "avoid": ["Standard KFold (causes leakage)", "RNNs/LSTMs (slower, often worse)"],
                    "feature_engineering": [
                        "Lag features (t-1, t-2, t-7, t-30)",
                        "Rolling windows (mean_7d, std_30d)",
                        "Time-based (hour, dayofweek, month, is_weekend, is_holiday)",
                        "Difference features (value - value_lag1)"
                    ],
                    "typical_time_min": "8-15",
                    "cv_strategy": "TimeSeriesSplit (MANDATORY)",
                    "common_pitfalls": ["Using standard CV (future leakage)", "Not handling seasonality"],
                    "expected_medal": "bronze-silver"
                }
            },
            "audio": {
                "any_size": {
                    "best_models": ["EfficientNet-B0 on mel-spectrograms", "ResNet-34", "Simple 2D CNN"],
                    "best_strategies": [
                        "Convert audio to 2D mel-spectrograms",
                        "Treat as image classification problem",
                        "Apply CV strategies (see image_classification)",
                        "3-fold CV, 3-5 epochs",
                        "Standard image augmentations work"
                    ],
                    "preprocessing": ["Audio → Mel-spectrogram (maps to human perception)", "Normalize"],
                    "avoid": ["Raw audio waveforms", "Large models"],
                    "typical_time_min": "10-20",
                    "cv_strategy": "StratifiedKFold (k=3)",
                    "expected_medal": "bronze-silver"
                }
            },
            "recommender_systems": {
                "any_size": {
                    "best_models": ["Hybrid: Collaborative Filtering + Content-Based"],
                    "best_strategies": [
                        "Collaborative filtering for user-item interactions",
                        "Content-based filtering for item features",
                        "Deep learning: Embedding layers for users/items",
                        "Matrix factorization + NN"
                    ],
                    "typical_time_min": "15-25",
                    "expected_medal": "bronze-silver"
                }
            }
        }

    def _load_strategies(self) -> Dict:
        """Load successful strategies from disk"""
        if self.strategies_file.exists():
            try:
                with open(self.strategies_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass

        # Default parallel training strategies from playbook Part VI
        return {
            "parallel_patterns": {
                "image_classification": {
                    "models": ["LightGBM (features)", "ResNet-34", "EfficientNet-B0"],
                    "resource_split": ["12 cores CPU", "12 cores + 8GB GPU", "12 cores + 8GB GPU"],
                    "time_min": "10-12",
                    "diversity_bonus": "+1-3%",
                    "description": "Extract features with LightGBM while training 2 CNNs in parallel"
                },
                "tabular": {
                    "models": ["LightGBM", "XGBoost", "CatBoost"],
                    "resource_split": ["12 cores", "12 cores", "12 cores"],
                    "time_min": "8-10",
                    "diversity_bonus": "+0.5-2%",
                    "description": "Train 3 different GBDT models simultaneously, ensemble results"
                },
                "tabular_mixed": {
                    "models": ["LightGBM", "Tabular NN"],
                    "resource_split": ["18 cores CPU", "18 cores + 24GB GPU"],
                    "time_min": "10",
                    "diversity_bonus": "+0.5-1%",
                    "description": "GBDT + Neural Net hybrid approach"
                }
            },
            "ensemble_strategies": {
                "simple_blending": {
                    "method": "Weighted average based on CV scores",
                    "weights_formula": "weight_i = cv_score_i / sum(cv_scores)",
                    "time_cost": "+1-2 min",
                    "boost": "+0.5-1%"
                },
                "stacking": {
                    "method": "Meta-model trained on out-of-fold predictions",
                    "time_cost": "+3-5 min",
                    "boost": "+0.5-1% over simple blending",
                    "requires": "Time budget allows (use if <25 min spent)"
                }
            },
            "advanced_techniques": {
                "mixup_cutmix": {
                    "applies_to": ["image_classification"],
                    "boost": "+1-2% accuracy",
                    "description": "Forces smooth decision boundaries",
                    "avoid_on": "validation set"
                },
                "tta": {
                    "applies_to": ["image_classification", "image_segmentation"],
                    "boost": "+0.5-1%",
                    "time_cost": "2x inference time",
                    "description": "Multiple augmented versions → average predictions"
                },
                "pseudo_labeling": {
                    "applies_to": ["any"],
                    "boost": "Variable",
                    "requires": "Initial model must be strong",
                    "description": "Use predictions on test as labels, retrain on train+test"
                }
            }
        }

    def get_strategy_for_competition(
        self,
        data_type: str,
        dataset_size: int,
        time_budget_min: int = 30
    ) -> Dict:
        """Get recommended strategy based on competition characteristics"""

        # Normalize data type
        data_type_map = {
            "image": "image_classification",
            "images": "image_classification",
            "tabular": "tabular",
            "text": "nlp",
            "nlp": "nlp",
            "time_series": "time_series",
            "audio": "audio",
            "segmentation": "image_segmentation",
            "detection": "object_detection"
        }

        normalized_type = data_type_map.get(data_type.lower(), data_type.lower())

        # Get size category for image classification
        if normalized_type == "image_classification":
            if dataset_size < 10000:
                size_category = "small_dataset"
            elif dataset_size < 100000:
                size_category = "medium_dataset"
            else:
                size_category = "large_dataset"

            pattern = self.patterns.get(normalized_type, {}).get(size_category, {})
        elif normalized_type == "object_detection":
            # Check if 2D or 3D based on data characteristics (default to 2D)
            size_category = "2d_detection"
            pattern = self.patterns.get(normalized_type, {}).get(size_category, {})
        else:
            pattern = self.patterns.get(normalized_type, {}).get("any_size", {})

        if not pattern:
            return {
                "strategies": [],
                "models": [],
                "warning": f"Unknown data type: {data_type}"
            }

        # Check if parallel training is beneficial
        estimated_time = self._parse_time_range(pattern.get("typical_time_min", "20"))
        use_parallel = estimated_time > time_budget_min * 0.8  # If close to budget

        recommendations = {
            "data_type": normalized_type,
            "dataset_size": dataset_size,
            "recommended_models": pattern.get("best_models", []),
            "recommended_strategies": pattern.get("best_strategies", []),
            "avoid": pattern.get("avoid", []),
            "advanced_techniques": pattern.get("advanced_techniques", []),
            "estimated_time_min": pattern.get("typical_time_min", "15-25"),
            "cv_strategy": pattern.get("cv_strategy", "KFold (k=3)"),
            "common_pitfalls": pattern.get("common_pitfalls", []),
            "expected_medal": pattern.get("expected_medal", "bronze-silver"),
            "use_parallel_training": use_parallel
        }

        # Add parallel training recommendation if beneficial
        if use_parallel:
            parallel_key = "tabular" if normalized_type == "tabular" else "image_classification"
            parallel_info = self.strategies.get("parallel_patterns", {}).get(parallel_key, {})
            if parallel_info:
                recommendations["parallel_training"] = parallel_info

        # Add preprocessing for special cases
        if "preprocessing" in pattern:
            recommendations["preprocessing"] = pattern["preprocessing"]

        # Add feature engineering for time series/tabular
        if "feature_engineering" in pattern:
            recommendations["feature_engineering"] = pattern["feature_engineering"]

        return recommendations

    def _parse_time_range(self, time_str: str) -> float:
        """Parse time range string to average minutes"""
        try:
            if '-' in time_str:
                low, high = time_str.split('-')
                return (float(low) + float(high)) / 2
            return float(time_str)
        except:
            return 20.0  # default

    def record_competition_result(
        self,
        competition_id: str,
        data_type: str,
        dataset_size: int,
        strategy: str,
        models_used: List[str],
        final_score: float,
        time_minutes: float,
        medal: str,
        notes: str = ""
    ):
        """Record competition result to learn from"""

        record = {
            "competition_id": competition_id,
            "timestamp": datetime.now().isoformat(),
            "data_type": data_type,
            "dataset_size": dataset_size,
            "strategy": strategy,
            "models_used": models_used,
            "final_score": final_score,
            "time_minutes": time_minutes,
            "medal": medal,
            "notes": notes
        }

        # Append to history file
        history_file = self.memory_dir / "competition_history.jsonl"
        with open(history_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

        # Update patterns if this was particularly successful
        self._update_patterns_if_better(record)

    def _update_patterns_if_better(self, record: Dict):
        """Update patterns if new record shows better approach"""

        # Medal ranking
        medal_rank = {"gold": 3, "silver": 2, "bronze": 1, "none": 0}

        data_type = record["data_type"]
        if data_type not in self.patterns:
            return

        # Determine size category
        if data_type == "image_classification":
            if record["dataset_size"] < 10000:
                size_key = "small_dataset"
            elif record["dataset_size"] < 100000:
                size_key = "medium_dataset"
            else:
                size_key = "large_dataset"
        else:
            size_key = "any_size"

        if size_key not in self.patterns[data_type]:
            return

        current_pattern = self.patterns[data_type][size_key]
        current_expected = current_pattern.get("expected_medal", "bronze")

        # If achieved better medal and faster time, update
        achieved_medal = record.get("medal", "none")
        if medal_rank.get(achieved_medal, 0) >= medal_rank.get(current_expected, 0):
            # Update expected medal if better
            if medal_rank.get(achieved_medal, 0) > medal_rank.get(current_expected, 0):
                current_pattern["expected_medal"] = achieved_medal

            # Add models to best_models if not present
            for model in record.get("models_used", []):
                if model not in current_pattern.get("best_models", []):
                    current_pattern.setdefault("best_models", []).append(model)

            # Save updated patterns
            self._save_patterns()

    def _save_patterns(self):
        """Save patterns to disk"""
        with open(self.patterns_file, 'w') as f:
            json.dump(self.patterns, f, indent=2)

    def get_memory_summary(self) -> str:
        """Get formatted summary of learned patterns"""
        summary = ["📚 **COMPETITION MEMORY INSIGHTS:**\n"]

        for data_type, patterns in self.patterns.items():
            summary.append(f"\n**{data_type.upper().replace('_', ' ')}:**")
            for size_cat, info in patterns.items():
                summary.append(f"\n  {size_cat}:")
                summary.append(f"    • Models: {', '.join(info.get('best_models', []))}")
                summary.append(f"    • Time: {info.get('typical_time_min', 'unknown')} min")
                summary.append(f"    • Expected: {info.get('expected_medal', 'unknown')} medal")
                if info.get('avoid'):
                    summary.append(f"    • Avoid: {', '.join(info.get('avoid', []))[:80]}")
                if info.get('cv_strategy'):
                    summary.append(f"    • CV: {info.get('cv_strategy')}")

        # Count competitions in history
        history_file = self.memory_dir / "competition_history.jsonl"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    count = sum(1 for _ in f)
                summary.append(f"\n\n**Total competitions learned from: {count}**")
            except:
                pass

        return "\n".join(summary)

    def get_similar_competitions(
        self,
        data_type: str,
        dataset_size: int,
        limit: int = 5
    ) -> List[Dict]:
        """Find similar past competitions"""
        history_file = self.memory_dir / "competition_history.jsonl"
        if not history_file.exists():
            return []

        similar = []
        with open(history_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record["data_type"] == data_type:
                        size_ratio = record["dataset_size"] / max(dataset_size, 1)
                        if 0.3 < size_ratio < 3.0:  # Within 3x size range
                            similar.append(record)
                except:
                    continue

        # Sort by medal (best first), then by time (fastest first)
        medal_rank = {"gold": 3, "silver": 2, "bronze": 1, "none": 0}
        similar.sort(
            key=lambda x: (medal_rank.get(x.get("medal", "none"), 0), -x.get("time_minutes", 999)),
            reverse=True
        )

        return similar[:limit]
