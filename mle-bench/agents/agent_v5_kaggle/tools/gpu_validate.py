"""
GPUValidate Tool - Verify GPU training is working correctly

Usage by agent:
    GPUValidate with framework='pytorch', model_size='small', batch_size=256
    GPUValidate with framework='lightgbm', rows=100000

Returns confirmation if GPU training is working, or error if CPU fallback detected.
"""
from typing import Dict
import asyncio
import time


class GPUValidateTool:
    """Tool for validating GPU training is working - timing-based detection"""

    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir

    @property
    def name(self) -> str:
        return "GPUValidate"

    @property
    def schema(self) -> Dict:
        return {
            "name": "GPUValidate",
            "description": (
                "Validate GPU training is working correctly by running a quick benchmark. "
                "Use this BEFORE training to verify GPU setup. "
                "Returns timing-based confirmation (more reliable than memory checks). "
                "Takes ~2-3 seconds to run."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "framework": {
                        "type": "string",
                        "enum": ["pytorch", "lightgbm", "xgboost"],
                        "description": "Framework to validate (pytorch/lightgbm/xgboost)"
                    },
                    "batch_size": {
                        "type": "number",
                        "description": "Batch size for PyTorch validation. Optional, default: 256"
                    }
                },
                "required": ["framework"]
            }
        }

    async def execute(self, input: Dict) -> Dict:
        """Execute GPU validation with timing benchmark"""
        framework = input["framework"]

        try:
            if framework == "pytorch":
                batch_size = input.get("batch_size", 256)
                success, message = await self._validate_pytorch(batch_size)
            elif framework == "lightgbm":
                success, message = await self._validate_lightgbm()
            elif framework == "xgboost":
                success, message = await self._validate_xgboost()
            else:
                return {
                    "content": f"❌ Unknown framework: {framework}",
                    "is_error": True
                }

            if success:
                return {
                    "content": (
                        f"✅ GPU validation PASSED for {framework}\n\n"
                        f"{message}\n\n"
                        f"GPU training is working correctly. Proceed with full training."
                    ),
                    "is_error": False
                }
            else:
                return {
                    "content": (
                        f"❌ GPU validation FAILED for {framework}\n\n"
                        f"{message}\n\n"
                        f"CPU fallback detected. Check your training code:\n"
                        f"- PyTorch: Ensure model and data use .to(device) or .cuda()\n"
                        f"- LightGBM: Use device_type='cuda' (not device='gpu')\n"
                        f"- XGBoost: Use tree_method='gpu_hist'\n"
                        f"\n"
                        f"Fix the issue before proceeding with full training."
                    ),
                    "is_error": True
                }

        except Exception as e:
            return {
                "content": f"❌ GPU validation error: {str(e)}",
                "is_error": True
            }

    async def _validate_pytorch(self, batch_size: int) -> tuple:
        """
        Validate PyTorch GPU with timing benchmark

        Expected on A100: 1-2s for 100 batches
        Expected on CPU: 10-20s for 100 batches (10-20x slower)
        """
        try:
            import torch
            import torch.nn as nn

            if not torch.cuda.is_available():
                return False, "CUDA not available: torch.cuda.is_available() = False"

            device = torch.device('cuda')

            # Simple CNN model
            model = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 10)
            ).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Warmup (GPU needs initialization)
            for _ in range(5):
                x = torch.randn(batch_size, 3, 224, 224, device=device)
                y = torch.randint(0, 10, (batch_size,), device=device)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

            torch.cuda.synchronize()

            # Benchmark 100 batches
            start = time.time()
            for _ in range(100):
                x = torch.randn(batch_size, 3, 224, 224, device=device)
                y = torch.randint(0, 10, (batch_size,), device=device)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

            torch.cuda.synchronize()
            elapsed = time.time() - start

            # Decision: GPU should be <10s, CPU >15s
            gpu_threshold = 10.0
            if elapsed < gpu_threshold:
                return True, f"Timing confirms GPU (100 batches in {elapsed:.1f}s, threshold <{gpu_threshold}s)"
            else:
                return False, f"Timing too slow ({elapsed:.1f}s >= {gpu_threshold}s threshold) - likely CPU training"

        except ImportError:
            return False, "PyTorch not installed"
        except Exception as e:
            return False, f"PyTorch validation failed: {str(e)}"

    async def _validate_lightgbm(self) -> tuple:
        """
        Validate LightGBM GPU with timing benchmark

        Expected on A100: 1-3s for 100 rounds
        Expected on CPU: 10-20s for 100 rounds (10-15x slower)
        """
        try:
            import lightgbm as lgb
            import numpy as np

            # Create small dataset
            X = np.random.rand(100000, 50)
            y = np.random.randint(0, 2, 100000)
            train_data = lgb.Dataset(X, label=y)

            # Try GPU training
            params = {
                'device_type': 'cuda',
                'objective': 'binary',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'verbose': -1
            }

            start = time.time()
            model = lgb.train(params, train_data, num_boost_round=100)
            elapsed = time.time() - start

            # Decision: GPU should be <5s, CPU >8s
            gpu_threshold = 5.0
            if elapsed < gpu_threshold:
                return True, f"Timing confirms GPU (100 rounds in {elapsed:.1f}s, threshold <{gpu_threshold}s)"
            else:
                return False, f"Timing too slow ({elapsed:.1f}s >= {gpu_threshold}s threshold) - likely CPU training"

        except ImportError:
            return False, "LightGBM not installed"
        except Exception as e:
            if "CUDA Tree Learner was not enabled" in str(e):
                return False, f"LightGBM not built with CUDA support: {str(e)}"
            return False, f"LightGBM validation failed: {str(e)}"

    async def _validate_xgboost(self) -> tuple:
        """
        Validate XGBoost GPU with timing benchmark

        Expected on A100: 1-2s for 100 rounds
        Expected on CPU: 8-15s for 100 rounds (8-10x slower)
        """
        try:
            import xgboost as xgb
            import numpy as np

            # Create small dataset
            X = np.random.rand(50000, 50)
            y = np.random.randint(0, 2, 50000)
            dtrain = xgb.DMatrix(X, label=y)

            # Try GPU training
            params = {
                'tree_method': 'gpu_hist',
                'objective': 'binary:logistic',
                'max_depth': 6,
                'learning_rate': 0.1,
                'verbosity': 0
            }

            start = time.time()
            model = xgb.train(params, dtrain, num_boost_round=100)
            elapsed = time.time() - start

            # Decision: GPU should be <4s, CPU >6s
            gpu_threshold = 4.0
            if elapsed < gpu_threshold:
                return True, f"Timing confirms GPU (100 rounds in {elapsed:.1f}s, threshold <{gpu_threshold}s)"
            else:
                return False, f"Timing too slow ({elapsed:.1f}s >= {gpu_threshold}s threshold) - likely CPU training"

        except ImportError:
            return False, "XGBoost not installed"
        except Exception as e:
            return False, f"XGBoost validation failed: {str(e)}"
