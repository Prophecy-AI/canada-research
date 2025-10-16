"""
Write tool for creating and overwriting files
"""
import os
from typing import Dict
from .base import BaseTool


class WriteTool(BaseTool):
    """Write content to files in workspace"""

    @property
    def name(self) -> str:
        return "Write"

    @property
    def schema(self) -> Dict:
        return {
            "name": "Write",
            "description": "Write content to a file (creates or overwrites)",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to file"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write"
                    }
                },
                "required": ["file_path", "content"]
            }
        }

    async def execute(self, input: Dict) -> Dict:
        file_path = input["file_path"]
        content = input["content"]

        try:
            if not file_path.startswith('/'):
                file_path = os.path.join(self.workspace_dir, file_path)

            file_exists = os.path.exists(file_path)

            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'w') as f:
                f.write(content)

            if file_exists:
                message = f"File updated successfully at: {file_path}"
            else:
                message = f"File created successfully at: {file_path}"

            # Add GPU reminder if this looks like a training/ML script
            gpu_hint = self._check_gpu_usage(file_path, content)
            if gpu_hint:
                message += f"\n\n{gpu_hint}"

            return {
                "content": message,
                "is_error": False,
                "debug_summary": f"{len(content)} bytes, {content.count(chr(10))+1} lines"
            }

        except Exception as e:
            return {
                "content": f"Error writing file: {str(e)}",
                "is_error": True
            }

    def _check_gpu_usage(self, file_path: str, content: str) -> str:
        """Check if file should use GPU and return reminder if needed"""
        # Only check Python files
        if not file_path.endswith('.py'):
            return ""
        
        # Check if filename suggests training/ML task
        filename_lower = os.path.basename(file_path).lower()
        is_ml_script = any(keyword in filename_lower for keyword in [
            'train', 'model', 'fit', 'predict', 'inference', 'cv', 'fold'
        ])
        
        if not is_ml_script:
            return ""
        
        # Check for FORBIDDEN sklearn imports (CPU-only alternatives to cuML)
        forbidden_sklearn = [
            'sklearn.feature_extraction.text',
            'sklearn.linear_model',
            'sklearn.ensemble.RandomForest',
            'sklearn.ensemble.GradientBoosting',
            'sklearn.neighbors',
            'sklearn.cluster',
            'sklearn.decomposition.PCA',
            'sklearn.svm'
        ]
        
        found_forbidden = []
        for forbidden in forbidden_sklearn:
            if forbidden in content or forbidden.replace('sklearn', 'from sklearn') in content:
                found_forbidden.append(forbidden)
        
        if found_forbidden:
            replacement_map = {
                'sklearn.feature_extraction.text': 'cuml.feature_extraction.text (GPU-accelerated)',
                'sklearn.linear_model': 'cuml.linear_model (GPU-accelerated)',
                'sklearn.ensemble.RandomForest': 'cuml.ensemble.RandomForestClassifier (GPU-accelerated)',
                'sklearn.ensemble.GradientBoosting': 'xgboost with tree_method="gpu_hist" or lightgbm with device="gpu"',
                'sklearn.neighbors': 'cuml.neighbors (GPU-accelerated)',
                'sklearn.cluster': 'cuml.cluster (GPU-accelerated)',
                'sklearn.decomposition.PCA': 'cuml.decomposition.PCA (GPU-accelerated)',
                'sklearn.svm': 'cuml.svm (GPU-accelerated)'
            }
            
            warnings = ["🔴 FORBIDDEN CPU-ONLY IMPORTS DETECTED:"]
            for forbidden in found_forbidden:
                replacement = replacement_map.get(forbidden, 'cuML equivalent')
                warnings.append(f"   • {forbidden} → USE {replacement} INSTEAD")
            
            warnings.append("")
            warnings.append("⚠️  This code will run 10-100x SLOWER on CPU than GPU alternatives.")
            warnings.append("File written to disk, but you MUST rewrite with cuML before running.")
            warnings.append("")
            warnings.append("Quick fix:")
            warnings.append("  from cuml.feature_extraction.text import TfidfVectorizer  # GPU")
            warnings.append("  from cuml.linear_model import LogisticRegression          # GPU")
            
            return "\n".join(warnings)
        
        # Check if content contains ML frameworks
        content_lower = content.lower()
        has_pytorch = 'import torch' in content_lower or 'from torch' in content_lower
        has_xgboost = 'import xgboost' in content_lower or 'from xgboost' in content_lower
        has_lightgbm = 'import lightgbm' in content_lower or 'from lightgbm' in content_lower
        has_tensorflow = 'import tensorflow' in content_lower or 'from tensorflow' in content_lower
        has_catboost = 'import catboost' in content_lower or 'from catboost' in content_lower
        has_cuml = 'import cuml' in content_lower or 'from cuml' in content_lower
        
        if not (has_pytorch or has_xgboost or has_lightgbm or has_tensorflow or has_catboost or has_cuml):
            return ""
        
        # Check if GPU is already configured
        gpu_configured = False
        
        if has_pytorch:
            gpu_configured = any(indicator in content for indicator in [
                '.cuda()', '.to(\'cuda\')', '.to("cuda")', 'device=\'cuda\'', 'device="cuda"',
                'torch.device(\'cuda\')', 'torch.device("cuda")'
            ])
        
        if has_xgboost:
            gpu_configured = gpu_configured or 'tree_method' in content_lower and 'gpu_hist' in content_lower
        
        if has_lightgbm:
            gpu_configured = gpu_configured or 'device' in content_lower and ('gpu' in content_lower or 'cuda' in content_lower)
        
        if has_catboost:
            gpu_configured = gpu_configured or 'task_type' in content and 'GPU' in content
        
        if has_tensorflow:
            # TensorFlow auto-detects GPU, so assume configured
            gpu_configured = True
        
        # Return hint if GPU not configured
        if not gpu_configured:
            hints = []
            if has_pytorch:
                hints.append("PyTorch: Add device = torch.device('cuda') and .to(device)")
            if has_xgboost:
                hints.append("XGBoost: Add 'tree_method': 'gpu_hist' to params")
            if has_lightgbm:
                hints.append("LightGBM: Add 'device': 'gpu' to params")
            if has_catboost:
                hints.append("CatBoost: Add task_type='GPU' to constructor")
            
            return f"⚠️  GPU CHECK: This script may not be using GPU!\n" + "\n".join(f"   • {hint}" for hint in hints) + "\n   • CPU training is 10-100x slower - verify GPU usage before running"
        
        # Check for resource maximization issues
        resource_issues = []
        
        # Check for n_jobs
        if (has_cuml or 'sklearn' in content_lower) and 'n_jobs' not in content_lower:
            resource_issues.append("Missing n_jobs=-1 (not using all CPU cores)")
        elif 'n_jobs=1' in content_lower or 'n_jobs = 1' in content_lower:
            resource_issues.append("n_jobs=1 found - change to n_jobs=-1 to use all cores")
        
        # Check for PyTorch DataLoader optimization
        if has_pytorch and 'DataLoader' in content:
            if 'num_workers' not in content:
                resource_issues.append("PyTorch DataLoader missing num_workers (should use all CPU cores)")
            if 'pin_memory' not in content:
                resource_issues.append("PyTorch DataLoader missing pin_memory=True (faster GPU transfer)")
        
        # Check for batch size configuration
        if (has_pytorch or has_tensorflow) and 'batch_size' not in content_lower:
            resource_issues.append("No batch_size specified - set large batch (e.g., 2048+) to max GPU usage")
        
        # Check for mixed precision
        if has_pytorch and 'train' in filename_lower:
            if 'amp' not in content_lower and 'autocast' not in content_lower:
                resource_issues.append("Consider torch.cuda.amp for 2x faster training with mixed precision")
        
        if has_tensorflow and 'train' in filename_lower:
            if 'mixed_float16' not in content:
                resource_issues.append("Consider mixed_float16 policy for faster TF training")
        
        # If resource issues found, return warning
        if resource_issues:
            warning = ["⚡ RESOURCE OPTIMIZATION CHECK:"]
            for issue in resource_issues:
                warning.append(f"   • {issue}")
            warning.append("")
            warning.append("💡 CRITICAL: You MUST max out all resources (CPU cores, GPU RAM, batch sizes)")
            warning.append("   Scripts should print: 'Using X CPU cores, batch_size=Y, GPU RAM=Z GB'")
            return "\n".join(warning)
        
        # If GPU is configured and this is a training/prediction script, remind to consult Oracle
        if is_ml_script and (has_pytorch or has_xgboost or has_lightgbm or has_tensorflow or has_catboost or has_cuml):
            return (
                "💡 STRATEGIC HINT: Instead of running iterative baselines, pause and open an extended brainstorming session with Oracle.\n"
                "   Discuss: (a) fastest GPU-first pipeline, (b) risk of data leakage, (c) minimal full-dataset runs.\n"
                "   Aim to produce ONE high-quality training script that can reach medal range in the first attempt."
            )
        
        return ""
