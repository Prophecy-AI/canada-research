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
        
        # Check if content contains ML frameworks
        content_lower = content.lower()
        has_pytorch = 'import torch' in content_lower or 'from torch' in content_lower
        has_xgboost = 'import xgboost' in content_lower or 'from xgboost' in content_lower
        has_lightgbm = 'import lightgbm' in content_lower or 'from lightgbm' in content_lower
        has_tensorflow = 'import tensorflow' in content_lower or 'from tensorflow' in content_lower
        has_catboost = 'import catboost' in content_lower or 'from catboost' in content_lower
        
        if not (has_pytorch or has_xgboost or has_lightgbm or has_tensorflow or has_catboost):
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
        
        # If GPU is configured and this is a training/prediction script, remind to consult Oracle
        if is_ml_script and (has_pytorch or has_xgboost or has_lightgbm or has_tensorflow or has_catboost):
            return (
                "💡 STRATEGIC HINT: Instead of running iterative baselines, pause and open an extended brainstorming session with Oracle.\n"
                "   Discuss: (a) fastest GPU-first pipeline, (b) risk of data leakage, (c) minimal full-dataset runs.\n"
                "   Aim to produce ONE high-quality training script that can reach medal range in the first attempt."
            )
        
        return ""
