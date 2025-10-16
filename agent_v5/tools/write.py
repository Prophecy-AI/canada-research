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
        
        # Check for BANNED internet access patterns (CRITICAL - check first)
        internet_violations = []
        
        # Check for from_pretrained without offline mode
        if '.from_pretrained(' in content:
            # Check if TRANSFORMERS_OFFLINE is set or local path is used
            has_offline = 'TRANSFORMERS_OFFLINE' in content or "os.environ['TRANSFORMERS_OFFLINE']" in content
            has_local_path = '/opt/' in content or '/home/' in content or '/models/' in content
            
            if not (has_offline or has_local_path):
                internet_violations.append("from_pretrained() without TRANSFORMERS_OFFLINE=1 or local path")
        
        # Check for other network calls
        network_patterns = [
            ('requests.get', 'HTTP GET request'),
            ('requests.post', 'HTTP POST request'),
            ('urllib.request', 'urllib network call'),
            ('urllib.urlopen', 'urllib network call'),
            ('wget ', 'wget download'),
            ('curl ', 'curl download'),
            ('.download(', 'download call'),
        ]
        
        for pattern, description in network_patterns:
            if pattern in content:
                internet_violations.append(f"{description} ({pattern})")
        
        if internet_violations:
            warnings = ["CRITICAL: INTERNET ACCESS VIOLATION DETECTED"]
            for violation in internet_violations:
                warnings.append(f"  - {violation}")
            warnings.append("")
            warnings.append("NO INTERNET ACCESS in competition environment!")
            warnings.append("Consult Oracle if you need external models/data")
            warnings.append("")
            warnings.append("File written but you MUST fix this before running!")
            return "\n".join(warnings)
        
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

            warnings = ["WARNING: CPU-ONLY SKLEARN IMPORTS DETECTED"]
            for forbidden in found_forbidden:
                replacement = replacement_map.get(forbidden, 'cuML equivalent')
                warnings.append(f"  - {forbidden} -> USE {replacement} INSTEAD")

            warnings.append("")
            warnings.append("This code will run 10-100x SLOWER on CPU than GPU alternatives.")
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

            return "WARNING: GPU not detected in script\n" + "\n".join(f"  - {hint}" for hint in hints) + "\n  - CPU training is 10-100x slower - verify GPU usage before running"
        
        # Check for resource maximization issues
        resource_issues = []
        
        # Check for n_jobs
        if (has_cuml or 'sklearn' in content_lower) and 'n_jobs' not in content_lower:
            resource_issues.append("Missing n_jobs=-1 (not using all CPU cores)")
        elif 'n_jobs=1' in content_lower or 'n_jobs = 1' in content_lower:
            resource_issues.append("CRITICAL: n_jobs=1 found - change to n_jobs=-1 to use all cores")
        
        # Check for SMALL batch sizes (CRITICAL)
        import re
        if 'batch_size' in content_lower or 'BATCH_SIZE' in content:
            batch_matches = re.findall(r'[Bb][Aa][Tt][Cc][Hh]_?[Ss][Ii][Zz][Ee]\s*=\s*(\d+)', content)
            for batch_str in batch_matches:
                batch_val = int(batch_str)
                # Determine minimum based on model type
                min_batch = 256  # Default for transformers
                model_type = "transformers"
                
                if 'conv' in content_lower or 'cnn' in content_lower or 'resnet' in content_lower:
                    min_batch = 512
                    model_type = "CNNs"
                elif 'tabular' in content_lower or 'xgboost' in content_lower or 'lightgbm' in content_lower:
                    min_batch = 2048
                    model_type = "tabular models"
                
                if batch_val < min_batch:
                    resource_issues.append(f"CRITICAL: batch_size={batch_val} is TOO SMALL - minimum {min_batch} for {model_type}")
        
        # Check for HARDCODED num_workers (CRITICAL)
        if 'num_workers' in content_lower:
            num_worker_matches = re.findall(r'num_workers\s*=\s*(\d+)', content)
            for nw_str in num_worker_matches:
                nw_val = int(nw_str)
                if nw_val < 8:  # Assume most machines have 8+ cores
                    resource_issues.append(f"CRITICAL: num_workers={nw_val} is HARDCODED - use os.cpu_count() to max out cores")
        
        # Check for PyTorch DataLoader optimization
        if has_pytorch and 'DataLoader' in content:
            if 'num_workers' not in content:
                resource_issues.append("CRITICAL: PyTorch DataLoader missing num_workers (should use os.cpu_count())")
            if 'pin_memory' not in content:
                resource_issues.append("PyTorch DataLoader missing pin_memory=True (faster GPU transfer)")
        
        # Check for missing RESOURCE CONFIG PRINTING (MANDATORY)
        if 'train' in filename_lower or 'predict' in filename_lower:
            has_resource_print = any(phrase in content for phrase in [
                'RESOURCES:', 'CPU cores', 'Resource configuration', 'Resource config',
                'NUM_WORKERS', 'num_workers=', 'batch_size='
            ])
            
            # More strict check - must have explicit print
            has_explicit_print = 'print(' in content and ('CPU' in content or 'cores' in content or 'batch' in content)
            
            if not has_explicit_print:
                resource_issues.append("CRITICAL: Missing resource config print - must print: 'RESOURCES: X cores, batch_size=Y, GPU=Z'")
        
        # Check for mixed precision
        if has_pytorch and 'train' in filename_lower:
            if 'amp' not in content_lower and 'autocast' not in content_lower:
                resource_issues.append("Missing torch.cuda.amp for 2x faster training with mixed precision")
        
        if has_tensorflow and 'train' in filename_lower:
            if 'mixed_float16' not in content:
                resource_issues.append("Missing mixed_float16 policy for faster TF training")
        
        # If resource issues found, return warning
        if resource_issues:
            warning = ["RESOURCE UTILIZATION VIOLATIONS DETECTED"]
            for issue in resource_issues:
                warning.append(f"  - {issue}")
            warning.append("")
            warning.append("This code will waste 80%+ of compute capacity")
            warning.append("Training will be 5-10x slower than necessary")
            warning.append("You MUST rewrite before running - consult Oracle if needed")
            warning.append("")
            warning.append("Required template:")
            warning.append("  import os")
            warning.append("  NUM_WORKERS = os.cpu_count()")
            warning.append("  BATCH_SIZE = 1024  # or larger")
            warning.append("  print(f'RESOURCES: {NUM_WORKERS} cores, batch_size={BATCH_SIZE}, GPU={torch.cuda.get_device_name(0)}')")
            return "\n".join(warning)
        
        # If GPU is configured and this is a training/prediction script, remind to consult Oracle
        if is_ml_script and (has_pytorch or has_xgboost or has_lightgbm or has_tensorflow or has_catboost or has_cuml):
            return (
                "REMINDER: Before running training, consult Oracle for strategy validation.\n"
                "Discuss: GPU pipeline optimization, data leakage risks, and approach validation.\n"
                "Goal: Produce high-quality training script on first attempt."
            )
        
        return ""
