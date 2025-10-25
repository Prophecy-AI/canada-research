"""
Explorer: Analyzes competition data and metadata to classify and understand the problem
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio
import re

from agent_v6.core.agent import Agent
from agent_v6.core.tools import ToolRegistry, BashTool, ReadTool, WriteTool


class Explorer:
    """
    Explores and analyzes:
    - Competition description and instructions
    - Data structure and format
    - Input/output relationships
    - Evaluation metric (log loss, AUC, RMSE, etc.)
    - Classifies into one of 16 competition types
    """
    
    def __init__(self, data_dir: Path, workspace_dir: Path, instructions_path: Path):
        self.data_dir = Path(data_dir)
        self.workspace_dir = Path(workspace_dir)
        self.instructions_path = Path(instructions_path)
        
        # Results storage
        self.exploration_results = {
            "competition_type": None,
            "evaluation_metric": None,
            "data_description": {},
            "file_structure": {},
            "data_stats": {},
            "task_description": ""
        }
    
    async def explore(self) -> Dict:
        """Main exploration flow"""
        print("🔍 Starting exploration...")
        
        # 1. Read competition instructions
        await self._analyze_instructions()
        
        # 2. Analyze file structure
        await self._analyze_file_structure()
        
        # 3. Analyze data content
        await self._analyze_data_content()
        
        # 4. Determine competition type
        await self._classify_competition()
        
        # 5. Generate exploration notebook
        await self._generate_exploration_notebook()
        
        print(f"✅ Exploration complete")
        return self.exploration_results
    
    async def _analyze_instructions(self):
        """Read and analyze competition instructions"""
        if self.instructions_path.exists():
            instructions = self.instructions_path.read_text()
            self.exploration_results["task_description"] = instructions
            
            # Extract evaluation metric from instructions
            metric_patterns = {
                r"log.?loss": "log_loss",
                r"auc|area.?under": "auc",
                r"rmse|root.?mean": "rmse",
                r"mae|mean.?absolute": "mae",
                r"accuracy": "accuracy",
                r"f1.?score": "f1",
                r"precision": "precision",
                r"recall": "recall",
                r"r2|r.?squared": "r2",
                r"mape": "mape",
                r"iou|intersection": "iou",
                r"dice": "dice",
                r"bleu": "bleu"
            }
            
            instructions_lower = instructions.lower()
            for pattern, metric in metric_patterns.items():
                if re.search(pattern, instructions_lower):
                    self.exploration_results["evaluation_metric"] = metric
                    break
            
            # Extract task type hints
            self.exploration_results["task_hints"] = self._extract_task_hints(instructions)
    
    def _extract_task_hints(self, text: str) -> List[str]:
        """Extract hints about task type from text"""
        hints = []
        text_lower = text.lower()
        
        task_keywords = {
            "classification": ["classify", "classification", "predict class", "categorize"],
            "regression": ["regression", "predict value", "forecast", "estimate"],
            "clustering": ["cluster", "segment", "group"],
            "detection": ["detect", "locate", "find objects", "bounding box"],
            "segmentation": ["segment", "pixel-wise", "mask"],
            "generation": ["generate", "create", "produce"],
            "ranking": ["rank", "order", "sort", "recommend"],
            "time-series": ["time series", "forecast", "temporal", "sequence"]
        }
        
        for task_type, keywords in task_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                hints.append(task_type)
        
        return hints
    
    async def _analyze_file_structure(self):
        """Analyze the structure of data files"""
        file_info = {}
        
        for file_path in self.data_dir.iterdir():
            if file_path.is_file():
                file_info[file_path.name] = {
                    "size": file_path.stat().st_size,
                    "extension": file_path.suffix
                }
                
                # Check for specific file types
                if file_path.suffix in ['.csv', '.tsv']:
                    file_info[file_path.name]["type"] = "tabular"
                elif file_path.suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']:
                    file_info[file_path.name]["type"] = "image"
                elif file_path.suffix in ['.txt', '.json', '.xml']:
                    file_info[file_path.name]["type"] = "text"
                elif file_path.suffix in ['.mp3', '.wav', '.flac']:
                    file_info[file_path.name]["type"] = "audio"
                elif file_path.suffix in ['.mp4', '.avi', '.mov']:
                    file_info[file_path.name]["type"] = "video"
                elif file_path.suffix in ['.npy', '.npz', '.h5', '.hdf5']:
                    file_info[file_path.name]["type"] = "array"
        
        self.exploration_results["file_structure"] = file_info
    
    async def _analyze_data_content(self):
        """Analyze the content of data files"""
        workspace = self.workspace_dir / "exploration"
        workspace.mkdir(exist_ok=True)
        
        tools = ToolRegistry(str(workspace))
        tools.register(BashTool(str(workspace)))
        tools.register(ReadTool(str(workspace)))
        tools.register(WriteTool(str(workspace)))
        
        # Create exploration prompt
        exploration_prompt = f"""You are a data scientist exploring a Kaggle competition dataset.

Data directory: {self.data_dir}
Files found: {list(self.exploration_results['file_structure'].keys())}

Your task:
1. Load and examine the data files
2. Understand the structure and content
3. Identify key features and target variables
4. Calculate basic statistics
5. Determine data types (tabular, image paths, text, etc.)

Write a Python script 'explore_data.py' that:
- Loads the main data files (train, test, sample_submission if they exist)
- Prints data shapes and types
- Shows first few rows/samples
- Calculates statistics
- Outputs findings in JSON format

End your script with:
```python
findings = {{
    "data_type": "tabular|image|text|mixed",
    "num_train_samples": <number>,
    "num_test_samples": <number>,
    "num_features": <number>,
    "target_type": "classification|regression|other",
    "num_classes": <number if classification>,
    "feature_types": {{"numeric": [], "categorical": [], "text": [], "other": []}},
    "has_missing_values": true/false,
    "data_balance": "balanced|imbalanced|unknown"
}}

import json
print("DATA_FINDINGS_JSON:")
print(json.dumps(findings, indent=2))
```"""
        
        agent = Agent(str(workspace), exploration_prompt, tools)
        output = await agent.run("Explore the competition data")
        
        # Parse findings
        if "DATA_FINDINGS_JSON:" in output:
            json_str = output.split("DATA_FINDINGS_JSON:")[1].strip()
            try:
                findings = json.loads(json_str)
                self.exploration_results["data_stats"] = findings
            except json.JSONDecodeError:
                print("⚠️ Could not parse data findings")
    
    async def _classify_competition(self):
        """Classify competition into one of 16 types based on exploration"""
        data_stats = self.exploration_results.get("data_stats", {})
        file_structure = self.exploration_results.get("file_structure", {})
        task_hints = self.exploration_results.get("task_hints", [])
        
        # Decision tree for classification
        data_type = data_stats.get("data_type", "unknown")
        target_type = data_stats.get("target_type", "unknown")
        
        # Check for specific file patterns
        has_images = any(f.get("type") == "image" for f in file_structure.values())
        has_audio = any(f.get("type") == "audio" for f in file_structure.values())
        has_video = any(f.get("type") == "video" for f in file_structure.values())
        has_text_files = any(f.get("type") == "text" for f in file_structure.values())
        
        # Classification logic
        if has_video:
            competition_type = "video-processing"
        elif has_audio:
            competition_type = "audio-processing"
        elif "detection" in task_hints and (has_images or data_type == "image"):
            competition_type = "object-detection"
        elif "segmentation" in task_hints and (has_images or data_type == "image"):
            competition_type = "image-segmentation"
        elif data_type == "image" and target_type == "classification":
            competition_type = "image-classification"
        elif data_type == "image" and "caption" in str(task_hints):
            competition_type = "image-to-text"
        elif data_type == "text" and target_type == "classification":
            competition_type = "text-classification"
        elif data_type == "text" and "generation" in task_hints:
            competition_type = "text-generation"
        elif data_type == "tabular":
            if "time" in str(task_hints) or "forecast" in str(task_hints):
                competition_type = "time-series-forecasting"
            elif target_type == "regression":
                competition_type = "tabular-regression"
            elif target_type == "classification":
                competition_type = "tabular-classification"
            elif "clustering" in task_hints:
                competition_type = "clustering"
            elif "anomaly" in str(task_hints):
                competition_type = "anomaly-detection"
            else:
                competition_type = "tabular-classification"  # Default for tabular
        elif "recommend" in str(task_hints) or "ranking" in task_hints:
            competition_type = "recommender-systems"
        elif "reinforce" in str(task_hints) or "game" in str(task_hints):
            competition_type = "reinforcement-learning"
        elif "graph" in str(task_hints) or "network" in str(task_hints):
            competition_type = "graph-ml"
        else:
            competition_type = "general-else"
        
        self.exploration_results["competition_type"] = competition_type
        
        # Set default metric if not found
        if not self.exploration_results["evaluation_metric"]:
            default_metrics = {
                "image-classification": "accuracy",
                "text-classification": "accuracy",
                "tabular-classification": "accuracy",
                "tabular-regression": "rmse",
                "time-series-forecasting": "rmse",
                "object-detection": "map",
                "image-segmentation": "iou",
                "clustering": "silhouette",
                "anomaly-detection": "auc"
            }
            self.exploration_results["evaluation_metric"] = default_metrics.get(
                competition_type, "accuracy"
            )
    
    async def _generate_exploration_notebook(self):
        """Generate a Jupyter notebook with exploration results"""
        import nbformat
        
        notebook = nbformat.v4.new_notebook()
        
        # Title cell
        notebook.cells.append(nbformat.v4.new_markdown_cell(
            f"# Competition Exploration: {self.exploration_results['competition_type']}\n\n"
            f"**Evaluation Metric**: {self.exploration_results['evaluation_metric']}"
        ))
        
        # Data loading cell
        notebook.cells.append(nbformat.v4.new_code_cell(f"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

data_dir = Path('{self.data_dir}')

# Load data files
files = list(data_dir.glob('*'))
print(f"Found {{len(files)}} files:")
for f in files:
    print(f"  - {{f.name}} ({{f.stat().st_size / 1024:.1f}} KB)")"""))
        
        # Data description cell
        if self.exploration_results["data_stats"]:
            stats_str = json.dumps(self.exploration_results["data_stats"], indent=2)
            notebook.cells.append(nbformat.v4.new_markdown_cell("## Data Statistics"))
            notebook.cells.append(nbformat.v4.new_code_cell(f"data_stats = {stats_str}\nprint(data_stats)"))
        
        # Save notebook
        notebook_path = self.workspace_dir / "exploration.ipynb"
        nbformat.write(notebook, str(notebook_path))
        
        print(f"   📓 Exploration notebook saved to {notebook_path}")
    
    def get_data_description(self) -> Dict:
        """Get condensed data description for prompts"""
        return {
            "competition_type": self.exploration_results["competition_type"],
            "evaluation_metric": self.exploration_results["evaluation_metric"],
            "data_type": self.exploration_results["data_stats"].get("data_type", "unknown"),
            "num_features": self.exploration_results["data_stats"].get("num_features", 0),
            "num_train_samples": self.exploration_results["data_stats"].get("num_train_samples", 0),
            "target_type": self.exploration_results["data_stats"].get("target_type", "unknown"),
            "num_classes": self.exploration_results["data_stats"].get("num_classes", 0)
        }
