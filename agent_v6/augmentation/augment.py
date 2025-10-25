"""Data Augmentation: Guides data modification strategies"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import json
import nbformat
import asyncio

from agent_v6.core.agent import Agent
from agent_v6.core.tools import ToolRegistry, BashTool, ReadTool, WriteTool


class DataAugmenter:
    """Guides the generation of data augmentation strategies"""
    
    def __init__(self, data_dir: Path, augmented_dir: Path, competition_type: str):
        self.data_dir = Path(data_dir)
        self.augmented_dir = Path(augmented_dir)
        self.competition_type = competition_type
        self.augmented_dir.mkdir(parents=True, exist_ok=True)
        
    async def augment(self, iteration: int) -> Dict[Path, Path]:
        """Generate and execute data augmentation"""
        print(f"🔧 Augmenting data for iteration {iteration}...")
        
        # Create augmentation notebook
        notebook = await self._generate_augmentation_notebook(iteration)
        
        # Save notebook
        notebook_path = self.augmented_dir / f"augment_v{iteration}.ipynb"
        with open(notebook_path, 'w') as f:
            nbformat.write(notebook, f)
        
        # For now, copy original data as fallback
        augmented_paths = await self._copy_original_data()
        
        print(f"✅ Data augmentation complete")
        return augmented_paths
    
    async def _generate_augmentation_notebook(self, iteration: int) -> nbformat.NotebookNode:
        """Generate notebook with augmentation guidance"""
        
        notebook = nbformat.v4.new_notebook()
        
        # Add metadata
        notebook.cells.append(nbformat.v4.new_markdown_cell(
            f"# Data Augmentation - Iteration {iteration}\n"
            f"Type: {self.competition_type}"
        ))
        
        # Add augmentation guidance
        augmentation_guidance = f"""# Generate Data Augmentation Strategy

## Task
Create appropriate data augmentation for {self.competition_type} competition.

## Data Location
- Original data: {self.data_dir}
- Save augmented data to: {self.augmented_dir}

## Augmentation Principles

### General Guidelines:
1. **Understand the data first** - Load and analyze before augmenting
2. **Preserve data validity** - Don't create impossible/invalid samples
3. **Maintain label correctness** - Augmentation shouldn't change labels
4. **Consider computational cost** - Balance benefit vs processing time
5. **Version control** - Save augmented data separately

### Iteration-Specific Strategy:
- Iteration {iteration}: {"Conservative augmentation" if iteration <= 2 else "Moderate augmentation" if iteration <= 4 else "Aggressive augmentation"}

### Type-Specific Considerations for {self.competition_type}:
{self._get_type_specific_guidance()}

## Implementation Requirements:
1. Load original training data
2. Apply appropriate augmentation techniques
3. Validate augmented data quality
4. Save augmented data with clear naming
5. Create mapping of original to augmented files
6. Log augmentation statistics

## Code Generation Task:
Generate code that:
- Identifies which data files to augment (typically training data only)
- Applies suitable augmentation based on data type and competition
- Ensures test data remains unchanged
- Handles errors gracefully
- Reports what augmentation was applied

Remember: Not all competitions benefit from augmentation. Use judgment based on:
- Data size (small datasets benefit more)
- Task type (some tasks are augmentation-friendly)
- Validation performance (check if augmentation helps)
"""
        
        notebook.cells.append(nbformat.v4.new_code_cell(augmentation_guidance))
        
        return notebook
    
    def _get_type_specific_guidance(self) -> str:
        """Get competition-type specific augmentation guidance"""
        
        guidance_map = {
            "image-classification": """
- Geometric: rotations, flips, crops (preserve object identity)
- Color: brightness, contrast, saturation (maintain recognizability)
- Advanced: mixup, cutout, random erasing (if dataset is small)
- Avoid: excessive distortion that changes semantic meaning""",
            
            "text-classification": """
- Synonym replacement (preserve meaning)
- Back-translation (if multilingual models available)
- Paraphrasing (maintain sentiment/class)
- Avoid: changes that alter the classification label""",
            
            "tabular-regression": """
- Noise injection (small gaussian noise to continuous features)
- SMOTE for imbalanced regression
- Feature perturbation (within reasonable ranges)
- Avoid: creating out-of-distribution samples""",
            
            "tabular-classification": """
- SMOTE/ADASYN for class imbalance
- Feature noise (maintain class boundaries)
- Synthetic minority oversampling
- Avoid: overlapping class boundaries""",
            
            "time-series-forecasting": """
- Window sliding (create more training sequences)
- Time warping (slight speed changes)
- Magnitude warping (scale variations)
- Avoid: breaking temporal dependencies""",
            
            "object-detection": """
- Spatial augmentations with bbox adjustment
- Copy-paste augmentation
- Mosaic augmentation
- Ensure bounding boxes remain valid""",
            
            "general-else": """
- Analyze data type and structure first
- Apply conservative augmentation
- Focus on addressing data imbalances
- Test augmentation impact on validation"""
        }
        
        return guidance_map.get(self.competition_type, guidance_map["general-else"])
    
    async def _copy_original_data(self) -> Dict[Path, Path]:
        """Fallback: Copy original data without augmentation"""
        augmented_paths = {}
        
        # Copy all CSV files
        for file_path in self.data_dir.glob("*.csv"):
            dest_path = self.augmented_dir / file_path.name
            shutil.copy2(file_path, dest_path)
            augmented_paths[file_path] = dest_path
        
        # Copy other data files
        for pattern in ["*.json", "*.txt", "*.parquet"]:
            for file_path in self.data_dir.glob(pattern):
                dest_path = self.augmented_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                augmented_paths[file_path] = dest_path
        
        return augmented_paths