"""Solution Improvement: Guides iterative refinement strategies"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import nbformat

from agent_v6.core.agent import Agent
from agent_v6.core.tools import ToolRegistry, WriteTool, ReadTool, BashTool


class SolutionImprover:
    """Guides solution improvement through iterative refinement"""
    
    def __init__(self, workspace_dir: Path, competition_type: str, evaluation_metric: str):
        self.workspace_dir = Path(workspace_dir)
        self.competition_type = competition_type
        self.evaluation_metric = evaluation_metric
        
    async def improve(self, notebook_path: Path, iteration: int, previous_scores: List[float] = None) -> Path:
        """Generate improvements for the solution"""
        
        print(f"✨ Improving solution for iteration {iteration}...")
        
        # Load current notebook
        with open(notebook_path, 'r') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Generate improvement strategy
        improved_notebook = await self._generate_improvements(notebook, iteration, previous_scores)
        
        # Save improved notebook
        improved_path = self.workspace_dir / f"improved_v{iteration}.ipynb"
        with open(improved_path, 'w') as f:
            nbformat.write(improved_notebook, f)
        
        print(f"✅ Solution improved")
        return improved_path
    
    async def _generate_improvements(self, notebook: nbformat.NotebookNode, iteration: int, 
                                    previous_scores: List[float] = None) -> nbformat.NotebookNode:
        """Generate improvement guidance and modify notebook"""
        
        # Create new notebook with improvements
        improved = nbformat.v4.new_notebook()
        
        # Add metadata
        improved.cells.append(nbformat.v4.new_markdown_cell(
            f"# Improved Solution - Iteration {iteration}\n"
            f"Type: {self.competition_type}\n"
            f"Metric: {self.evaluation_metric}"
        ))
        
        # Analyze current solution
        analysis = self._analyze_current_solution(notebook)
        
        # Generate improvement guidance
        improvement_guidance = f"""# Task: Improve Current Solution

## Current Solution Analysis
{json.dumps(analysis, indent=2)}

## Previous Performance
{"No previous scores" if not previous_scores else f"Scores: {previous_scores}"}
{"" if not previous_scores or len(previous_scores) < 2 else f"Trend: {'Improving' if previous_scores[-1] > previous_scores[-2] else 'Declining'}"}

## Iteration {iteration} Improvement Strategy

### Based on iteration number:
{self._get_iteration_strategy(iteration)}

### Based on competition type ({self.competition_type}):
{self._get_type_specific_improvements()}

### Based on current solution analysis:
{self._get_analysis_based_improvements(analysis)}

## Improvement Implementation Guidelines:

1. **Feature Engineering**
   - Have we explored all meaningful feature interactions?
   - Can we create domain-specific features?
   - Should we try different encoding strategies?

2. **Model Selection**
   - Is the current model appropriate for data size/type?
   - Should we try different algorithms?
   - Would an ensemble improve performance?

3. **Hyperparameter Optimization**
   - Are we using default parameters?
   - Have we tried systematic tuning (Grid/Random/Bayesian)?
   - Are there key parameters we haven't explored?

4. **Validation Strategy**
   - Is our validation strategy appropriate?
   - Should we use different CV folds?
   - Are we detecting overfitting properly?

5. **Data Handling**
   - Are we handling missing values optimally?
   - Should we try different preprocessing?
   - Can we better handle outliers?

## Task: Generate Improved Solution

Take the current solution and improve it by:
1. Addressing the weaknesses identified above
2. Implementing the suggested improvements
3. Maintaining what's working well
4. Adding new techniques appropriate for iteration {iteration}

Ensure the improved solution:
- Builds upon current strengths
- Addresses identified weaknesses
- Remains executable and robust
- Properly validates improvements
- Outputs VALIDATION_SCORE for tracking

Generate the complete improved solution below:
"""
        
        improved.cells.append(nbformat.v4.new_code_cell(improvement_guidance))
        
        # Add original solution for reference
        improved.cells.append(nbformat.v4.new_markdown_cell("## Original Solution (for reference):"))
        improved.cells.extend(notebook.cells)
        
        return improved
    
    def _analyze_current_solution(self, notebook: nbformat.NotebookNode) -> Dict:
        """Analyze current solution to identify improvement opportunities"""
        
        analysis = {
            "has_feature_engineering": False,
            "has_hyperparameter_tuning": False,
            "has_ensemble": False,
            "has_cross_validation": False,
            "model_types": [],
            "preprocessing_steps": [],
            "validation_strategy": "unknown"
        }
        
        # Analyze code cells
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                code = cell.source.lower()
                
                # Check for feature engineering
                if any(term in code for term in ['feature', 'polynomial', 'interaction', 'transform']):
                    analysis["has_feature_engineering"] = True
                
                # Check for hyperparameter tuning
                if any(term in code for term in ['gridsearch', 'randomsearch', 'optuna', 'hyperopt']):
                    analysis["has_hyperparameter_tuning"] = True
                
                # Check for ensemble
                if any(term in code for term in ['voting', 'stacking', 'ensemble', 'blend']):
                    analysis["has_ensemble"] = True
                
                # Check for cross-validation
                if any(term in code for term in ['cross_val', 'kfold', 'stratifiedkfold']):
                    analysis["has_cross_validation"] = True
                
                # Identify models
                for model in ['randomforest', 'xgboost', 'lightgbm', 'catboost', 'neural', 'svm', 'logistic']:
                    if model in code:
                        if model not in analysis["model_types"]:
                            analysis["model_types"].append(model)
        
        return analysis
    
    def _get_iteration_strategy(self, iteration: int) -> str:
        """Get iteration-specific improvement strategy"""
        
        if iteration == 1:
            return "Focus on getting a solid baseline with good preprocessing"
        elif iteration == 2:
            return "Add basic feature engineering and try different models"
        elif iteration == 3:
            return "Implement cross-validation and initial hyperparameter tuning"
        elif iteration == 4:
            return "Try ensemble methods and advanced features"
        elif iteration == 5:
            return "Fine-tune best approaches and optimize hyperparameters"
        else:
            return "Polish solution, ensure robustness, try innovative approaches"
    
    def _get_type_specific_improvements(self) -> str:
        """Get competition-type specific improvements"""
        
        improvements_map = {
            "image-classification": "Try different architectures, augmentation levels, learning rates, and TTA",
            "text-classification": "Experiment with different embeddings, sequence lengths, and model architectures",
            "tabular-regression": "Focus on feature engineering, try different scalers, and ensemble tree models",
            "tabular-classification": "Handle class imbalance, try different encodings, and calibrate probabilities",
            "time-series-forecasting": "Add lag features, try different window sizes, and ensemble statistical with ML",
            "general-else": "Try diverse approaches, focus on validation, and ensemble different model types"
        }
        
        return improvements_map.get(self.competition_type, improvements_map["general-else"])
    
    def _get_analysis_based_improvements(self, analysis: Dict) -> str:
        """Get improvements based on solution analysis"""
        
        suggestions = []
        
        if not analysis["has_feature_engineering"]:
            suggestions.append("- Add feature engineering (interactions, polynomials, domain-specific)")
        
        if not analysis["has_hyperparameter_tuning"]:
            suggestions.append("- Implement hyperparameter optimization")
        
        if not analysis["has_ensemble"]:
            suggestions.append("- Try ensemble methods (voting, stacking)")
        
        if not analysis["has_cross_validation"]:
            suggestions.append("- Use cross-validation for robust evaluation")
        
        if len(analysis["model_types"]) < 2:
            suggestions.append("- Try different model types for diversity")
        
        if not suggestions:
            suggestions.append("- Focus on fine-tuning and optimization")
            suggestions.append("- Try advanced techniques specific to the domain")
        
        return "\n".join(suggestions)