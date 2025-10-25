"""General Prompt: Guides notebook structure generation"""

import nbformat
from typing import Dict, List, Optional
from pathlib import Path


class GeneralPrompt:
    """Guides the LLM to generate well-structured ML competition solutions"""
    
    def __init__(self, competition_type: str, data_description: Dict, evaluation_metric: str):
        self.competition_type = competition_type
        self.data_description = data_description
        self.evaluation_metric = evaluation_metric
    
    async def generate(self, specialized_context: str, iteration: int) -> Dict:
        """Generate notebook with guidance for the LLM to create code"""
        
        notebook = nbformat.v4.new_notebook()
        
        # Metadata
        notebook.cells.append(nbformat.v4.new_markdown_cell(
            f"# Competition Solution - Iteration {iteration}\n"
            f"**Type**: {self.competition_type}\n"
            f"**Metric**: {self.evaluation_metric}"
        ))
        
        # Main guidance cell - this is where the LLM generates the solution
        guidance = f"""# TASK: Generate Complete ML Competition Solution

## Competition Context
- Type: {self.competition_type}
- Evaluation Metric: {self.evaluation_metric}
- Iteration: {iteration}

## Requirements
Create a complete, working solution that:
1. Loads and explores the data
2. Preprocesses and engineers features appropriately
3. Trains a model suitable for the task
4. Validates performance properly
5. Generates predictions for submission
6. Outputs VALIDATION_SCORE for tracking

## Structure Your Solution With These Sections:

### 1. IMPORTS
Import necessary libraries based on the task. Consider what you'll need for:
- Data manipulation (pandas, numpy)
- Machine learning (sklearn, xgboost, lightgbm, pytorch, tensorflow as needed)
- Visualization (if exploring data)
- Task-specific libraries based on competition type

### 2. CONFIGURATION
Set up:
- File paths for data
- Random seeds for reproducibility
- Model parameters
- Any competition-specific settings

### 3. DATA LOADING
- Load training and test data
- Handle different file formats appropriately
- Check data shapes and basic statistics

### 4. EXPLORATORY DATA ANALYSIS
- Understand the data structure
- Check for missing values
- Analyze target distribution
- Identify feature types

### 5. DATA PREPROCESSING
- Handle missing values appropriately
- Encode categorical variables if present
- Scale/normalize if necessary
- Create train/validation split

### 6. FEATURE ENGINEERING
- Create relevant features for the task
- Apply domain-specific transformations
- Consider interactions if beneficial

### 7. MODEL TRAINING
- Choose appropriate algorithm(s)
- Train model(s) with proper parameters
- Use validation strategy suitable for the data

### 8. VALIDATION & EVALUATION
- Evaluate on validation set
- Use the specified metric: {self.evaluation_metric}
- Print validation score as: print(f"VALIDATION_SCORE: {{score:.4f}}")

### 9. PREDICTION & SUBMISSION
- Generate predictions for test set
- Create submission file in required format
- Save to submission directory

## Specialized Guidance for {self.competition_type}:
{specialized_context}

## Important Notes:
- Ensure all code is executable and error-free
- Handle edge cases gracefully
- Use appropriate validation strategy for the competition
- The validation score must be printed in the exact format shown
- Save the final submission.csv to the submission directory

Now, generate the complete solution code following this structure and guidance."""
        
        # Add the guidance as a code cell where the LLM will generate the solution
        notebook.cells.append(nbformat.v4.new_code_cell(guidance))
        
        # Return as dictionary
        return {
            "cells": notebook.cells,
            "metadata": notebook.metadata
        }
    
    def _get_iteration_hints(self, iteration: int) -> str:
        """Provide iteration-specific guidance"""
        if iteration == 1:
            return "Focus on creating a simple, working baseline"
        elif iteration <= 3:
            return "Build upon the baseline with feature engineering and better models"
        elif iteration <= 5:
            return "Optimize hyperparameters and try ensemble methods"
        else:
            return "Fine-tune the best approaches and ensure robustness"