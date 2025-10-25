"""Tabular Data Specialized Prompt"""

from typing import Dict


class SpecializedPrompt:
    """Guidance for tabular data competitions (regression and classification)"""
    
    def __init__(self, data_description: Dict):
        self.data_description = data_description
        
    def get_context(self) -> str:
        """Provide guidance for tabular data approach"""
        
        return """# Tabular Data Guidance

APPROACH:
- Understand feature types: numerical, categorical, datetime, text
- Identify target variable type: regression vs classification
- Check for data quality issues
- Determine if time-based splitting is needed

KEY CONSIDERATIONS:
1. Exploratory Data Analysis:
   - Feature distributions and outliers
   - Missing value patterns (random, systematic)
   - Correlations and multicollinearity
   - Target variable distribution (imbalanced? skewed?)

2. Feature Engineering:
   - Numerical: Scaling, transformations, binning, polynomial features
   - Categorical: Encoding strategies (label, one-hot, target, embedding)
   - Datetime: Extract components (year, month, day, hour, cyclical features)
   - Interactions: Create meaningful feature combinations
   - Domain-specific: Use domain knowledge for custom features

3. Missing Value Strategy:
   - Understand missingness mechanism (MAR, MCAR, MNAR)
   - Simple: Mean/median/mode, forward fill
   - Advanced: KNN imputation, iterative imputation, model-based
   - Consider: Creating missingness indicators

4. Model Selection:
   - Linear models for interpretability and baseline
   - Tree-based: Random Forest, XGBoost, LightGBM, CatBoost
   - Consider dataset size and feature count
   - Ensemble different model types for best performance

5. Validation Strategy:
   - Random split vs stratified vs time-based
   - K-fold cross-validation for robust estimates
   - Hold-out set for final evaluation
   - Consider: Nested CV for hyperparameter tuning

6. Hyperparameter Optimization:
   - Start with default parameters
   - Use Bayesian optimization or random search
   - Focus on important parameters first
   - Monitor for overfitting

COMMON PITFALLS TO AVOID:
- Data leakage in feature engineering
- Improper handling of categorical variables
- Not checking for multicollinearity
- Ignoring business constraints
- Overfitting on small datasets
- Using test set for any decisions

Generate your solution considering these guidelines and the specific problem requirements."""
