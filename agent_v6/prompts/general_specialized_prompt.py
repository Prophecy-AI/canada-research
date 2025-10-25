"""General/Fallback Specialized Prompt"""

from typing import Dict


class SpecializedPrompt:
    """Guidance for general/unclassified competitions"""
    
    def __init__(self, data_description: Dict):
        self.data_description = data_description
        
    def get_context(self) -> str:
        """Provide general ML guidance when type is unclear"""
        
        return """# General Machine Learning Guidance

APPROACH:
- First understand the problem type and success metric
- Analyze all available data files and formats
- Start simple, then increase complexity
- Focus on creating a working baseline first

KEY CONSIDERATIONS:
1. Problem Understanding:
   - What are we trying to predict?
   - What metric defines success?
   - What constraints exist (time, memory, accuracy)?
   - Are there business rules to follow?

2. Data Analysis:
   - Explore all provided files
   - Understand relationships between files
   - Identify train/test/validation splits
   - Check data quality and consistency

3. Baseline Approach:
   - Start with simplest reasonable model
   - Use default parameters initially
   - Ensure pipeline works end-to-end
   - Establish performance baseline

4. Iterative Improvement:
   - Add complexity gradually
   - Validate each change improves performance
   - Keep track of what works and what doesn't
   - Don't overthink - simple often wins

5. Model Selection:
   - Consider data size and type
   - Try multiple algorithms
   - Ensemble if time permits
   - Validate thoroughly

6. Best Practices:
   - Set random seeds for reproducibility
   - Use proper validation strategy
   - Monitor for overfitting
   - Create clean, modular code
   - Save intermediate results

UNIVERSAL PRINCIPLES:
- Data quality > Model complexity
- Feature engineering > Hyperparameter tuning
- Ensemble > Single model
- Cross-validation > Single split
- Simple baseline > Complex initial approach

Generate a solution that:
1. Works reliably end-to-end
2. Produces valid submissions
3. Follows ML best practices
4. Is appropriate for the data scale

Remember: A working solution that scores moderately is better than a complex solution that fails."""
