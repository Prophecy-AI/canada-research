"""Anomaly Detection Specialized Prompt"""

from typing import Dict


class SpecializedPrompt:
    """Guidance for anomaly detection competitions"""
    
    def __init__(self, data_description: Dict):
        self.data_description = data_description
        
    def get_context(self) -> str:
        """Provide guidance for anomaly detection approach"""
        
        return """# Anomaly Detection Guidance

APPROACH:
- Understand the specific task requirements and constraints
- Analyze the data format and characteristics
- Choose appropriate methods based on data scale
- Start with proven baselines before complex approaches

KEY CONSIDERATIONS:
- Anomaly types: point, contextual, collective
- Supervised vs unsupervised
- Contamination rate estimation
- Feature selection importance
- Threshold selection

SUGGESTED APPROACHES:
- Statistical: Z-score, IQR
- Isolation Forest, LOF
- Autoencoders for complex data
- One-class SVM

COMMON PITFALLS TO AVOID:
- Wrong contamination assumption
- Not handling different scales
- Overfitting on normal data
- Poor threshold selection

IMPLEMENTATION STRATEGY:
1. Start with data exploration and understanding
2. Implement a simple baseline
3. Iterate with more sophisticated approaches
4. Validate thoroughly at each step
5. Ensemble if beneficial

Generate your solution based on these principles, adapting to the specific competition requirements and constraints."""
