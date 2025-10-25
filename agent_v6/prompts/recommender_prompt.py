"""Recommender Systems Specialized Prompt"""

from typing import Dict


class SpecializedPrompt:
    """Guidance for recommender systems competitions"""
    
    def __init__(self, data_description: Dict):
        self.data_description = data_description
        
    def get_context(self) -> str:
        """Provide guidance for recommender systems approach"""
        
        return """# Recommender Systems Guidance

APPROACH:
- Understand the specific task requirements and constraints
- Analyze the data format and characteristics
- Choose appropriate methods based on data scale
- Start with proven baselines before complex approaches

KEY CONSIDERATIONS:
- Implicit vs explicit feedback
- Cold start problem
- Evaluation: NDCG, MAP, recall@k
- User-item interaction matrix
- Feature engineering for users/items

SUGGESTED APPROACHES:
- Collaborative filtering (Matrix Factorization)
- Content-based filtering
- Deep learning (Neural CF, AutoRec)
- Hybrid approaches

COMMON PITFALLS TO AVOID:
- Not handling sparsity
- Popularity bias
- Not considering temporal aspects
- Wrong train/test splitting

IMPLEMENTATION STRATEGY:
1. Start with data exploration and understanding
2. Implement a simple baseline
3. Iterate with more sophisticated approaches
4. Validate thoroughly at each step
5. Ensemble if beneficial

Generate your solution based on these principles, adapting to the specific competition requirements and constraints."""
