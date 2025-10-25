"""Clustering Specialized Prompt"""

from typing import Dict


class SpecializedPrompt:
    """Guidance for clustering competitions"""
    
    def __init__(self, data_description: Dict):
        self.data_description = data_description
        
    def get_context(self) -> str:
        """Provide guidance for clustering approach"""
        
        return """# Clustering Guidance

APPROACH:
- Understand the specific task requirements and constraints
- Analyze the data format and characteristics
- Choose appropriate methods based on data scale
- Start with proven baselines before complex approaches

KEY CONSIDERATIONS:
- Number of clusters determination
- Distance metrics selection
- Feature scaling importance
- Cluster validation metrics
- Outlier handling

SUGGESTED APPROACHES:
- K-means for spherical clusters
- DBSCAN for arbitrary shapes
- Hierarchical for dendrograms
- Gaussian Mixture Models

COMMON PITFALLS TO AVOID:
- Not scaling features
- Wrong distance metric
- Forcing wrong k
- Ignoring cluster quality

IMPLEMENTATION STRATEGY:
1. Start with data exploration and understanding
2. Implement a simple baseline
3. Iterate with more sophisticated approaches
4. Validate thoroughly at each step
5. Ensemble if beneficial

Generate your solution based on these principles, adapting to the specific competition requirements and constraints."""
