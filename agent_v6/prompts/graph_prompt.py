"""Graph ML Specialized Prompt"""

from typing import Dict


class SpecializedPrompt:
    """Guidance for graph ml competitions"""
    
    def __init__(self, data_description: Dict):
        self.data_description = data_description
        
    def get_context(self) -> str:
        """Provide guidance for graph ml approach"""
        
        return """# Graph ML Guidance

APPROACH:
- Understand the specific task requirements and constraints
- Analyze the data format and characteristics
- Choose appropriate methods based on data scale
- Start with proven baselines before complex approaches

KEY CONSIDERATIONS:
- Graph structure: directed, weighted
- Node vs edge vs graph prediction
- Feature engineering on graphs
- Message passing understanding
- Graph sampling strategies

SUGGESTED APPROACHES:
- Traditional: PageRank, Node2Vec
- GNNs: GCN, GraphSAGE, GAT
- Graph transformers
- Spectral methods

COMMON PITFALLS TO AVOID:
- Not handling disconnected components
- Ignoring graph properties
- Over-smoothing in deep GNNs
- Wrong adjacency normalization

IMPLEMENTATION STRATEGY:
1. Start with data exploration and understanding
2. Implement a simple baseline
3. Iterate with more sophisticated approaches
4. Validate thoroughly at each step
5. Ensemble if beneficial

Generate your solution based on these principles, adapting to the specific competition requirements and constraints."""
