"""Reinforcement Learning Specialized Prompt"""

from typing import Dict


class SpecializedPrompt:
    """Guidance for reinforcement learning competitions"""
    
    def __init__(self, data_description: Dict):
        self.data_description = data_description
        
    def get_context(self) -> str:
        """Provide guidance for reinforcement learning approach"""
        
        return """# Reinforcement Learning Guidance

APPROACH:
- Understand the specific task requirements and constraints
- Analyze the data format and characteristics
- Choose appropriate methods based on data scale
- Start with proven baselines before complex approaches

KEY CONSIDERATIONS:
- Environment dynamics understanding
- State/action space analysis
- Reward shaping
- Exploration vs exploitation
- Episode length and termination

SUGGESTED APPROACHES:
- Value-based: DQN, Rainbow
- Policy gradient: PPO, A2C, SAC
- Model-based approaches
- Imitation learning if demos available

COMMON PITFALLS TO AVOID:
- Poor reward design
- Insufficient exploration
- Not normalizing observations
- Training instability

IMPLEMENTATION STRATEGY:
1. Start with data exploration and understanding
2. Implement a simple baseline
3. Iterate with more sophisticated approaches
4. Validate thoroughly at each step
5. Ensemble if beneficial

Generate your solution based on these principles, adapting to the specific competition requirements and constraints."""
