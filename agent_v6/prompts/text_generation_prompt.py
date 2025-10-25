"""Text Generation Specialized Prompt"""

from typing import Dict


class SpecializedPrompt:
    """Guidance for text generation competitions"""
    
    def __init__(self, data_description: Dict):
        self.data_description = data_description
        
    def get_context(self) -> str:
        """Provide guidance for text generation approach"""
        
        return """# Text Generation Guidance

APPROACH:
- Understand the specific task requirements and constraints
- Analyze the data format and characteristics
- Choose appropriate methods based on data scale
- Start with proven baselines before complex approaches

KEY CONSIDERATIONS:
- Generation strategy (greedy, beam, sampling)
- Context length management
- Prompt engineering
- Output quality metrics
- Controlling generation parameters

SUGGESTED APPROACHES:
- Fine-tune GPT-style models
- Seq2seq with attention
- Prompt-based few-shot learning
- Retrieval-augmented generation

COMMON PITFALLS TO AVOID:
- Repetitive or incoherent output
- Not setting proper stop tokens
- Poor prompt design
- Ignoring context limits

IMPLEMENTATION STRATEGY:
1. Start with data exploration and understanding
2. Implement a simple baseline
3. Iterate with more sophisticated approaches
4. Validate thoroughly at each step
5. Ensemble if beneficial

Generate your solution based on these principles, adapting to the specific competition requirements and constraints."""
