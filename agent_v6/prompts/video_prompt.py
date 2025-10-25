"""Video Processing Specialized Prompt"""

from typing import Dict


class SpecializedPrompt:
    """Guidance for video processing competitions"""
    
    def __init__(self, data_description: Dict):
        self.data_description = data_description
        
    def get_context(self) -> str:
        """Provide guidance for video processing approach"""
        
        return """# Video Processing Guidance

APPROACH:
- Understand the specific task requirements and constraints
- Analyze the data format and characteristics
- Choose appropriate methods based on data scale
- Start with proven baselines before complex approaches

KEY CONSIDERATIONS:
- Temporal and spatial dimensions
- Frame sampling strategy
- Memory constraints
- Motion information
- Video-level vs frame-level prediction

SUGGESTED APPROACHES:
- Frame-based: CNN + aggregation
- 3D CNNs (C3D, I3D)
- Two-stream networks
- Video transformers (TimeSformer)

COMMON PITFALLS TO AVOID:
- Processing all frames (too expensive)
- Ignoring temporal consistency
- Poor frame sampling
- Memory overflow

IMPLEMENTATION STRATEGY:
1. Start with data exploration and understanding
2. Implement a simple baseline
3. Iterate with more sophisticated approaches
4. Validate thoroughly at each step
5. Ensemble if beneficial

Generate your solution based on these principles, adapting to the specific competition requirements and constraints."""
