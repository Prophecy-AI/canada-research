"""Image Segmentation Specialized Prompt"""

from typing import Dict


class SpecializedPrompt:
    """Guidance for image segmentation competitions"""
    
    def __init__(self, data_description: Dict):
        self.data_description = data_description
        
    def get_context(self) -> str:
        """Provide guidance for image segmentation approach"""
        
        return """# Image Segmentation Guidance

APPROACH:
- Understand the specific task requirements and constraints
- Analyze the data format and characteristics
- Choose appropriate methods based on data scale
- Start with proven baselines before complex approaches

KEY CONSIDERATIONS:
- Semantic vs instance segmentation
- Mask encoding/decoding
- Loss functions (Dice, IoU, BCE)
- Data augmentation for masks
- Post-processing techniques

SUGGESTED APPROACHES:
- U-Net and variants
- DeepLab family
- Mask R-CNN for instance
- Transformer-based (SegFormer)

COMMON PITFALLS TO AVOID:
- Mask-image misalignment in augmentation
- Class imbalance in pixels
- Not using appropriate loss
- Poor boundary handling

IMPLEMENTATION STRATEGY:
1. Start with data exploration and understanding
2. Implement a simple baseline
3. Iterate with more sophisticated approaches
4. Validate thoroughly at each step
5. Ensemble if beneficial

Generate your solution based on these principles, adapting to the specific competition requirements and constraints."""
