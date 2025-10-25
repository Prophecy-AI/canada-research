"""Object Detection Specialized Prompt"""

from typing import Dict


class SpecializedPrompt:
    """Guidance for object detection competitions"""
    
    def __init__(self, data_description: Dict):
        self.data_description = data_description
        
    def get_context(self) -> str:
        """Provide guidance for object detection approach"""
        
        return """# Object Detection Guidance

APPROACH:
- Understand the specific task requirements and constraints
- Analyze the data format and characteristics
- Choose appropriate methods based on data scale
- Start with proven baselines before complex approaches

KEY CONSIDERATIONS:
- Bounding box format (YOLO, COCO, Pascal VOC)
- Image preprocessing and augmentation
- Anchor box configuration
- NMS threshold tuning
- mAP evaluation metric

SUGGESTED APPROACHES:
- YOLO family (v5, v7, v8)
- Faster R-CNN, RetinaNet
- EfficientDet for efficiency
- Consider pretrained models

COMMON PITFALLS TO AVOID:
- Wrong bbox format conversion
- Inappropriate image resizing
- Not handling class imbalance
- Ignoring small objects

IMPLEMENTATION STRATEGY:
1. Start with data exploration and understanding
2. Implement a simple baseline
3. Iterate with more sophisticated approaches
4. Validate thoroughly at each step
5. Ensemble if beneficial

Generate your solution based on these principles, adapting to the specific competition requirements and constraints."""
