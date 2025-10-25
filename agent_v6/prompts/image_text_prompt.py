"""Image-to-Text Specialized Prompt"""

from typing import Dict


class SpecializedPrompt:
    """Guidance for image-to-text competitions"""
    
    def __init__(self, data_description: Dict):
        self.data_description = data_description
        
    def get_context(self) -> str:
        """Provide guidance for image-to-text approach"""
        
        return """# Image-to-Text Guidance

APPROACH:
- Understand the specific task requirements and constraints
- Analyze the data format and characteristics
- Choose appropriate methods based on data scale
- Start with proven baselines before complex approaches

KEY CONSIDERATIONS:
- Task type: captioning, OCR, VQA
- Image-text alignment
- Sequence generation strategy
- Attention mechanisms
- Evaluation metrics (BLEU, CIDEr)

SUGGESTED APPROACHES:
- Encoder-decoder architectures
- Vision transformers + GPT
- CLIP-based models
- OCR: Tesseract, EasyOCR

COMMON PITFALLS TO AVOID:
- Mismatched image-text pairs
- Poor tokenization strategy
- Not handling variable length outputs
- Ignoring visual context

IMPLEMENTATION STRATEGY:
1. Start with data exploration and understanding
2. Implement a simple baseline
3. Iterate with more sophisticated approaches
4. Validate thoroughly at each step
5. Ensemble if beneficial

Generate your solution based on these principles, adapting to the specific competition requirements and constraints."""
