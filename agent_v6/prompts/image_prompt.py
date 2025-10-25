"""Image Classification Specialized Prompt"""

from typing import Dict


class SpecializedPrompt:
    """Guidance for image classification competitions"""
    
    def __init__(self, data_description: Dict):
        self.data_description = data_description
        
    def get_context(self) -> str:
        """Provide guidance for image classification approach"""
        
        return """# Image Classification Guidance

APPROACH:
- Analyze image dimensions, channels (RGB/grayscale), and format
- Consider if images need resizing or normalization
- Think about appropriate data augmentation based on the domain
- Choose model architecture based on dataset size and complexity

KEY CONSIDERATIONS:
1. Data Loading:
   - Efficient image loading (consider memory constraints)
   - Handle various image formats (jpg, png, etc.)
   - Implement proper train/validation split

2. Preprocessing:
   - Normalization strategy (ImageNet stats vs custom)
   - Resizing approach (maintain aspect ratio vs squash)
   - Channel handling (RGB, grayscale, RGBA)

3. Augmentation Strategy:
   - Light: Basic flips and slight rotations
   - Medium: Add color jittering, random crops
   - Heavy: Include distortions, cutout, mixup
   - Domain-specific: Consider what makes sense for your images

4. Model Selection:
   - Small dataset (<1000): Consider transfer learning with frozen backbone
   - Medium dataset: Fine-tune pretrained models
   - Large dataset: Train from scratch or full fine-tuning
   - Consider: ResNet, EfficientNet, Vision Transformers, ConvNets

5. Training Strategy:
   - Start with lower learning rate for pretrained models
   - Use appropriate loss (CrossEntropy for single-label, BCE for multi-label)
   - Monitor validation metrics to prevent overfitting
   - Consider: learning rate scheduling, early stopping

6. Optimization Tips:
   - Mixed precision training for faster execution
   - Gradient accumulation for larger effective batch sizes
   - Test-time augmentation for improved predictions

COMMON PITFALLS TO AVOID:
- Not checking image statistics before normalization
- Too aggressive augmentation that changes image semantics
- Forgetting to resize images consistently
- Not using pretrained weights when available
- Overfitting on small datasets

Generate your solution considering these guidelines and the specific competition requirements."""
