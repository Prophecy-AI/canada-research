"""Audio Processing Specialized Prompt"""

from typing import Dict


class SpecializedPrompt:
    """Guidance for audio processing competitions"""
    
    def __init__(self, data_description: Dict):
        self.data_description = data_description
        
    def get_context(self) -> str:
        """Provide guidance for audio processing approach"""
        
        return """# Audio Processing Guidance

APPROACH:
- Understand the specific task requirements and constraints
- Analyze the data format and characteristics
- Choose appropriate methods based on data scale
- Start with proven baselines before complex approaches

KEY CONSIDERATIONS:
- Audio representations: waveform, spectrogram, MFCC
- Sampling rate considerations
- Window size and hop length
- Noise handling
- Augmentation strategies

SUGGESTED APPROACHES:
- Traditional: MFCC + ML
- CNNs on spectrograms
- RNNs for sequences
- Transformers (Wav2Vec, Whisper)

COMMON PITFALLS TO AVOID:
- Wrong sampling rate
- Not handling silence
- Poor windowing choices
- Ignoring phase information

IMPLEMENTATION STRATEGY:
1. Start with data exploration and understanding
2. Implement a simple baseline
3. Iterate with more sophisticated approaches
4. Validate thoroughly at each step
5. Ensemble if beneficial

Generate your solution based on these principles, adapting to the specific competition requirements and constraints."""
