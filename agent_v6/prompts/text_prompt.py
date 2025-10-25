"""Text Classification Specialized Prompt"""

from typing import Dict


class SpecializedPrompt:
    """Guidance for text classification competitions"""
    
    def __init__(self, data_description: Dict):
        self.data_description = data_description
        
    def get_context(self) -> str:
        """Provide guidance for text classification approach"""
        
        return """# Text Classification Guidance

APPROACH:
- Analyze text characteristics: length, language, domain
- Determine if multi-class or multi-label classification
- Choose between traditional ML and deep learning based on data size
- Consider computational constraints

KEY CONSIDERATIONS:
1. Text Preprocessing:
   - Cleaning: Handle special characters, HTML, URLs
   - Case normalization (depends on domain)
   - Tokenization strategy (word, subword, character)
   - Handling missing or empty texts

2. Feature Engineering (Traditional ML):
   - TF-IDF with appropriate parameters
   - N-grams (unigrams, bigrams, trigrams)
   - Text statistics (length, punctuation, capitals)
   - Domain-specific features if relevant

3. Deep Learning Approaches:
   - Small data: Use pretrained embeddings (Word2Vec, GloVe)
   - Medium data: Fine-tune BERT-based models
   - Large data: Train custom architectures
   - Consider: BERT, RoBERTa, DistilBERT, ALBERT

4. Model Selection Strategy:
   - <5K samples: Traditional ML (Logistic Regression, SVM, Random Forest)
   - 5K-50K: Light transformers or CNN/LSTM with embeddings
   - >50K: Full transformer fine-tuning

5. Training Considerations:
   - Handle class imbalance (weighted loss, oversampling)
   - Text-specific augmentation (paraphrasing, back-translation)
   - Appropriate sequence length truncation
   - Batch size based on sequence length

6. Optimization Tips:
   - Start with simple baseline (TF-IDF + Logistic Regression)
   - Ensemble different approaches (traditional + deep learning)
   - Use appropriate evaluation metrics for the task

COMMON PITFALLS TO AVOID:
- Over-preprocessing (removing too much information)
- Ignoring class imbalance
- Using transformers on tiny datasets
- Not setting random seeds for reproducibility
- Forgetting to handle out-of-vocabulary words

Generate your solution based on these principles and the specific data characteristics."""
