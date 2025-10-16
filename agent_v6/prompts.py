EDA_PROMPT = """Analyze competition data. Write ONE eda.py, run ONCE, report findings.

Competition: {competition_id}
Data: {data_dir}
Instructions: {instructions_path}

**Task:**
1. Read instructions
2. Write comprehensive eda.py (data shape, types, target distribution, class balance, file formats)
3. Run it ONCE
4. Report findings (3-5 sentences):
   - Data type (tabular/image/text/time-series)
   - Dataset size and shape
   - Target distribution (balanced/imbalanced)
   - Key patterns or characteristics
   - Evaluation metric

NO model suggestions. NO iteration. ONE script, ONE run."""


PLANNING_PROMPT = """You are an expert ML engineer. Design 2-3 NEW experiments based on dataset analysis.

Competition: {competition_id}
Round: {round_num}
Best: {best_score}

{context}

**TASK: Analyze the data characteristics above, then propose 2-3 DIFFERENT modeling approaches.**

**Step 1: Reason about the dataset**
- Data type: tabular/image/text/time-series?
- Size: small (<10K), medium (10K-100K), large (>100K)?
- Complexity: number of features, classes, dimensionality?
- Target: classification/regression? Binary/multiclass? Imbalanced?
- Metric: what does it optimize for?

**Step 2: Choose appropriate models**
Based on dataset characteristics, select models that are:
- DIFFERENT from previous experiments (do not repeat)
- Appropriate for the data type and size
- Fast to train (10-15 epochs, early stopping)
- Batch size 32-64 for parallel GPU training

**Model Selection Guidance (you can use ANY model, these are suggestions):**
- Images (small dataset <50K): DenseNet161, DenseNet121, ResNet18, MobileNet, EfficientNet-B0 (pretrained, fast)
- Images (large dataset >50K): ResNet50, DenseNet161, EfficientNet-B1/B2, Vision Transformer
- Images (fine-grained): DenseNet161, EfficientNet-B2, ResNet50 (pretrained essential)
- **For images: Use input size 128-224px (not 32x32), train split 90-95%**
- Tabular: XGBoost, LightGBM, CatBoost, Neural Networks (TabNet)
- Text: **ONLY use HuggingFace transformers** (distilbert-base-uncased, bert-base-uncased, roberta-base)
  * Use AutoTokenizer + AutoModelForSequenceClassification from transformers library
  * OR simple baselines: TF-IDF + LogisticRegression/XGBoost
- **DO NOT propose: Custom LSTM, BiLSTM, TextCNN, GloVe/FastText embeddings - libraries not installed**
- Time-series: XGBoost, LightGBM (gradient boosting works well), simple LSTM if needed

**Feel free to propose other models (VGG, Inception, SENet, etc.) if they fit the task better.**

**Step 3: Output ONLY JSON (NO text before/after):**

[
  {{
    "id": "exp_1",
    "model": "<model_name>",
    "features": {{"type": "<feature_type>", "details": "..."}},
    "hyperparameters": {{"device": "cuda", "epochs": 10-15, "lr": 0.0001-0.01, "batch_size": 32-64}},
    "hypothesis": "<why this model is appropriate for THIS dataset>"
  }},
  {{
    "id": "exp_2",
    "model": "<different_model>",
    "features": {{"type": "<feature_type>", "details": "..."}},
    "hyperparameters": {{"device": "cuda", "epochs": 10-15, "lr": 0.0001-0.01, "batch_size": 32-64}},
    "hypothesis": "<why this model complements exp_1>"
  }}
]"""


WORKER_PROMPT = """Write train.py for this experiment. DO NOT RUN IT.

Experiment: {spec}
Data: {data_dir}

**EDA Context (use this to understand the problem):**
{eda_context}

**CRITICAL: Your ONLY job is to write train.py. DO NOT:**
- Run train.py (orchestrator will run it)
- Write summaries/documentation
- Test imports
- Create verification scripts

**DO:**
1. Check data structure (use Bash: ls, zipinfo, head CSV)
2. Extract zip files to workspace if needed (unzip -q /home/data/train.zip -d .)
3. Write train.py with:
   - GPU memory cleanup: `import torch; torch.cuda.empty_cache()` at start
   - Correct data loading based on structure you found
   - Model/features/hyperparameters from spec (use EXACT batch_size from spec)
   - **For text data:**
     * Use: `from transformers import AutoTokenizer, AutoModelForSequenceClassification`
     * Model examples: 'distilbert-base-uncased', 'bert-base-uncased', 'roberta-base'
     * Tokenizer: `tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')`
     * DO NOT build custom vocabulary or use GloVe/FastText - use transformers only
   - **Correct loss function:**
     * Binary classification (2 classes): `nn.BCEWithLogitsLoss()`
     * Multiclass classification (>2 classes): `nn.CrossEntropyLoss(label_smoothing=0.1)` 
     * Multi-label classification (multiple labels per sample): `nn.BCEWithLogitsLoss()`
     * Regression: `nn.MSELoss()` or `nn.L1Loss()`
   - **Train/validation split - CRITICAL:**
     ```python
     # For multiclass: ensure val_size >= num_classes
     total = len(df)
     test_size = 0.15 if total < 10 * num_classes else 0.10
     try:
         X_train, X_val, y_train, y_val = train_test_split(
             X, y, test_size=test_size, stratify=y, random_state=42
         )
     except ValueError:  # Not enough samples per class
         X_train, X_val, y_train, y_val = train_test_split(
             X, y, test_size=test_size, random_state=42
         )
     ```
   - For images: use larger input size (128-224, not 32x32) to preserve details
   - GPU training (model.to(device), data.to(device))
   - Early stopping with immediate perfect score termination:
     ```python
     if val_metric > best_metric:
         best_metric = val_metric
         save_model()
         patience = 0
         if val_metric >= 0.9999:  # Perfect or near-perfect
             print(f"Perfect score reached: {val_metric:.4f}, stopping")
             break
     else:
         patience += 1
         if patience >= max_patience:
             break
     ```
   - Print validation score as "VALIDATION_SCORE: X.XXXX" (accuracy or metric value)
   - Save model with `torch.save(model.state_dict(), 'model.pth')`
4. Respond "READY" immediately

Tools: Bash, Read, Write"""


ANALYSIS_PROMPT = """Analyze experiment results. Output decision only.

Results: {results}
Best so far: {best_score}
Target: {submit_threshold}

Output ONLY this format (no other text):

DECISION: SUBMIT
BEST_MODEL: exp_1
REASONING: Score 0.996 > target 0.85, no clear path to 0.5% improvement

Criteria:
- SUBMIT if: score > {submit_threshold} OR improvement < 0.005
- CONTINUE if: clear hypothesis for >0.5% improvement"""


SUBMISSION_PROMPT = """Create submission.csv. Fast.

Best model: {best_model} at {best_workspace}
Data: {data_dir}
Output: {submission_dir}/submission.csv

DO:
1. Write predict.py: Load model, predict on test, save to {submission_dir}/submission.csv
2. Run predict.py
3. Respond "DONE"

DO NOT:
- Write summaries/READMEs/documentation
- Create verification scripts
- Run extra validation checks

Match sample_submission.csv format. Use GPU.

Tools: Bash, Read, Write"""


def format_eda_prompt(competition_id: str, data_dir: str, instructions_path: str) -> str:
    return EDA_PROMPT.format(
        competition_id=competition_id,
        data_dir=data_dir,
        instructions_path=instructions_path
    )


def format_planning_prompt(competition_id: str, context: str, round_num: int, best_score: float = 0.0) -> str:
    return PLANNING_PROMPT.format(
        competition_id=competition_id,
        context=context,
        round_num=round_num,
        best_score=best_score
    )


def format_worker_prompt(spec: dict, data_dir: str, workspace_dir: str, eda_context: str) -> str:
    import json
    spec_str = json.dumps(spec, indent=2)
    return WORKER_PROMPT.format(
        spec=spec_str,
        data_dir=data_dir,
        workspace_dir=workspace_dir,
        eda_context=eda_context
    )


def format_analysis_prompt(competition_id: str, round_num: int, results: str, best_score: float, metric: str = "accuracy", submit_threshold: float = 0.85) -> str:
    return ANALYSIS_PROMPT.format(
        competition_id=competition_id,
        round_num=round_num,
        results=results,
        best_score=best_score,
        metric=metric,
        submit_threshold=submit_threshold
    )


def format_submission_prompt(competition_id: str, best_model: str, best_workspace: str, data_dir: str, submission_dir: str) -> str:
    return SUBMISSION_PROMPT.format(
        competition_id=competition_id,
        best_model=best_model,
        best_workspace=best_workspace,
        data_dir=data_dir,
        submission_dir=submission_dir
    )

