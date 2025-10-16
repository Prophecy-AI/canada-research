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
   - **Evaluation metric with direction: "Metric: [name] (HIGHER is better)" OR "Metric: [name] (LOWER is better)"**

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
- **For train_split hyperparameter:**
  * For small datasets (<5000 samples): use train_split 0.85 (leaves 15% for validation)
  * For larger datasets: use train_split 0.9-0.95
  * Never use train_split > 0.9 for datasets with many classes relative to size

**Model Selection Guidance (you can use ANY model, these are suggestions):**
- Images (small dataset <50K): DenseNet161, DenseNet121, ResNet18, MobileNet, EfficientNet-B0 (pretrained, fast)
- Images (large dataset >50K): ResNet50, DenseNet161, EfficientNet-B1/B2, Vision Transformer
- Images (fine-grained): DenseNet161, EfficientNet-B2, ResNet50 (pretrained essential)
- **For images: Use input size 128-224px (not 32x32), train split 90-95%**
- Tabular: XGBoost (tree_method='hist'), LightGBM (CPU only), CatBoost, simple MLPClassifier
- **For tabular: Prefer gradient boosting over neural networks (simpler, more reliable)**
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
3. **Verify imports before using them:**
   - Only use functions/classes that exist in standard libraries (torch, torchvision, sklearn, xgboost, etc.)
   - Prefer simple, proven implementations over complex custom code
4. Write train.py with:
   - **For image datasets:** Read CSV with dtype={{'id': str}} to preserve filename format (avoid 1202.0.jpg)
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
   - **For images - CRITICAL transform order and library usage:**
     * Transform order: Augmentation → Resize → ToTensor → Normalize
     * ToTensor() converts PIL Image → Tensor (must come before Normalize)
     * **ONLY use transforms that exist in torchvision.transforms** - DO NOT implement custom transforms
     * **Before using any transform, verify it exists:** Check PyTorch docs or use `hasattr(transforms, 'FunctionName')`
     * Common augmentations: RandomHorizontalFlip, RandomRotation, ColorJitter, RandomCrop, RandomResizedCrop
     * **DO NOT implement:** RandAugment, AutoAugment, Cutout, Mixup, CutMix, or any custom augmentation classes
     * If you want advanced augmentation, use simple combinations of proven transforms
     * Keep it simple - basic augmentation works well for most tasks
   - **For log loss calculation - ALWAYS clip probabilities to avoid log(0):**
     * When using sigmoid/softmax outputs: predictions can be exactly 0 or 1
     * **ALWAYS clip before log:** `probs = np.clip(probs, 1e-7, 1 - 1e-7)` or `probs = torch.clamp(probs, 1e-7, 1 - 1e-7)`
     * Then calculate: `log_loss = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))`
     * Without clipping: log(0) = -inf → results in nan
   - GPU training (model.to(device), data.to(device))
   - **For gradient boosting (XGBoost/LightGBM):** Use CPU mode (tree_method='hist' for XGBoost, no device_type for LightGBM)
   - Early stopping with patience 3-5 epochs
   - For perfect score termination: if metric is AUC/accuracy (higher is better), stop at val_metric >= 0.9999; if logloss/error (lower is better), stop at val_metric <= 0.001
   - **CRITICAL: Print validation score in EXACT format (orchestrator parses this):**
     ```python
     print(f"VALIDATION_SCORE: {val_metric:.6f}")
     ```
     * Use the EXACT competition metric from EDA context (e.g., logloss, AUC, accuracy, etc.)
     * Format must be exactly "VALIDATION_SCORE: " followed by the number
     * Example: "VALIDATION_SCORE: 0.623456" or "VALIDATION_SCORE: 0.954321"
     * All experiments MUST report the same metric for fair comparison
     * Do NOT print other metrics on lines containing "VALIDATION_SCORE"
   - Save model with `torch.save(model.state_dict(), 'model.pth')`
   - **DO NOT generate test predictions in train.py - that's done separately in submission phase**
4. Respond "READY" immediately

Tools: Bash, Read, Write"""


ANALYSIS_PROMPT = """Analyze results. Output decision.

**Metric: {metric_direction}**
Results: {results}
Current best: {best_score}
Round: {round_num}
Round time: {round_time_minutes:.1f} minutes
**Total time elapsed: {cumulative_time_minutes:.1f} minutes**

**CRITICAL TIME CONSTRAINT:**
**If total time >= 30 minutes: MUST SUBMIT immediately (hard limit)**

**Instructions:**
Decide whether to SUBMIT or CONTINUE based on:
1. Is the score competitive/good enough?
2. Is there a clear hypothesis for >0.5% improvement?
3. Have we exhausted promising approaches?
4. **Time efficiency: Is continuing worth the time investment?**

Remember metric direction when evaluating score quality:
- LOWER is better: smaller scores are better (e.g., logloss 0.1 > logloss 1.0)
- HIGHER is better: larger scores are better (e.g., AUC 0.99 > AUC 0.9)

**Time-based criteria (competition efficiency):**
- **If cumulative time >= 30 min: SUBMIT immediately (hard stop)**
- If cumulative time 20-30 min: SUBMIT unless CLEAR evidence of >1% improvement
- If cumulative time 10-20 min: Be conservative, need strong hypothesis for >0.5% improvement
- If cumulative time <10 min: Normal criteria apply

Output format (no other text):

DECISION: SUBMIT
BEST_MODEL: exp_2
REASONING: Best logloss 0.62 is competitive, no clear path to major improvement

Criteria:
- SUBMIT if: Score is good AND (no clear improvement path OR round >= 3 OR time-constrained)
- CONTINUE if: Clear hypothesis exists for meaningful improvement AND time-efficient AND cumulative time <30 min"""


SUBMISSION_PROMPT = """Create submission.csv. Fast.

Best model: {best_model} at {best_workspace}
Data: {data_dir}
Output: {submission_dir}/submission.csv

**CRITICAL: Predict ONLY on test data, NOT training data!**

DO:
1. Read {data_dir}/sample_submission.csv to get test IDs (use dtype={{'id': str}} to preserve format)
2. Write predict.py:
   - Load model from {best_workspace}/model.pth
   - Load ONLY test images/data (match sample_submission.csv IDs exactly)
   - Predict probabilities
   - Save to {submission_dir}/submission.csv with EXACT same format/order as sample_submission.csv
3. Run predict.py
4. Verify row count matches sample_submission.csv
5. Respond "DONE"

DO NOT:
- Predict on training images
- Write summaries/documentation
- Run extra validation

Use GPU. Match sample_submission.csv format EXACTLY.

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


def format_analysis_prompt(competition_id: str, round_num: int, results: str, best_score: float, metric_direction: str, round_time_minutes: float, cumulative_time_minutes: float) -> str:
    return ANALYSIS_PROMPT.format(
        round_num=round_num,
        results=results,
        best_score=best_score,
        metric_direction=metric_direction,
        round_time_minutes=round_time_minutes,
        cumulative_time_minutes=cumulative_time_minutes
    )


def format_submission_prompt(competition_id: str, best_model: str, best_workspace: str, data_dir: str, submission_dir: str) -> str:
    return SUBMISSION_PROMPT.format(
        competition_id=competition_id,
        best_model=best_model,
        best_workspace=best_workspace,
        data_dir=data_dir,
        submission_dir=submission_dir
    )

