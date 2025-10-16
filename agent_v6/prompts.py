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

**Step 2: Choose strategy and models based on data characteristics**

**CRITICAL: ONLY use models from standard libraries (torchvision.models, transformers, sklearn, xgboost, lightgbm)**
**DO NOT propose custom architectures or models implemented from scratch - worker will fail**

**Available Strategies (choose mix based on EDA context above):**

**Strategy 1: "bottleneck_features"** (feature extraction + simple classifier)
- **When to try:** Images + Classification, especially with <50K samples
- **How it works:** Extract features from pretrained CNN (no training), train LogisticRegression/XGBoost on features
- **Pros:** Very fast (trains in seconds), often better with limited data, low memory
- **Single model:** Use one backbone (EfficientNet-B0, ResNet50, DenseNet121, MobileNetV2)
- **Multi-model (RECOMMENDED for max performance):** Use 2-3 backbones, concatenate features, then train classifier
  * Much better performance (2-10x lower logloss)
  * Still fast (parallel feature extraction)
  * **Best model combinations (from gold solutions):** 
    - ResNet50 (2048-dim) + InceptionV3 (2048-dim) = 4096-dim
    - EfficientNet-B2 (1408-dim) + DenseNet161 (2208-dim) + ResNet50 (2048-dim) = 5664-dim
    - Wide_ResNet50_2 (2048-dim) + RegNetY_8GF (2016-dim) = 4064-dim
- **Example single:** {{"strategy": "bottleneck_features", "model": "ResNet50", "classifier": "LogisticRegression"}}
- **Example multi:** {{"strategy": "bottleneck_features", "models": ["ResNet50", "InceptionV3"], "classifier": "LogisticRegression"}}

**Strategy 2: "fine_tuning"** (standard deep learning)
- **When to try:** Images + Classification, any dataset size
- **How it works:** Full CNN training with pretrained weights, train 10-15 epochs
- **Pros:** Can learn task-specific features, proven approach
- **Models:** DenseNet161, EfficientNet-B0/B1, ResNet18/50, MobileNet
- **Example:** {{"strategy": "fine_tuning", "model": "ResNet50", "epochs": 12}}

**Strategy 3: "gradient_boosting"** (for tabular)
- **When to try:** Tabular data
- **Models:** XGBoost (tree_method='hist'), LightGBM (CPU), CatBoost

**Strategy 4: "transformer_features"** (for text)
- **When to try:** Text data
- **Models:** distilbert-base-uncased, bert-base-uncased, roberta-base

**Recommendation for Round 1:**
- **Images <50K (espescially for small sample sizes):** 
  * **FOCUS ON BOTTLENECK ONLY** - fine-tuning wastes time and gets worse scores
  * exp_1: Multi-model bottleneck (ResNet50 + InceptionV3)
  * exp_2: Multi-model bottleneck (EfficientNet-B2 + DenseNet161)
  * exp_3: Multi-model bottleneck (Wide_ResNet50_2 + RegNetY_8GF) OR single EfficientNet-B0 for speed
  * This explores WHICH model combinations work best, all finish in 3-5 min each
- **Images >50K:** Try 2-3 different fine_tuning models (enough data to train)
- **Tabular:** Try 2-3 different gradient boosting models
- **Multi-model bottleneck >> fine-tuning for small datasets** (faster AND better scores)

**Experiment Design Guidelines:**
- **For images <50K: Use ONLY bottleneck strategies (skip fine-tuning - wastes time and gets worse results)**
- **Multi-model selection:** Choose 2-3 complementary backbones with different architectures:
  * Good pairs: ResNet50 + InceptionV3, DenseNet161 + Wide_ResNet50_2, EfficientNet-B2 + ResNet50
  * Good triplets: EfficientNet-B2 + DenseNet121 + ResNet50
  * Different architectures capture different features
- **Different model combinations:** Each experiment should try different backbone combinations to find best ensemble
- **For images >50K: Use fine_tuning (enough data to train full networks)**
- **Batch size:** 32-64 for GPU training
- **Train split:** <5000 samples use 0.85, larger use 0.9-0.95
- **Image augmentation:** ONLY RandomHorizontalFlip, RandomRotation, ColorJitter, RandomCrop, RandomResizedCrop
- **DO NOT use:** AutoAugment, RandAugment, Mixup, CutMix, Cutout, RandomErasing, Custom LSTM, BiLSTM, GloVe/FastText

**Step 3: Output ONLY JSON (NO text before/after):**

[
  {{
    "id": "exp_1",
    "strategy": "bottleneck_features",
    "models": ["ResNet50", "InceptionV3"],
    "classifier": "LogisticRegression",
    "features": {{"type": "pretrained_features", "details": "Multi-model ensemble: ResNet50 (2048-dim) + InceptionV3 (2048-dim) = 4096-dim features"}},
    "hyperparameters": {{"device": "cuda", "batch_size": 64, "C": 1.0}},
    "hypothesis": "<why multi-model bottleneck is best for this dataset>"
  }},
  {{
    "id": "exp_2",
    "strategy": "bottleneck_features",
    "model": "EfficientNet-B0",
    "classifier": "LogisticRegression",
    "features": {{"type": "pretrained_features", "details": "Single-model for speed"}},
    "hyperparameters": {{"device": "cuda", "batch_size": 64}},
    "hypothesis": "<fast baseline to compare against multi-model>"
  }},
  {{
    "id": "exp_3",
    "strategy": "fine_tuning",
    "model": "DenseNet161",
    "features": {{"type": "image", "details": "Fine-tune to see if training helps"}},
    "hyperparameters": {{"device": "cuda", "epochs": 12, "lr": 0.0001, "batch_size": 32}},
    "hypothesis": "<why fine-tuning might work>"
  }}
]

**Important:** 
- Always include "strategy" field in each experiment
- For bottleneck_features: Use "models" (array) for multi-model OR "model" (string) for single
- **Multi-model bottleneck recommended for best performance** (proven in gold-medal solutions)
- Round 1: Try multi-model bottleneck + comparison experiments. Round 2+: Double down on what worked."""


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
4. **Check strategy from spec** (spec['strategy'] or default to 'fine_tuning')
5. Write train.py based on strategy:

**STRATEGY: "bottleneck_features"** (extract features, train simple classifier):
```python
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import numpy as np
import joblib

# Check if single model or multi-model ensemble
if 'models' in spec and isinstance(spec['models'], list):
    model_names = spec['models']  # Multi-model ensemble
else:
    model_names = [spec['model']]  # Single model

print(f"Using {{len(model_names)}} backbone(s): {{model_names}}")

# Load all backbones
backbones = []
for model_name in model_names:
    backbone = getattr(torchvision.models, model_name.lower().replace('-', '_'))(pretrained=True)
    if hasattr(backbone, 'fc'):
        backbone.fc = nn.Identity()
    elif hasattr(backbone, 'classifier'):
        backbone.classifier = nn.Identity()  
    elif hasattr(backbone, 'head'):
        backbone.head = nn.Identity()
    backbone.eval().to(device)
    backbones.append(backbone)
    print(f"Loaded {{model_name}}")

# Extract features from all backbones
def extract_features(loader, backbones, device):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            batch_features = []
            for backbone in backbones:
                feats = backbone(images).cpu().numpy()
                batch_features.append(feats)
            concat_feats = np.hstack(batch_features)  # Concatenate features from all models
            all_features.append(concat_feats)
            all_labels.extend(labels.numpy())
    
    return np.vstack(all_features), np.array(all_labels)

print("Extracting training features...")
X_train, y_train = extract_features(train_loader, backbones, device)
print(f"Training features shape: {{X_train.shape}}")

print("Extracting validation features...")
X_val, y_val = extract_features(val_loader, backbones, device)
print(f"Validation features shape: {{X_val.shape}}")

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train LogisticRegression
print("Training LogisticRegression...")
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# Validation
val_probs = clf.predict_proba(X_val)
val_metric = log_loss(y_val, val_probs)
val_acc = (clf.predict(X_val) == y_val).mean()
print(f"VALIDATION_SCORE: {{val_metric:.6f}}")
print(f"Validation Accuracy: {{val_acc:.4f}}")

# Save models
for i, backbone in enumerate(backbones):
    torch.save(backbone.state_dict(), f'backbone_{{i}}.pth')
joblib.dump({{'model_names': model_names}}, 'model_config.pkl')
joblib.dump(clf, 'classifier.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Models saved!")
```

**STRATEGY: "fine_tuning"** (standard CNN training):
   - **For image datasets:** Read CSV with dtype={{'id': str}} to preserve filename format
   - GPU memory cleanup: `import torch; torch.cuda.empty_cache()` at start
   - Correct data loading based on structure you found
   - Load pretrained model, replace final layer with num_classes
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
     print(f"VALIDATION_SCORE: {{val_metric:.6f}}")
     ```
     * Use the EXACT competition metric from EDA context (e.g., logloss, AUC, accuracy, etc.)
     * Format must be exactly "VALIDATION_SCORE: " followed by the number
     * Example: "VALIDATION_SCORE: 0.623456" or "VALIDATION_SCORE: 0.954321"
     * All experiments MUST report the same metric for fair comparison
     * Do NOT print other metrics on lines containing "VALIDATION_SCORE"
   - **Save model appropriately:**
     * bottleneck_features: Save backbone + classifier + scaler
     * fine_tuning: Save `torch.save(model.state_dict(), 'model.pth')`
   - **DO NOT generate test predictions in train.py - that's done separately in submission phase**
6. Respond "READY" immediately

**FALLBACK:** If strategy is unclear or missing, default to "fine_tuning" (standard approach)

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
2. Is there a **fundamentally different architecture** worth trying?
3. Have we exhausted promising approaches?
4. **Time efficiency: Is continuing worth the time investment?**

**CRITICAL - What counts as "worth trying":**
- ✅ **CONTINUE only if:** 
  * Completely different model family (e.g., CNN → Transformer, DenseNet → XGBoost)
  * OR different strategy (e.g., fine_tuning → bottleneck_features, or vice versa)
- ❌ **DO NOT continue for:** More epochs, different learning rate, minor hyperparameter tweaks, same architecture with variations
- **Goal: Get a good solution FAST, not perfect. Competition rewards speed.**

Remember metric direction when evaluating score quality:
- LOWER is better: smaller scores are better (e.g., logloss 0.1 > logloss 1.0)
- HIGHER is better: larger scores are better (e.g., AUC 0.99 > AUC 0.9)

**Time-based criteria (competition efficiency):**
- **If cumulative time >= 30 min: SUBMIT immediately (hard stop)**
- If cumulative time 20-30 min: SUBMIT unless CLEAR evidence of >1% improvement from fundamentally different architecture
- If cumulative time 10-20 min: Be conservative, SUBMIT unless untried architecture family exists
- If cumulative time <10 min: SUBMIT if score is decent, CONTINUE only for major architecture changes

Output format (no other text):

DECISION: SUBMIT
BEST_MODEL: exp_2
REASONING: Best logloss 0.62 is competitive, no clear path to major improvement

Criteria:
- SUBMIT if: Score is decent AND (no fundamentally different architecture OR round >= 3 OR time-constrained)
- CONTINUE if: Untried architecture family exists AND strong hypothesis for >1% improvement AND time-efficient AND cumulative time <30 min

**Remember: Speed > Perfection. Minor tweaks waste time. SUBMIT early and often.**"""


SUBMISSION_PROMPT = """Create submission.csv. Fast.

Best model: {best_model} at {best_workspace}
Data: {data_dir}
Output: {submission_dir}/submission.csv

**CRITICAL: Predict ONLY on test data, NOT training data!**

DO:
1. Read {data_dir}/sample_submission.csv to get test IDs (use dtype={{'id': str}} to preserve format)
2. Check what files exist in {best_workspace}/ to determine strategy:
   - If backbone.pth + classifier.pkl + scaler.pkl exist: Use **bottleneck_features** approach
   - If model.pth exists: Use **fine_tuning** approach
3. Write predict.py based on strategy:

**STRATEGY: bottleneck_features**
```python
import joblib
import torch
import torch.nn as nn
import numpy as np

# Load model config to check if single or multi-model
config = joblib.load('{best_workspace}/model_config.pkl')
model_names = config['model_names']
print(f"Loading {{len(model_names)}} backbone(s): {{model_names}}")

# Load all backbones
backbones = []
for i, model_name in enumerate(model_names):
    backbone = getattr(torchvision.models, model_name.lower().replace('-', '_'))(pretrained=True)
    backbone.load_state_dict(torch.load(f'{best_workspace}/backbone_{{i}}.pth'))
    if hasattr(backbone, 'fc'): backbone.fc = nn.Identity()
    elif hasattr(backbone, 'classifier'): backbone.classifier = nn.Identity()
    elif hasattr(backbone, 'head'): backbone.head = nn.Identity()
    backbone.eval().to(device)
    backbones.append(backbone)

# Load classifier and scaler
clf = joblib.load('{best_workspace}/classifier.pkl')
scaler = joblib.load('{best_workspace}/scaler.pkl')

# Extract test features from all backbones
with torch.no_grad():
    test_features = []
    for images in test_loader:
        images = images.to(device)
        batch_features = []
        for backbone in backbones:
            feats = backbone(images).cpu().numpy()
            batch_features.append(feats)
        concat_feats = np.hstack(batch_features)
        test_features.append(concat_feats)
    X_test = np.vstack(test_features)
    X_test = scaler.transform(X_test)

# Predict
predictions = clf.predict_proba(X_test)
```

**STRATEGY: fine_tuning**
```python
# Load model
model.load_state_dict(torch.load('model.pth'))
model.eval().to(device)

# Predict
with torch.no_grad():
    predictions = []
    for images in test_loader:
        preds = model(images.to(device)).cpu()
        predictions.append(preds)
    predictions = torch.cat(predictions).numpy()
```

4. Save to {submission_dir}/submission.csv with EXACT same format/order as sample_submission.csv
5. Run predict.py
6. Verify row count matches sample_submission.csv
7. Respond "DONE"

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

