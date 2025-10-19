EDA_PROMPT = """Analyze competition data. Write ONE eda.py, run ONCE, report findings.

Competition: {competition_id}
Data: {data_dir}
Instructions: {instructions_path}

**Task:**
1. Read instructions
2. Write comprehensive eda.py (data shape, types, target distribution, class balance, file formats)
3. Run it ONCE
4. Report findings (MUST include ALL bullet points below):
   - Data type (tabular/image/text/time-series)
   - Dataset size and shape
   - Target distribution (balanced/imbalanced)
   - Key patterns or characteristics
   - **CRITICAL - Evaluation metric with direction (REQUIRED):**
     * Format: "**Evaluation Metric:** [metric_name] (HIGHER is better)" OR "**Evaluation Metric:** [metric_name] (LOWER is better)"
     * Examples: "**Evaluation Metric:** AUC (HIGHER is better)" or "**Evaluation Metric:** Log Loss (LOWER is better)"
     * This is REQUIRED for planning - do not skip!

NO model suggestions. NO iteration. ONE script, ONE run."""


PLANNING_PROMPT = """You are an expert ML engineer. Analyze the dataset and design 2-3 experiments that will achieve the best performance.

Competition: {competition_id}
Round: {round_num}
Best: {best_score}

{context}

**Your task:** Read the EDA above carefully, reason about the problem, then design 2-3 experiments.

**Step 1: Understand the problem**
- What type of data? (images, tabular, text, audio)
- How much data? (sample size, features, classes)
- What's the evaluation metric? (optimize for that specific metric!)
- What's challenging about this dataset? (class imbalance, small data, high dimensionality)
- What approaches would work best given these characteristics?

**Step 2: Design experiments based on your analysis**

**CRITICAL: ONLY use models from standard libraries (torchvision.models, transformers, sklearn, xgboost, lightgbm)**
**DO NOT propose custom architectures or models implemented from scratch - worker will fail**

**🚨 CRITICAL - MATCH STRATEGY TO DATA TYPE (READ EDA CAREFULLY!):**

**IF EDA says "Tabular" OR "pre-extracted features" OR "CSV with X numerical features":**
→ MUST use "gradient_boosting" strategy ONLY
→ DO NOT use bottleneck_features/fine_tuning (those are for raw images!)
→ Example: leaf-classification has "192 tabular features" → gradient_boosting, NOT image strategies

**IF EDA says "Image" AND describes actual image files (.jpg/.png) AND NO mention of "pre-extracted features":**
→ PRIMARY: Use "fastai_vision" (proven gold-medal approach, 5-10 min, 95-100% accuracy)
→ FALLBACK: Use "bottleneck_features" if need <3 min baseline
→ Load actual image files, use CNN models

**IF EDA says "Text" or "NLP" or "comments/reviews/documents":**
→ Use "transformer_features" (distilbert/roberta) OR "gradient_boosting" with TF-IDF features

**IF EDA says "Audio" or describes .wav/.mp3/.aif files:**
→ Convert to spectrograms, then use "fastai_vision" (treat spectrograms as images)
→ FALLBACK: Use "bottleneck_features" if fastai too slow

**IF EDA says "Seq->Seq" or "text normalization" or "translation":**
→ This is advanced - use "transformer_features" or skip if too complex

**Double-check before proceeding:** Does EDA mention "pre-extracted features" or "CSV columns with features"? 
→ If YES: It's TABULAR (use gradient_boosting), NOT images!

**Available Strategies (choose mix based on EDA context above):**

**Strategy 1: "fastai_vision"** (RECOMMENDED for images, proven gold-medal approach)
- **When to try:** Image classification with <50K samples (PRIMARY strategy for images!)
- **Why it works:** Simple, fast (5-10 min), consistently achieves gold medals
- **How it works:** Fine-tune pretrained CNN using fastai's fit_one_cycle (automatic LR scheduling + heavy augmentation)
- **Models:** models.densenet161, models.resnet50, models.efficientnet_b2, models.resnet34
- **Key advantages:**
  * fit_one_cycle() automatically finds optimal learning rate and schedule
  * Heavy augmentation (flip, rotate, zoom, lighting, warp) prevents overfitting
  * Standard validation split: 15-20% (ensures robust generalization)
  * Image size 128-224 (smaller = faster training, still excellent accuracy)
  * Much simpler code than manual PyTorch (less room for bugs)
- **Performance:** Regularly achieves 95-100% accuracy in 5-10 minutes
- **Example:** {{"strategy": "fastai_vision", "model": "densenet161", "epochs": 5, "size": 128, "lr": 3e-2, "split_pct": 0.15}}

**Strategy 2: "fastai_tabular"** (neural networks for small tabular datasets)
- **When to try:** Tabular data with <10K samples and high dimensionality (features > samples per class)
- **Why it works:** Neural networks with dropout regularization prevent overfitting better than gradient boosting on small data
- **How it works:** Use fastai TabularLearner with automatic normalization and dropout
- **Best for:** Small datasets where gradient boosting overfits (e.g., 891 samples, 99 classes, 192 features)
- **Performance:** Often beats gradient boosting by 50-200% on small tabular data
- **Example:** {{"strategy": "fastai_tabular", "layers": [200, 100], "epochs": 50, "lr": 1e-3}}

**Strategy 3: "bottleneck_features"** (feature extraction + LogisticRegression, fallback for images)
- **When to try:** Images + Classification, especially with <50K samples
- **How it works:** Extract features from pretrained CNN (no training), train LogisticRegression on features
- **Pros:** Very fast (trains in seconds), often better with limited data, low memory
- **Classifier:** ALWAYS use LogisticRegression (DO NOT use XGBoost - too slow, not worth it)
- **🏆 For MAX ACCURACY (gold-medal techniques, still fast):**
  * **ALWAYS use 3-model ensembles** (not 2) - adds 30s, improves score 2-10x
  * **Best 3-model combinations (proven in gold solutions):**
    - EfficientNet-B2 (1408-dim) + DenseNet161 (2208-dim) + ResNet50 (2048-dim) = 5664-dim
    - ResNet50 + InceptionV3 + DenseNet121 = 5120-dim
    - Wide_ResNet50_2 + EfficientNet-B2 + DenseNet161 = 5664-dim
  * **Image size: 299x299** (better than 224x224 for feature quality)
  * **Train split: 0.8-0.85** (15-20% validation for proper generalization)
  * **LogReg tuning: Try C=0.1, 0.5, 1.0** (pick best on val, adds 5s)
  * **Test-time augmentation (TTA):** Predict on original + horizontal flip, average predictions (+10s, +2-5% accuracy)
- **Example:** {{"strategy": "bottleneck_features", "models": ["EfficientNet-B2", "DenseNet161", "ResNet50"], "classifier": "LogisticRegression", "image_size": 299, "tta": true, "train_split": 0.85}}

**Strategy 4: "gradient_boosting"** (for medium/large tabular data)
- **When to try:** Tabular data with numerical/categorical features in CSV format
- **Models:** LightGBM (fast, handles categoricals), XGBoost (robust, tree_method='hist')
- **DO NOT use CatBoost** (parameter conflicts, not worth debugging time)
- **Example:** {{"strategy": "gradient_boosting", "model": "LightGBM", "hyperparameters": {{"n_estimators": 500, "learning_rate": 0.05}}}}

**Strategy 5: "transformer_features"** (for text)
- **When to try:** Text data
- **Models:** distilbert-base-uncased, bert-base-uncased, roberta-base

**Hyperparameter Selection Principles:**

**For validation split:**
- Use 15-20% validation (0.15-0.20 split_pct) for proper generalization estimates
- Smaller datasets may use 20%, larger can use 15%
- NEVER use <10% - validation too small to detect overfitting

**For image size:**
- Consider source image resolution and task complexity
- Smaller images (128-160): Faster training, good for simple patterns
- Larger images (224-256): Better feature learning, needed for fine details
- Balance speed vs accuracy based on time budget

**For learning rate:**
- Standard ranges: 1e-2 to 3e-2 for fastai, 0.01-0.1 for gradient boosting
- Adjust based on batch size and architecture
- Lower for larger models/images

**For epochs:**
- Images with fastai: 5-10 epochs typically sufficient
- Gradient boosting: 500-1000 trees with early stopping
- More epochs if dataset is complex or large

**For batch size:**
- GPU memory constraint: Larger images → smaller batch
- Standard: 32-64 for most image tasks
- Adjust based on image size and model

**Choose models and strategies based on:**
- Dataset characteristics (size, type, complexity)
- Time budget (fastai ~5-10min, bottleneck ~2-3min, gradient boosting varies)
- What's most likely to achieve best performance on the evaluation metric

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


WORKER_PROMPT = """You're a skilled ML engineer. Write train.py for this experiment.

Experiment spec: {spec}
Data location: {data_dir}

EDA context: {eda_context}

**Your task:**
1. Explore {data_dir} using Bash tools (ls, zipinfo, head) to understand structure
2. Write train.py that:
   - Loads data based on what you find (don't assume filenames!)
   - Trains model using spec['strategy'] and ALL spec['hyperparameters']
   - Calculates competition metric from EDA context (search for "Evaluation Metric:")
   - Prints required output (see below)
   - Exits

**Required output format:**
```
VALIDATION_SCORE: {{primary_metric}}
VAL_LOSS: {{val_loss}}
TRAIN_LOSS: {{train_loss}}  
TRAIN_TIME: {{seconds}}
```

**Key rules:**
- Explore data structure FIRST - use ls, zipinfo, head to see what files exist
- Don't hardcode paths/filenames - adapt to what you find
- Use ALL hyperparameters from spec (epochs, lr, size, batch_size, split_pct, etc.)
- Import sklearn.metrics at top of train.py
- Convert tensors to float before printing (.item() or .numpy())
- Don't save models or generate test predictions in train.py

**Common strategies:**
- fastai_vision: Fine-tune with fit_one_cycle
- bottleneck_features: Extract features, train LogReg
- gradient_boosting: LightGBM/XGBoost
- transformer_features: Extract embeddings, train LogReg

Respond "READY" when done.

Tools: Bash, Read, Write"""


ANALYSIS_PROMPT = """Analyze results. Output decision.

**Metric: {metric_direction}**
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import numpy as np
import joblib

# Load models
if 'models' in spec and isinstance(spec['models'], list):
    model_names = spec['models']
else:
    model_names = [spec['model']]

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

# Standardize features (CRITICAL for LogReg performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train LogisticRegression (try multiple C if in spec, else use default)
print("Training LogisticRegression...")
C_value = spec.get('C', 1.0)
clf = LogisticRegression(C=C_value, multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# Validation - CALCULATE THE COMPETITION METRIC FROM EDA!
# Metrics already imported at top of file:
# from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, mean_squared_error, mean_absolute_error
val_probs = clf.predict_proba(X_val)
val_probs = np.clip(val_probs, 1e-7, 1 - 1e-7)

# Use the EXACT competition metric from EDA context:
# Check EDA for "Evaluation Metric: XXX (HIGHER/LOWER is better)"
# Example: If EDA says "AUC-ROC (HIGHER is better)", use roc_auc_score
if "AUC" in eda_metric:
    val_metric = roc_auc_score(y_val, val_probs[:, 1])  # binary AUC
elif "Accuracy" in eda_metric:
    val_metric = accuracy_score(y_val, val_probs.argmax(axis=1))
else:  # Log Loss or default
    num_classes = len(clf.classes_)
    val_metric = log_loss(y_val, val_probs, labels=list(range(num_classes)))

print(f"VALIDATION_SCORE: {{val_metric:.6f}}")

# Save models
for i, backbone in enumerate(backbones):
    torch.save(backbone.state_dict(), f'backbone_{{i}}.pth')
joblib.dump({{'model_names': model_names}}, 'model_config.pkl')
joblib.dump(clf, 'classifier.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Models saved!")
```

**Key techniques for better accuracy (adapt as needed):**
- Image size: Larger (299x299) > smaller (224x224) for feature quality
- Train split: 0.95 (more data) > 0.85 for small datasets (<10K samples)
- TTA: If spec['tta']=true, extract features on original + HorizontalFlip, average features
- C tuning: Try C=[0.1, 0.5, 1.0], pick best on validation (quick cross-validation)
- Ensure StandardScaler is applied to all feature sets (train/val/test)

**STRATEGY: "gradient_boosting"** (for tabular):
   1. Load CSV: `train_df = pd.read_csv('/home/data/train.csv')`
   2. Identify target, separate X and y
   3. Encode categoricals if needed
   4. Train/val split: 15-20% validation
   5. Train using **spec['model']** and **spec['hyperparameters']**: `LGBMClassifier(**spec['hyperparameters'])`
   6. Calculate competition metric from EDA, print VALIDATION_SCORE
   7. Exit (don't save model)

**STRATEGY: "transformer_features"** (for text):
   1. Load text data from CSV
   2. Load tokenizer and model from **spec['model']** (e.g., 'distilbert-base-uncased', 'roberta-base')
   3. Tokenize text: `tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')`
   4. Extract embeddings (CLS token or mean pooling)
   5. Train/val split: 15-20% validation
   6. Train LogisticRegression on embeddings using **spec hyperparameters**
   7. Calculate competition metric from EDA (AUC/Accuracy for classification)
   8. Print VALIDATION_SCORE
   9. Exit (don't save model)

**STRATEGY: "fine_tuning"** (for images):
   - Similar to fastai_vision but uses manual PyTorch training loop
   - Extract data, adapt to zip structure (same as fastai_vision)
   - Use **spec hyperparameters** (epochs, lr, batch_size)
   - Calculate competition metric
   - Exit (don't save model)

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
   - Early stopping with patience 3-5 epochs
   - For perfect score termination: if metric is AUC/accuracy (higher is better), stop at val_metric >= 0.9999; if logloss/error (lower is better), stop at val_metric <= 0.001
   - **🚨 CRITICAL: Calculate and print THE COMPETITION METRIC, NOT LOSS! 🚨**
     
     **Step 1: READ THE EVALUATION METRIC FROM EDA CONTEXT ABOVE**
     Look for lines like "Evaluation Metric: XXX (HIGHER/LOWER is better)"
     Examples:
     - "Evaluation Metric: AUC-ROC (HIGHER is better)" → Calculate AUC-ROC score
     - "Evaluation Metric: Log Loss (LOWER is better)" → Calculate Log Loss
     - "Evaluation Metric: Accuracy (HIGHER is better)" → Calculate Accuracy
     - "Evaluation Metric: RMSE (LOWER is better)" → Calculate RMSE
     
     **Step 2: CALCULATE THAT EXACT METRIC ON VALIDATION SET**
     ```python
     # Metrics already imported at top of file!
     # from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, mean_squared_error
     
     # Get predictions on validation set
     val_probs = model.predict_proba(X_val)  # or learn.get_preds(dl=val_dl)[0]
     val_probs = np.clip(val_probs, 1e-7, 1 - 1e-7)  # prevent log(0) errors
     
     # Calculate THE COMPETITION METRIC (match what's in EDA):
     if competition_metric == "AUC" or "ROC":
         val_metric = roc_auc_score(y_val, val_probs[:, 1])  # binary
         # OR for multiclass: roc_auc_score(y_val, val_probs, multi_class='ovr', average='macro')
     elif competition_metric == "Log Loss" or "Logloss":
         val_metric = log_loss(y_val, val_probs, labels=list(range(num_classes)))
     elif competition_metric == "Accuracy":
         val_metric = accuracy_score(y_val, val_probs.argmax(axis=1))
     elif competition_metric == "RMSE":
         val_metric = mean_squared_error(y_val, val_preds, squared=False)
     # ... add other metrics as needed
     
     # ⚠️ DO NOT USE: val_metric = loss  # WRONG! Loss ≠ competition metric
     # ⚠️ DO NOT USE: val_metric = learn.recorder.final_record[0]  # That's loss!
     ```
     
     **Step 3: PRINT IN EXACT FORMAT**
     ```python
     if isinstance(val_metric, torch.Tensor):
         val_metric = val_metric.item()
     print(f"VALIDATION_SCORE: {{val_metric:.6f}}")
     ```
     
     **EXAMPLES OF CORRECT VALIDATION_SCORE:**
     - AUC-ROC competition: `VALIDATION_SCORE: 0.9876` (not 0.0124 which is loss!)
     - Accuracy competition: `VALIDATION_SCORE: 0.9500` (not 0.0500 which is loss!)
     - Log Loss competition: `VALIDATION_SCORE: 0.1234` (log loss itself, OK)
     - RMSE competition: `VALIDATION_SCORE: 2.3456` (RMSE itself, OK)
     
     **Common mistakes to AVOID:**
     - ❌ Printing training loss instead of competition metric
     - ❌ Printing wrong metric (accuracy when competition uses AUC)
     - ❌ For AUC/accuracy competitions: printing loss (0.01) instead of score (0.99)
     - ✅ ALWAYS calculate the exact metric mentioned in EDA, not training loss!
   - **Save model appropriately:**
     * bottleneck_features: Save backbone + classifier + scaler
     * fine_tuning: Save `torch.save(model.state_dict(), 'model.pth')`
   - **DO NOT generate test predictions in train.py - that's done separately in submission phase**
6. Respond "READY" immediately

**FALLBACK:** If strategy is unclear or missing, default to "fine_tuning" (standard approach)

Tools: Bash, Read, Write"""


ANALYSIS_PROMPT = """Analyze results. Output decision.

**Metric: {metric_direction}**
This round's results: {results}
**🏆 CUMULATIVE BEST (across all rounds): {best_experiment_id} with score {best_score}**
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

Most of the time you should SUBMIT!!!! Unless there is a critically different architecture that is worth trying based on the results of the previous round.

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
BEST_MODEL: r1_exp_3
REASONING: Cumulative best score 0.2509 from r1_exp_3 is competitive, this round's experiments didn't improve, no fundamentally different architecture justifies more time

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
   - If learner.pkl exists: Use **fastai** approach (vision or tabular)
   - Elif model.pkl exists: Use **gradient_boosting** approach (tabular data)
   - Elif backbone_0.pth + classifier.pkl exist: Use **bottleneck_features** approach
   - Elif model.pth exists: Use **manual pytorch** approach
3. Write predict.py based on strategy:

**STRATEGY: fastai** (vision or tabular)
```python
from fastai.vision.all import *

learn = load_learner('{best_workspace}/learner.pkl')

test_dl = learn.dls.test_dl(test_items)
preds, _ = learn.get_preds(dl=test_dl)

predictions = preds.numpy()
```

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

# Extract test features
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

# Predict (use predict_proba for classification)
predictions = clf.predict_proba(X_test)
```

**For better accuracy:** If training used TTA (test-time augmentation), apply same during prediction:
- Extract features on original + flipped images, average before predict_proba
- Improves calibration and reduces logloss by 2-5%

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

**STRATEGY: gradient_boosting** (tabular data)
```python
import joblib
model = joblib.load('{best_workspace}/model.pkl')

# Load test CSV, drop ID columns, apply same preprocessing as training
# Use predict_proba() for classification, predict() for regression
predictions = model.predict_proba(X_test)
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


def format_analysis_prompt(competition_id: str, round_num: int, results: str, best_score: float, best_experiment_id: str, metric_direction: str, round_time_minutes: float, cumulative_time_minutes: float) -> str:
    return ANALYSIS_PROMPT.format(
        round_num=round_num,
        results=results,
        best_score=best_score,
        best_experiment_id=best_experiment_id,
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

