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

**🚨 CRITICAL - MATCH STRATEGY TO DATA TYPE (READ EDA CAREFULLY!):**

**IF EDA says "Tabular" OR "pre-extracted features" OR "CSV with X numerical features":**
→ Check dataset size:
  * **If <10K samples AND (features > 100 OR features/samples_per_class > 10):**
    → PRIMARY: Use "fastai_tabular" (neural networks prevent overfitting better than trees)
    → SECONDARY: Use "gradient_boosting" for comparison
    → Example: leaf-classification (891 samples, 192 features, 99 classes = 9 per class, ratio 192/9=21) → fastai_tabular PRIMARY
  * **If >=10K samples:**
    → PRIMARY: Use "gradient_boosting" for comparison
→ DO NOT use bottleneck_features/fine_tuning (those are for raw images!)

**IF EDA says "Image" AND describes actual image files (.jpg/.png) AND NO mention of "pre-extracted features":**
→ CRITICAL: ALWAYS TRY Keras Neural Network first. KNNs are the novel solution for any image classification.
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
  * 99/1 train/val split (maximum training data, minimal validation)
  * Image size 128-224 (smaller = faster training, still excellent accuracy)
  * Much simpler code than manual PyTorch (less room for bugs)
- **Performance:** Regularly achieves 95-100% accuracy in 5-10 minutes
- **Example:** {{"strategy": "fastai_vision", "model": "densenet161", "epochs": 5, "size": 128, "lr": 3e-2, "split_pct": 0.01}}

**Strategy 2: "fastai_tabular"** (Keras neural networks for small tabular datasets)
- **When to try:** Tabular data with <10K samples and high dimensionality (features/samples_per_class > 10)
- **Why it works:** Neural networks with dropout prevent overfitting better than gradient boosting on small high-dimensional data
- **How it works:** Keras Sequential model with large layers [1024, 512] or [1500, 1500], Dropout(0.1-0.2), EarlyStopping
- **Best for:** Small datasets where gradient boosting overfits (e.g., 891 samples, 192 features, 99 classes = ratio 21)
- **Performance:** Often beats gradient boosting by 2-10x on small tabular data (proven with gold solutions)
- **Hyperparameters:**
  * layers: Large networks work better, e.g., [1024, 512], [1500, 1500], [2048, 1024]
  * epochs: 100-800 (use EarlyStopping patience=50-100 to prevent overtraining)
  * lr: Learning rate 1e-3 to 1e-2, optimizer 'adam' or 'rmsprop'
  * batch_size: Large batches (128-192) act as regularization
- **Example:** {{"strategy": "fastai_tabular", "hyperparameters": {{"layers": [1024, 512], "epochs": 200, "lr": 1e-3, "batch_size": 128}}}}

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
  * **Train split: 0.95** (more data for LogReg = better calibration)
  * **LogReg tuning: Try C=0.1, 0.5, 1.0** (pick best on val, adds 5s)
  * **Test-time augmentation (TTA):** Predict on original + horizontal flip, average predictions (+10s, +2-5% accuracy)
- **Example:** {{"strategy": "bottleneck_features", "models": ["EfficientNet-B2", "DenseNet161", "ResNet50"], "classifier": "LogisticRegression", "image_size": 299, "tta": true, "train_split": 0.95}}

**Strategy 4: "keras_neural_network"** (general-purpose deep learning)
- **When to try:** 
  * Tabular data where neural networks might outperform trees (complex patterns, high dimensionality)
  * Custom architectures for specialized tasks
  * When you need fine control over architecture/training
- **Why it works:** Flexible, can model complex non-linear relationships, dropout prevents overfitting
- **Architecture guidelines:**
  * Small data (<5K): [512, 256] with Dropout(0.3-0.5), batch_size=32-64
  * Medium data (5K-50K): [1024, 512, 256] with Dropout(0.2-0.3), batch_size=64-128
  * Large data (>50K): [2048, 1024, 512] with Dropout(0.1-0.2), batch_size=128-256
  * Very high dim features: Go wider [1500, 1500] or [2048, 1024]
- **Training best practices:**
  * Activation: 'relu' for hidden layers, 'softmax'/'sigmoid' for output
  * Optimizer: 'adam' (lr=1e-3 to 1e-4) or 'rmsprop' (lr=1e-3)
  * Loss: 'categorical_crossentropy' (multiclass), 'binary_crossentropy' (binary), 'mse' (regression)
  * Regularization: Dropout + EarlyStopping(patience=15-30, restore_best_weights=True)
  * Batch normalization: Add after Dense layers for large networks (optional)
- **Data preprocessing:**
  * Features: StandardScaler or MinMaxScaler (REQUIRED)
  * Target: LabelEncoder → to_categorical for classification
  * Missing values: SimpleImputer (mean/median for numeric, most_frequent for categorical)
- **Hyperparameters:**
  * epochs: 100-300 (let EarlyStopping decide)
  * batch_size: 64-256 depending on data size
  * validation_split: 0.1-0.2
- **Example:** {{"strategy": "keras_neural_network", "architecture": [1024, 512, 256], "dropout": 0.2, "epochs": 200, "batch_size": 128, "lr": 1e-3, "optimizer": "adam"}}

**Strategy 5: "gradient_boosting"** (for medium/large tabular data)
- **When to try:** Tabular data with numerical/categorical features in CSV format
- **Models:** LightGBM (fast, handles categoricals), XGBoost (robust, tree_method='hist')
- **DO NOT use CatBoost** (parameter conflicts, not worth debugging time)
- **Example:** {{"strategy": "gradient_boosting", "model": "LightGBM", "hyperparameters": {{"n_estimators": 500, "learning_rate": 0.05}}}}

**Strategy 6: "transformer_features"** (for text)
- **When to try:** Text data
- **Models:** distilbert-base-uncased, bert-base-uncased, roberta-base

**Recommendation for Round 1:**
- **Images <50K (raw image files - PRIMARY STRATEGY):** 
  * **🏆 USE FASTAI FIRST** - proven gold-medal approach, simple, fast
  * exp_1: fastai_vision (densenet161, epochs=5, size=128, lr=3e-2, split_pct=0.01)
  * exp_2: fastai_vision (resnet50, epochs=5, size=128, lr=3e-2, split_pct=0.01)
  * exp_3: fastai_vision (efficientnet_b2, epochs=7, size=160, lr=2e-2, split_pct=0.01)
  * Each finishes in 5-10 min, explores different architectures
  * **ONLY use bottleneck_features if fastai fails or for very fast baseline (<3 min)**
- **Tabular <10K samples (high dimensionality, e.g., features/samples_per_class >10):**
  * exp_1: fastai_tabular (layers=[1024, 512], epochs=100, lr=1e-3) - neural network prevents overfitting
  * exp_2: gradient_boosting (LightGBM, conservative regularization)
  * exp_3: fastai_tabular (layers=[1500, 1500], epochs=200, lr=5e-4) - larger network for comparison
  * Neural networks often 2-10x better than gradient boosting on small high-dim data
- **Tabular >10K samples:**
  * exp_1: LightGBM with feature engineering
  * exp_2: XGBoost with different hyperparameters
  * exp_3: LightGBM with different feature combinations
  * **DO NOT use image/text strategies on tabular data!**
- **Text data:** 
  * exp_1: transformer_features (distilbert)
  * exp_2: transformer_features (roberta)
  * exp_3: gradient_boosting with TF-IDF features
- **Audio:** 
  * Convert to spectrograms
  * exp_1: fastai_vision (treat spectrograms as images)
  * exp_2: bottleneck_features (if fastai too slow)

**Experiment Design Guidelines:**
- **For images <50K: PREFER fastai_vision (gold-medal approach), bottleneck_features as fast fallback**
- **Multi-model selection:** Choose 2-3 complementary backbones with different architectures:
  * Good pairs: ResNet50 + InceptionV3, DenseNet161 + Wide_ResNet50_2, EfficientNet-B2 + ResNet50
  * Good triplets: EfficientNet-B2 + DenseNet121 + ResNet50
  * Different architectures capture different features
- **Classifier for bottleneck: ALWAYS LogisticRegression** (DO NOT use XGBoost/tree methods - too slow for time budget)
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
4. **Check strategy from spec** (spec['strategy'])
5. Write train.py based on strategy:

**CRITICAL FOR ALL FASTAI STRATEGIES:**
- **MUST import at top of file BEFORE using any fastai functions:** 
  * For images: `from fastai import *` then `from fastai.vision import *` (two lines)
  * For tabular: `from fastai.tabular.all import *`
- These imports include ALL needed functions (get_transforms, ImageList, cnn_learner, tabular_learner, etc.)
- **DO NOT** import individual functions - use the `import *` pattern
- **API version:** Use the v1-style API (ImageList, cnn_learner) - proven to work with our fastai version and gets gold medals

**STRATEGY: "fastai_vision"** (RECOMMENDED for images, gold-medal approach):
   - **CRITICAL: Import at top:** `from fastai import *` then `from fastai.vision import *`
   - **Use proven pattern from gold solutions (gets perfect scores):**
     * Load images: `ImageList.from_df(df, path=data_path, folder='train')` or `.from_folder()`
     * Split: `.split_by_rand_pct(0.01)` for 99/1 train/val (maximum training data!)
     * Labels: `.label_from_df()` if using CSV, or `.label_from_folder()` if folder structure
     * Add test: `.add_test(test_images)` where test_images = ImageList.from_df(test_df, ...)
     * Transform: `.transform(tfms, size=128)` where tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=10, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)
     * Create databunch: `.databunch(bs=64).normalize(imagenet_stats)`
     * Create learner: `learn = cnn_learner(data, models.densenet161, metrics=[error_rate, accuracy])`
     * Train: `learn.fit_one_cycle(epochs, slice(lr))` - epochs from spec (default 5), lr from spec (default 3e-2)
     * Predict: `preds, _ = learn.get_preds(ds_type=DatasetType.Test)`
   - Image size from spec (default 128, use 128-224)
   - Print VALIDATION_SCORE with final validation accuracy (1 - error_rate)
   - Save test predictions to submission.csv in correct format
   - Runtime: 5-10 minutes, achieves 95-100% accuracy consistently

**STRATEGY: "fastai_tabular"** (neural networks for small tabular):
   - **Use Keras/TensorFlow approach (RECOMMENDED for gold scores on small tabular):**
     * Import: `from tensorflow import keras` and `from sklearn.preprocessing import StandardScaler, LabelEncoder`
     * Preprocess: StandardScaler on features, LabelEncoder then to_categorical on target
     * Build model: Use large layers [1024, 512] or [1500, 1500] with Dropout(0.1-0.2) between layers
     * Architecture: Dense(1024, relu) → Dropout(0.2) → Dense(512, relu) → Dropout(0.1) → Dense(num_classes, softmax)
     * Compile: loss='categorical_crossentropy', optimizer='adam' (or 'rmsprop'), metrics=['accuracy']
     * Train: batch_size=128-192, epochs=200-800, validation_split=0.1-0.15, EarlyStopping(patience=50-100)
     * Predict: model.predict(X_test) gives probabilities
     * Save: model.save('model.h5') or keras.models.save_model()
   - Large networks work better on small data (counterintuitive but proven)
   - Print VALIDATION_SCORE with best validation metric (from history or callbacks)
   - Works when gradient boosting overfits: samples <10K, high features/sample ratio

**STRATEGY: "keras_neural_network"** (general-purpose deep learning):
   - **Complete implementation pattern:**
     ```python
     import pandas as pd
     import numpy as np
     from tensorflow import keras
     from tensorflow.keras import layers, callbacks
     from sklearn.model_selection import train_test_split
     from sklearn.preprocessing import StandardScaler, LabelEncoder
     from sklearn.impute import SimpleImputer
     from sklearn.metrics import log_loss, accuracy_score, mean_squared_error
     
     # 1. Load and preprocess data
     train_df = pd.read_csv(f'{data_dir}/train.csv')
     test_df = pd.read_csv(f'{data_dir}/test.csv')
     
     # Identify target (column in train but not in test)
     target_col = [col for col in train_df.columns if col not in test_df.columns and col.lower() not in ['id']][0]
     
     # Separate features and target
     X = train_df.drop([target_col] + id_cols, axis=1)
     y = train_df[target_col]
     X_test = test_df.drop(id_cols, axis=1)
     
     # 2. Handle missing values
     num_cols = X.select_dtypes(include=['float64', 'int64']).columns
     cat_cols = X.select_dtypes(include=['object']).columns
     
     num_imputer = SimpleImputer(strategy='median')
     X[num_cols] = num_imputer.fit_transform(X[num_cols])
     X_test[num_cols] = num_imputer.transform(X_test[num_cols])
     
     # 3. Encode categorical features
     for col in cat_cols:
         le = LabelEncoder()
         X[col] = le.fit_transform(X[col].astype(str))
         X_test[col] = le.transform(X_test[col].astype(str))
     
     # 4. Scale features (REQUIRED for neural networks)
     scaler = StandardScaler()
     X_scaled = scaler.fit_transform(X)
     X_test_scaled = scaler.transform(X_test)
     
     # 5. Prepare target
     is_classification = len(np.unique(y)) < 50
     if is_classification:
         le_target = LabelEncoder()
         y_encoded = le_target.fit_transform(y)
         num_classes = len(le_target.classes_)
         if num_classes > 2:
             y_cat = keras.utils.to_categorical(y_encoded, num_classes)
             loss = 'categorical_crossentropy'
             output_activation = 'softmax'
         else:
             y_cat = y_encoded
             loss = 'binary_crossentropy'
             output_activation = 'sigmoid'
     else:
         y_cat = y.values
         num_classes = 1
         loss = 'mse'
         output_activation = 'linear'
     
     # 6. Train/val split
     X_train, X_val, y_train, y_val = train_test_split(
         X_scaled, y_cat, test_size=0.15, random_state=42, 
         stratify=y_encoded if is_classification else None
     )
     
     # 7. Build model from spec
     architecture = spec.get('architecture', [1024, 512, 256])
     dropout_rate = spec.get('dropout', 0.2)
     
     model = keras.Sequential()
     model.add(layers.Input(shape=(X_train.shape[1],)))
     
     for units in architecture:
         model.add(layers.Dense(units, activation='relu'))
         model.add(layers.Dropout(dropout_rate))
     
     if num_classes > 2:
         model.add(layers.Dense(num_classes, activation=output_activation))
     else:
         model.add(layers.Dense(1 if num_classes == 1 else num_classes, activation=output_activation))
     
     # 8. Compile
     lr = spec.get('lr', 1e-3)
     optimizer_name = spec.get('optimizer', 'adam')
     if optimizer_name == 'adam':
         optimizer = keras.optimizers.Adam(learning_rate=lr)
     elif optimizer_name == 'rmsprop':
         optimizer = keras.optimizers.RMSprop(learning_rate=lr)
     else:
         optimizer = 'adam'
     
     model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
     
     # 9. Setup callbacks
     patience = spec.get('patience', 20)
     early_stop = callbacks.EarlyStopping(
         monitor='val_loss', patience=patience, 
         restore_best_weights=True, verbose=1
     )
     
     # 10. Train
     batch_size = spec.get('batch_size', 128)
     epochs = spec.get('epochs', 200)
     
     history = model.fit(
         X_train, y_train,
         validation_data=(X_val, y_val),
         batch_size=batch_size,
         epochs=epochs,
         callbacks=[early_stop],
         verbose=1
     )
     
     # 11. Evaluate
     val_preds = model.predict(X_val)
     if is_classification:
         if num_classes > 2:
             val_probs = np.clip(val_preds, 1e-7, 1 - 1e-7)
             val_score = log_loss(y_val, val_probs)
         else:
             val_probs = np.clip(val_preds.flatten(), 1e-7, 1 - 1e-7)
             val_score = log_loss(y_val, val_probs)
     else:
         val_score = mean_squared_error(y_val, val_preds, squared=False)
     
     print(f"VALIDATION_SCORE: {{val_score:.6f}}")
     
     # 12. Save model
     model.save('model.h5')
     import joblib
     joblib.dump(scaler, 'scaler.pkl')
     if is_classification:
         joblib.dump(le_target, 'label_encoder.pkl')
     ```
   - **Key points:**
     * ALWAYS use StandardScaler before training
     * Use architecture from spec, fallback to [1024, 512, 256]
     * EarlyStopping with patience from spec (default 20)
     * Print best validation score (from history or after training)
     * Save model.h5 + scaler.pkl + label_encoder.pkl

**STRATEGY: "bottleneck_features"** (extract features, train LogisticRegression):

**Implementation guidance (adapt based on spec):**
- **Image preprocessing:**
  * Use image_size from spec (default 299 for EfficientNet, 224 for others)
  * Normalize with ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  * Resize larger (299x299) gives better features than smaller (224x224)
- **Multi-model ensembles:**
  * Load all models from spec['models'] (or single spec['model'])
  * Remove final classification layer (fc/classifier/head → Identity)
  * Extract features from each backbone, concatenate (np.hstack)
  * Always use eval() mode, no_grad() for inference
- **Feature normalization:**
  * CRITICAL: Use StandardScaler().fit_transform() on features before LogReg
  * Without scaling, LogReg performs poorly
- **Train/val split:**
  * Use train_split from spec (default 0.95 for small datasets, 0.85 for large)
  * More training data = better LogReg calibration = lower logloss
- **LogisticRegression hyperparameters:**
  * If spec has 'C' values list → try each, pick best on validation
  * If spec has single C → use it (default C=1.0)
  * Use multi_class='multinomial', solver='lbfgs', max_iter=1000
- **Test-time augmentation (TTA) if spec['tta']=true:**
  * Extract features on: original images + horizontal flips
  * Average the feature vectors before feeding to LogReg
  * Adds ~10s but improves accuracy 2-5%
- **Validation metric:**
  * For classification: Use log_loss with clipped probabilities (1e-7, 1-1e-7)
  * Print VALIDATION_SCORE with the logloss value
- **Save:** backbone weights, classifier, scaler, model_names config

**Example code structure:**
```python
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

# Validation
val_probs = clf.predict_proba(X_val)
val_probs = np.clip(val_probs, 1e-7, 1 - 1e-7)  # Prevent log(0)
val_metric = log_loss(y_val, val_probs)
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

**STRATEGY: "gradient_boosting"** (for tabular data):
   - Load train CSV, identify target column (column in train but not in test)
   - Separate features (X) and target (y), drop ID columns
   - Handle categorical features (LabelEncoder for string columns)
   - Determine task type: classification (y.nunique() < 50) vs regression
   - Check class imbalance for classification:
     * If any class has < 2 samples → drop those classes OR use 95/5 train/val split
     * If min_class >= 2 → use stratified split (test_size=0.15, stratify=y)
     * For regression → standard split (no stratify)
   - Train model based on spec['model']:
     * LightGBM: Use LGBMClassifier/LGBMRegressor with hyperparams from spec
     * XGBoost: Use XGBClassifier/XGBRegressor with tree_method='hist'
     * Set appropriate objective: binary/multiclass/regression
   - Validate and print VALIDATION_SCORE based on task:
     * Binary classification → AUC (roc_auc_score)
     * Multiclass → Log Loss (log_loss)
     * Regression → RMSE (mean_squared_error with squared=False)
   - Save model: `joblib.dump(model, 'model.pkl')`

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
   - If learner.pkl exists: Use **fastai** approach (vision)
   - Elif model.h5 exists: Use **keras/tensorflow** approach (fastai_tabular)
   - Elif model.pkl exists: Use **gradient_boosting** approach (tabular data)
   - Elif backbone_0.pth + classifier.pkl exist: Use **bottleneck_features** approach
   - Elif model.pth exists: Use **manual pytorch** approach
3. Write predict.py based on strategy:

**STRATEGY: fastai** (vision)
```python
from fastai import *
from fastai.vision import *

# Load learner  
learn = load_learner('{best_workspace}', 'learner.pkl')

# Get test predictions
preds, _ = learn.get_preds(ds_type=DatasetType.Test)

# predictions.numpy() gives probability matrix
# Save to submission.csv in correct format
```

**STRATEGY: keras/tensorflow** (fastai_tabular or keras_neural_network)
```python
from tensorflow import keras
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessors
model = keras.models.load_model('{best_workspace}/model.h5')
scaler = joblib.load('{best_workspace}/scaler.pkl')

# Load test data
test_df = pd.read_csv(f'{{data_dir}}/test.csv')
sample_sub = pd.read_csv(f'{{data_dir}}/sample_submission.csv')

# Preprocess test data (same as training)
# 1. Drop ID columns
# 2. Handle missing values (same imputer as training)
# 3. Encode categoricals (same encoders as training)
# 4. Scale features
X_test_scaled = scaler.transform(X_test)

# Predict probabilities
predictions = model.predict(X_test_scaled)

# Format submission
if predictions.shape[1] > 1:
    # Multiclass: use probability columns
    for i in range(predictions.shape[1]):
        sample_sub[f'class_{{i}}'] = predictions[:, i]
else:
    # Binary or regression: single column
    sample_sub['target'] = predictions.flatten()

sample_sub.to_csv(f'{{submission_dir}}/submission.csv', index=False)
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

