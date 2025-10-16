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


PLANNING_PROMPT = """Design 1-3 ML experiments based on data analysis. Output ONLY JSON.

Competition: {competition_id}
Round: {round_num}
Best Score: {best_score}

**Data Analysis:**
{context}

**Your task:**
Based on the data characteristics above, select appropriate models and design experiments.

**Available Models:**
- XGBoost (tree_method='gpu_hist', device='cuda') - Fast GPU gradient boosting
- LightGBM (device='gpu') - Memory-efficient GPU gradient boosting  
- CatBoost (task_type='GPU') - Handles categorical features well
- RandomForest - Good for tabular data
- LogisticRegression - Fast baseline for binary classification
- Ridge - Fast baseline for regression

**For IMAGE data, you can also use PyTorch pretrained models:**
- ResNet18/ResNet50 (torchvision.models.resnet18(pretrained=True))
- EfficientNet-B0 (torchvision.models.efficientnet_b0(pretrained=True))
- MobileNetV2 (torchvision.models.mobilenet_v2(pretrained=True))
- Fine-tune on GPU with model.cuda(), use data augmentation
- Example: "model": "ResNet18", "features": {{"type": "pretrained_cnn", "pretrained": true}}

**DO NOT use tools. DO NOT explore. Output ONLY this JSON:**

[
  {{
    "id": "exp_1",
    "model": "XGBoost",
    "features": {{"type": "raw_pixels", "details": "Flatten and normalize"}},
    "hyperparameters": {{"tree_method": "gpu_hist", "device": "cuda", "n_estimators": 500}},
    "hypothesis": "Why this model/features will work for this data"
  }},
  {{
    "id": "exp_2",
    "model": "ResNet18",
    "features": {{"type": "pretrained_cnn", "pretrained": true, "fine_tune_layers": 2}},
    "hyperparameters": {{"device": "cuda", "epochs": 20, "lr": 0.001, "batch_size": 128}},
    "hypothesis": "Pretrained ImageNet features transfer well to this visual task"
  }}
]

Output 1 experiment if confident in approach, 2-3 if testing different hypotheses."""


WORKER_PROMPT = """Write train.py implementing this experiment. NO exploration, NO iteration.

Experiment Spec: {spec}
Data Context: {eda_context}
Data Directory: {data_dir}
Workspace: {workspace_dir}

**Requirements:**
1. **Import all needed modules** (os, io, numpy, pandas, zipfile, PIL, sklearn, pickle, torch if using CNN)
2. **Load data** from {data_dir} (images in train.zip, labels in train.csv)
3. **Implement model** with exact features/hyperparameters from spec
4. **80/20 train/val split** (random_state=42, stratify=y)
5. **Print score** in exact format: "VALIDATION_SCORE: 0.847123"
6. **Save model** as model.pkl
7. **Wrap in try/except** with error printing

**For IMAGE data from zip files:**
- Use ZipFile to read images: `z.read(f'train/{{img_id}}')`
- Use PIL to open: `Image.open(io.BytesIO(img_bytes))`
- For XGBoost/LightGBM: Flatten images to vectors, engineer features if specified
- For CNNs (ResNet/EfficientNet): Create PyTorch Dataset, DataLoader, use transforms

**For PRETRAINED models (ResNet18/EfficientNet/MobileNet):**
- Load with torchvision.models: `models.resnet18(pretrained=True)`
- Replace final layer for binary classification: `model.fc = nn.Linear(..., 1)`
- Use model.cuda(), train on GPU
- Resize images to 224x224, normalize with ImageNet stats
- Use BCEWithLogitsLoss, Adam optimizer

**GPU Settings:**
- XGBoost: tree_method='gpu_hist', device='cuda'
- LightGBM: device='gpu'
- PyTorch: model.cuda(), images.cuda()

Implement experiment from spec. Respond "READY".

Tools: Write"""


ANALYSIS_PROMPT = """You are an expert ML engineer analyzing experiment results and deciding next steps.

Competition: {competition_id}
Round: {round_num}
Target metric: {metric}

Experiment results:
{results}

Previous best score: {best_score}

Your task: Analyze results and decide next action.

Analysis checklist:
- Which experiment won and why?
- Were hypotheses confirmed or rejected?
- What patterns emerged?
- Is there clear path to >0.5% improvement?

Output format:
DECISION: [SUBMIT or CONTINUE]
REASONING: [Detailed analysis of results and why this decision]
BEST_MODEL: [Which experiment/model won]
IMPROVEMENT: [Absolute improvement from previous best]
NEXT_STRATEGY: [If CONTINUE, specific strategy for next round]

Decision criteria:
- SUBMIT if: Best score > {submit_threshold} OR improvement < 0.005 (0.5%) from previous round
- CONTINUE if: Clear hypothesis for >0.5% improvement exists

If CONTINUE:
- If an algorithm worked exceptionally well, iterate on it (tune hyperparameters, improve features)
- Don't abandon working approaches for untested ones
- Focus on ONE clear hypothesis for next round"""


SUBMISSION_PROMPT = """You are creating the final Kaggle submission. Format precision is critical.

Competition: {competition_id}
Best model: {best_model}
Best model workspace: {best_workspace}
Test data: {data_dir}/test.csv
Sample submission: {data_dir}/sample_submission.csv
Output: {submission_dir}/submission.csv

Your task:
1. Read sample_submission.csv to understand exact format
2. Write predict.py that:
   - Loads trained model from {best_workspace}/model.pkl
   - Loads and preprocesses test data (SAME preprocessing as training)
   - Generates predictions
   - Creates DataFrame matching sample_submission format EXACTLY
   - Saves to {submission_dir}/submission.csv
3. Run predict.py
4. Verify submission.csv matches sample format

Critical requirements:
- EXACT column names from sample_submission.csv
- EXACT row count from sample_submission.csv  
- EXACT data types (int/float/string)
- Same preprocessing as training (features, scaling, encoding)
- No extra columns, no missing rows
- Use GPU for inference if model supports it (faster predictions)

Double-check format before finishing."""


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

