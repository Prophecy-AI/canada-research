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


WORKER_PROMPT = """Write train.py for this experiment. DO NOT RUN IT.

Experiment: {spec}
Data: {data_dir}
EDA: {eda_context}

**CRITICAL: Your ONLY job is to write train.py. DO NOT:**
- Run train.py (orchestrator will run it)
- Write summaries/documentation
- Test imports
- Create verification scripts

**DO:**
1. Check data structure (use Bash: ls, zipinfo, head CSV)
2. Extract zip files to workspace if needed (unzip -q /home/data/train.zip -d .)
3. Write train.py with:
   - Correct data loading based on structure you found
   - Model/features/hyperparameters from spec
   - train_test_split (test_size=0.2, random_state=42)
   - GPU training
   - Print validation score
   - Save model
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

