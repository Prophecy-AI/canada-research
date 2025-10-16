EDA_PROMPT = """ML engineer: Analyze data FAST. Minimize output.

Competition: {competition_id}
Data: {data_dir}
Instructions: {instructions_path}

**Task:** Read instructions, write+run eda.py. Output findings in <3 sentences total.

**Output format:**
Data: [type, shape]
Models: [1-2 GPU models]
Why: [1 sentence]

GPU models: XGBoost (tree_method='gpu_hist'), PyTorch"""


PLANNING_PROMPT = """ML strategist: Design 1-3 experiments. Prefer FEWER (use 1 if confident).

Competition: {competition_id}
EDA: {context}
Round: {round_num}
Best: {best_score}

**Task:** Output JSON (1-3 experiments). Use 1 if confident, 2-3 if testing hypotheses.

```json
[
  {{
    "id": "exp_1",
    "model": "XGBoost",
    "features": {{"type": "...", "details": "..."}},
    "hyperparameters": {{"tree_method": "gpu_hist", "device": "cuda", ...}},
    "hypothesis": "1 sentence"
  }}
]
```

Models: XGBoost (tree_method='gpu_hist'), LightGBM (device='gpu'), CatBoost, RandomForest, LogisticRegression, Ridge"""


WORKER_PROMPT = """Write train.py for ML experiment. NO running, NO exploration - just write the file.

Experiment: {spec}
EDA context: {eda_context}
Data: {data_dir}
Workspace: {workspace_dir}

**Requirements:**
- Load data from {data_dir} (train.zip, test.zip, train.csv, sample_submission.csv)
- Implement model/features/hyperparameters from spec
- 80/20 train/val split (random_state=42)
- Print "VALIDATION_SCORE: 0.847" (exact format)
- Save model.pkl
- GPU: device='cuda', tree_method='gpu_hist'
- Handle errors with try/except

**For IMAGE data (zip files):**
```python
from zipfile import ZipFile
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('{data_dir}/train.csv')

def load_images_from_zip(zip_path, image_ids):
    images = []
    with ZipFile(zip_path, 'r') as z:
        for img_id in image_ids:
            img_data = z.read(f'train/{{img_id}}')
            img = Image.open(io.BytesIO(img_data))
            images.append(np.array(img))
    return np.array(images)

X = load_images_from_zip('{data_dir}/train.zip', df['id'])
y = df['has_cactus'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

**For TABULAR data (csv):**
```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('{data_dir}/train.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

Write train.py then respond "READY".

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

