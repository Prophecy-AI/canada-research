EDA_PROMPT = """You are an expert machine learning engineer analyzing Kaggle competition data. Optimize for speed while maintaining insight quality.

Competition: {competition_id}
Data directory: {data_dir}
Instructions file: {instructions_path}

Your task:
1. Read the competition instructions at {instructions_path}
2. Find all data files (train.csv, test.csv, sample_submission.csv)
3. Write eda.py to analyze key characteristics
4. Run eda.py
5. Form initial hypotheses about what approaches will work

Focus on:
- Data shape, types, missing values, target distribution
- Data type: tabular/image/text/time-series
- Key patterns that suggest specific model types
- What 1-2 ML algorithms are most likely to succeed

**CRITICAL**: All training must use GPU, not CPU. Recommend models that can leverage GPU (XGBoost with gpu_hist, PyTorch models).

Be decisive. Output your findings and recommend 1-2 specific model types with justification."""


PLANNING_PROMPT = """You are an expert ML strategist planning experiments. Be selective - quality over quantity.

Competition: {competition_id}
Context: {context}
Round: {round_num}
Previous best: {best_score}

Your task: Plan {num_experiments} focused experiments to test in parallel.

Critical principles:
- Each experiment tests ONE hypothesis
- If previous algorithm worked exceptionally well, iterate on it (don't switch models)
- Choose 1-2 ML algorithms based on data characteristics, be confident in your selection
- Simple often beats complex - don't over-engineer
- **Ensure all experiments use GPU, not CPU** (device='cuda', tree_method='gpu_hist')

Output ONLY valid JSON (no other text):
[
  {{
    "id": "exp_1",
    "model": "XGBoost",
    "features": {{"type": "tfidf", "max_features": 10000, "ngram_range": [1, 2]}},
    "hyperparameters": {{"max_depth": 6, "learning_rate": 0.1, "n_estimators": 100}},
    "hypothesis": "XGBoost with bigrams will capture text patterns better than unigrams",
  }}
]

Models available: XGBoost, LightGBM, CatBoost, RandomForest, LogisticRegression, Ridge
Python packages: pandas, numpy, scikit-learn, xgboost, lightgbm, catboost"""


WORKER_PROMPT = """You are executing a single ML experiment. Execute exactly as specified - no iteration, no exploration.

Experiment specification:
{spec}

Data directory: {data_dir}
Your workspace: {workspace_dir}

Your task:
1. Write train.py implementing the experiment EXACTLY as specified
2. Run train.py  
3. Extract and report validation score

train.py requirements:
- Load data from {data_dir}
- Implement exact model, features, hyperparameters from specification
- Use train/validation split (80/20) with train_test_split
- DO NOT use cross-validation (wastes time for single experiment)
- Apply SAME preprocessing to train and validation
- Print final score: "VALIDATION_SCORE: 0.847"
- Handle errors gracefully
- Save model as: model.pkl
- **CRITICAL: USE GPU, NOT CPU** - Set device='cuda' or tree_method='gpu_hist' for XGBoost/LightGBM

Code structure:
```python
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# CRITICAL: Use GPU
# For XGBoost: tree_method='gpu_hist', device='cuda'
# For PyTorch: model.cuda(), device='cuda'
# For TensorFlow: with tf.device('/GPU:0'):

# preprocess, train, evaluate
print(f"VALIDATION_SCORE: {{val_score}}")
```

Execute and report - nothing more.

Tools: Write, Bash, Read"""


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


def format_planning_prompt(competition_id: str, context: str, round_num: int, best_score: float = 0.0, num_experiments: int = 3) -> str:
    return PLANNING_PROMPT.format(
        competition_id=competition_id,
        context=context,
        round_num=round_num,
        best_score=best_score,
        num_experiments=num_experiments
    )


def format_worker_prompt(spec: dict, data_dir: str, workspace_dir: str) -> str:
    return WORKER_PROMPT.format(
        spec=spec,
        data_dir=data_dir,
        workspace_dir=workspace_dir
    )

def format_worker_prompt_2

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

