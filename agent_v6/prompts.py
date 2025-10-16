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


WORKER_PROMPT = """Write train.py. NO exploration, NO running. Just write file.

Spec: {spec}
EDA: {eda_context}
Data: {data_dir}
Workspace: {workspace_dir}

**Template for TABULAR/XGBoost/LightGBM:**
```python
import os, io, numpy as np, pandas as pd, pickle
from zipfile import ZipFile
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

try:
    df = pd.read_csv('{data_dir}/train.csv')
    
    # Load images from zip and flatten
    def load_images(zip_path, img_ids):
        images = []
        with ZipFile(zip_path, 'r') as z:
            for img_id in img_ids:
                img = Image.open(io.BytesIO(z.read(f'train/{{img_id}}')))
                images.append(np.array(img))
        return np.array(images)
    
    X = load_images('{data_dir}/train.zip', df['id'])
    y = df['has_cactus'].values
    X = X.reshape(len(X), -1)  # Flatten to (N, 3072)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train XGBoost/LightGBM with spec hyperparameters
    # model = xgboost.XGBClassifier(tree_method='gpu_hist', device='cuda', ...)
    # model.fit(X_train, y_train)
    
    y_pred = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, y_pred)
    print(f"VALIDATION_SCORE: {{score:.6f}}")
    pickle.dump(model, open('model.pkl', 'wb'))

except Exception as e:
    print(f"ERROR: {{e}}")
    import traceback; traceback.print_exc()
```

**Template for PRETRAINED CNN (ResNet/EfficientNet):**
```python
import os, io, numpy as np, pandas as pd, torch, pickle
from zipfile import ZipFile
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

try:
    df = pd.read_csv('{data_dir}/train.csv')
    
    class ImageDataset(Dataset):
        def __init__(self, zip_path, img_ids, labels, transform=None):
            self.zip_path = zip_path
            self.img_ids = img_ids
            self.labels = labels
            self.transform = transform
        
        def __len__(self):
            return len(self.img_ids)
        
        def __getitem__(self, idx):
            with ZipFile(self.zip_path, 'r') as z:
                img = Image.open(io.BytesIO(z.read(f'train/{{self.img_ids[idx]}}')))
                if self.transform:
                    img = self.transform(img)
                return img, self.labels[idx]
    
    transform = transforms.Compose([
        transforms.Resize(224),  # ResNet expects 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ids, val_ids, y_train, y_val = train_test_split(
        df['id'].values, df['has_cactus'].values, test_size=0.2, random_state=42, stratify=df['has_cactus']
    )
    
    train_dataset = ImageDataset('{data_dir}/train.zip', train_ids, y_train, transform)
    val_dataset = ImageDataset('{data_dir}/train.zip', val_ids, y_val, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    
    # Load pretrained model
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Binary classification
    model = model.cuda()
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train for specified epochs
    for epoch in range(20):
        model.train()
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.float().cuda()
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.cuda()
            outputs = torch.sigmoid(model(images).squeeze()).cpu().numpy()
            all_preds.extend(outputs)
            all_labels.extend(labels.numpy())
    
    score = roc_auc_score(all_labels, all_preds)
    print(f"VALIDATION_SCORE: {{score:.6f}}")
    torch.save(model.state_dict(), 'model.pkl')

except Exception as e:
    print(f"ERROR: {{e}}")
    import traceback; traceback.print_exc()
```

Implement spec EXACTLY. Respond "READY".

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

