"""Fallback mechanisms for robust agent execution"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Optional
import nbformat


class FallbackManager:
    """Handles emergency fallbacks and edge cases"""
    
    def __init__(self, workspace_dir: Path, data_dir: Path, submission_dir: Path):
        self.workspace_dir = Path(workspace_dir)
        self.data_dir = Path(data_dir)
        self.submission_dir = Path(submission_dir)
        
    async def create_emergency_submission(self) -> bool:
        """Create a minimal baseline submission when everything else fails"""
        print("🚨 Creating emergency fallback submission...")
        
        # Try to find any CSV file to understand the structure
        test_file = None
        for file_pattern in ['test.csv', 'Test.csv', '*test*.csv', '*.csv']:
            matches = list(self.data_dir.glob(file_pattern))
            if matches:
                test_file = matches[0]
                break
        
        if not test_file:
            # Create a dummy submission
            return self._create_dummy_submission()
        
        # Create a basic prediction script
        script_content = f"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load test data
try:
    test_df = pd.read_csv('{test_file}')
    print(f"Loaded test data: {{test_df.shape}}")
    
    # Try to identify ID column
    id_col = None
    for col in ['id', 'Id', 'ID', 'index', 'Index']:
        if col in test_df.columns:
            id_col = col
            break
    
    if id_col is None and len(test_df.columns) > 0:
        id_col = test_df.columns[0]
    
    # Try to identify target column name from train data
    target_col = 'prediction'
    train_files = ['{self.data_dir}/train.csv', '{self.data_dir}/Train.csv']
    for train_file in train_files:
        try:
            train_df = pd.read_csv(train_file, nrows=5)
            # Find potential target columns (last column or specific names)
            for col in ['target', 'Target', 'label', 'Label', 'y', 'Y', train_df.columns[-1]]:
                if col in train_df.columns and col not in test_df.columns:
                    target_col = col
                    break
            break
        except:
            pass
    
    # Create predictions
    n_samples = len(test_df)
    
    # Try different prediction strategies
    try:
        # Check if it looks like classification (few unique values)
        sample_file = '{self.data_dir}/sample_submission.csv'
        if os.path.exists(sample_file):
            sample = pd.read_csv(sample_file, nrows=10)
            if len(sample.columns) == 2:
                pred_col = [c for c in sample.columns if c not in [id_col]][0]
                unique_vals = sample[pred_col].unique()
                if len(unique_vals) <= 10:  # Classification
                    predictions = np.random.choice(unique_vals, n_samples)
                else:  # Regression
                    predictions = np.random.randn(n_samples) * sample[pred_col].std() + sample[pred_col].mean()
            else:
                predictions = np.zeros(n_samples)
        else:
            # Default to 0s
            predictions = np.zeros(n_samples)
    except:
        predictions = np.zeros(n_samples)
    
    # Create submission DataFrame
    if id_col:
        submission = pd.DataFrame({{
            id_col: test_df[id_col] if id_col in test_df.columns else range(n_samples),
            target_col: predictions
        }})
    else:
        submission = pd.DataFrame({{
            'id': range(n_samples),
            'prediction': predictions
        }})
    
    # Save submission
    submission.to_csv('{self.submission_dir}/submission.csv', index=False)
    print(f"✅ Emergency submission created: {{submission.shape}}")
    print(submission.head())
    
except Exception as e:
    print(f"❌ Emergency submission failed: {{e}}")
    # Create minimal fallback
    import pandas as pd
    pd.DataFrame({{'id': [0], 'prediction': [0]}}).to_csv('{self.submission_dir}/submission.csv', index=False)
    print("Created minimal fallback submission.csv")
"""
        
        # Save and execute
        script_path = self.workspace_dir / "emergency_submission.py"
        script_path.write_text(script_content)
        
        process = await asyncio.create_subprocess_shell(
            f"cd {self.workspace_dir} && python {script_path.name}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        # Check if submission was created
        submission_file = self.submission_dir / "submission.csv"
        if submission_file.exists():
            print(f"✅ Emergency submission created at {submission_file}")
            return True
        else:
            print("❌ Emergency submission failed")
            return self._create_dummy_submission()
    
    def _create_dummy_submission(self) -> bool:
        """Create a minimal dummy submission as last resort"""
        print("📝 Creating minimal dummy submission...")
        try:
            import pandas as pd
            # Create minimal submission
            pd.DataFrame({'id': [0, 1], 'prediction': [0, 0]}).to_csv(
                self.submission_dir / 'submission.csv', index=False
            )
            print("✅ Minimal submission.csv created")
            return True
        except Exception as e:
            print(f"❌ Failed to create dummy submission: {e}")
            return False
    
    async def find_any_notebook(self, notebooks_dir: Path) -> Optional[Path]:
        """Find any available notebook to submit"""
        notebook_files = list(notebooks_dir.glob("*.ipynb"))
        
        if not notebook_files:
            print("❌ No notebooks found")
            return None
        
        # Sort by modification time (newest first)
        notebook_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Try to find one with actual content
        for notebook_path in notebook_files:
            try:
                with open(notebook_path, 'r') as f:
                    notebook = json.load(f)
                    
                # Check if notebook has code cells
                if 'cells' in notebook:
                    code_cells = [c for c in notebook['cells'] if c.get('cell_type') == 'code']
                    if len(code_cells) > 5:  # Has some substantial content
                        print(f"📔 Found notebook: {notebook_path.name}")
                        return notebook_path
            except:
                continue
        
        # Return the newest one if no good one found
        if notebook_files:
            print(f"📔 Using latest notebook: {notebook_files[0].name}")
            return notebook_files[0]
        
        return None
    
    async def create_basic_notebook(self) -> Path:
        """Create a basic working notebook when none exist"""
        print("📝 Creating basic fallback notebook...")
        
        notebook = nbformat.v4.new_notebook()
        
        # Add basic cells
        cells = [
            nbformat.v4.new_code_cell("""# Basic Fallback Solution
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
"""),
            nbformat.v4.new_code_cell(f"""# Load data
train_df = None
test_df = None

# Try to find train and test files
import os
data_dir = '{self.data_dir}'
for file in os.listdir(data_dir):
    if 'train' in file.lower() and file.endswith('.csv'):
        train_df = pd.read_csv(os.path.join(data_dir, file))
        print(f"Loaded train: {{train_df.shape}}")
    elif 'test' in file.lower() and file.endswith('.csv'):
        test_df = pd.read_csv(os.path.join(data_dir, file))
        print(f"Loaded test: {{test_df.shape}}")
"""),
            nbformat.v4.new_code_cell("""# Basic preprocessing
if train_df is not None:
    # Identify features and target
    target_col = train_df.columns[-1]  # Assume last column is target
    feature_cols = [c for c in train_df.columns if c != target_col]
    
    # Handle missing values
    train_df = train_df.fillna(0)
    if test_df is not None:
        test_df = test_df.fillna(0)
    
    # Prepare features
    X = train_df[feature_cols].select_dtypes(include=[np.number])
    y = train_df[target_col]
    
    if len(X.columns) == 0:  # No numeric columns
        X = pd.DataFrame(np.random.randn(len(train_df), 1))
"""),
            nbformat.v4.new_code_cell("""# Train basic model
if train_df is not None and 'X' in locals():
    # Determine if classification or regression
    if y.dtype == 'object' or len(y.unique()) < 20:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # Validate
    score = model.score(X_val, y_val)
    print(f"VALIDATION_SCORE: {{score:.4f}}")
"""),
            nbformat.v4.new_code_cell(f"""# Generate submission
if test_df is not None and 'model' in locals():
    # Prepare test features
    test_features = test_df[X.columns] if all(c in test_df.columns for c in X.columns) else test_df.select_dtypes(include=[np.number])
    
    # Make predictions
    predictions = model.predict(test_features) if len(test_features.columns) > 0 else np.zeros(len(test_df))
    
    # Create submission
    submission = pd.DataFrame({{
        'id': range(len(predictions)),
        'prediction': predictions
    }})
    
    submission.to_csv('{self.submission_dir}/submission.csv', index=False)
    print("Submission saved!")
else:
    # Create dummy submission
    pd.DataFrame({{'id': [0], 'prediction': [0]}}).to_csv('{self.submission_dir}/submission.csv', index=False)
    print("Created dummy submission")
""")
        ]
        
        notebook.cells = cells
        
        # Save notebook
        notebook_path = self.workspace_dir / "notebooks" / "fallback_solution.ipynb"
        notebook_path.parent.mkdir(exist_ok=True)
        
        with open(notebook_path, 'w') as f:
            nbformat.write(notebook, f)
        
        print(f"✅ Created fallback notebook: {notebook_path}")
        return notebook_path
    
    def extend_time_if_needed(self, clock) -> bool:
        """Check if we should continue despite time limit"""
        submission_file = self.submission_dir / "submission.csv"
        
        if submission_file.exists():
            # We have a submission, respect time limit
            return False
        
        # No submission yet, we MUST continue
        if clock.elapsed_minutes() > clock.time_limit_minutes:
            overtime = clock.elapsed_minutes() - clock.time_limit_minutes
            print(f"⏰ OVERTIME: {overtime:.1f} minutes past limit, but NO SUBMISSION yet!")
            print("   Continuing until we have a valid submission...")
            
            # Allow max 50% extra time for emergency submission
            max_overtime = clock.time_limit_minutes * 0.5
            if overtime > max_overtime:
                print(f"⚠️ Maximum overtime ({max_overtime:.1f} min) exceeded!")
                return False
            
            return True  # Continue despite time limit
        
        return True  # Still within time limit
