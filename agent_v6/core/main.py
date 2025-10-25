"""
Main entry point for Agent V6 with new architecture
"""

import asyncio
import os
import sys
from pathlib import Path

# Add agent directory to path
AGENT_DIR = os.environ.get('AGENT_DIR', '/home/agent')
sys.path.insert(0, AGENT_DIR)

# Import after path setup
from agent_v6.core.orchestrator import Orchestrator


async def main():
    """Main entry point for Agent V6"""
    
    # Get environment variables
    competition_id = os.environ.get('COMPETITION_ID', 'unknown')
    data_dir = os.environ.get('DATA_DIR', '/home/data')
    submission_dir = os.environ.get('SUBMISSION_DIR', '/home/submission')
    code_dir = os.environ.get('CODE_DIR', '/home/code')
    instructions_path = os.environ.get('INSTRUCTIONS_PATH', '/home/instructions.txt')
    
    # Optional: time limit from environment
    time_limit = int(os.environ.get('TIME_LIMIT_MINUTES', '240'))  # 4 hours default
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                         AGENT V6                              ║
║  Competition: {competition_id:<45} ║
║  Time Limit: {time_limit} minutes                              ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    print(f"📁 Data Directory: {data_dir}")
    print(f"💾 Workspace: {code_dir}")
    print(f"📤 Submission: {submission_dir}")
    print(f"📄 Instructions: {instructions_path}")
    print()
    
    # Create orchestrator
    orchestrator = Orchestrator(
        competition_id=competition_id,
        data_dir=data_dir,
        submission_dir=submission_dir,
        workspace_dir=code_dir,
        instructions_path=instructions_path,
        time_limit_minutes=time_limit
    )
    
    # Run competition
    print("🚀 Starting orchestrated execution...\n")
    await orchestrator.run()
    
    # Check if submission was created
    submission_path = Path(submission_dir) / "submission.csv"
    
    print("\n" + "="*60)
    print("EXECUTION COMPLETE")
    print("="*60)
    
    if submission_path.exists():
        print(f"✅ SUCCESS: Submission created at {submission_path}")
        print(f"   File size: {submission_path.stat().st_size} bytes")
        
        # Show preview
        try:
            import pandas as pd
            df = pd.read_csv(submission_path)
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print("\n📊 First 5 rows:")
            print(df.head())
            print("\n📊 Last 5 rows:")
            print(df.tail())
            
            # Basic statistics
            if df.shape[1] > 1:
                pred_col = df.columns[1]  # Assume second column is predictions
                print(f"\n📈 Prediction Statistics ({pred_col}):")
                if df[pred_col].dtype in ['float64', 'int64']:
                    print(f"   Mean: {df[pred_col].mean():.6f}")
                    print(f"   Std:  {df[pred_col].std():.6f}")
                    print(f"   Min:  {df[pred_col].min():.6f}")
                    print(f"   Max:  {df[pred_col].max():.6f}")
                else:
                    print(f"   Unique values: {df[pred_col].nunique()}")
                    print(f"   Most common: {df[pred_col].value_counts().head(3).to_dict()}")
        
        except Exception as e:
            print(f"⚠️ Could not analyze submission: {e}")
        
        return 0
    else:
        print(f"❌ ERROR: No submission file found at {submission_path}")
        
        # Check for alternative locations
        print("\n📁 Checking alternative locations...")
        
        alternative_paths = [
            Path(code_dir) / "submission.csv",
            Path(code_dir) / "notebooks" / "submission.csv",
            Path(code_dir) / "output" / "submission.csv"
        ]
        
        for alt_path in alternative_paths:
            if alt_path.exists():
                print(f"   Found at: {alt_path}")
                # Try to copy to correct location
                try:
                    import shutil
                    Path(submission_dir).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(alt_path, submission_path)
                    print(f"   ✅ Copied to {submission_path}")
                    return 0
                except Exception as e:
                    print(f"   ❌ Could not copy: {e}")
        
        print("\n📁 Contents of submission directory:")
        try:
            for item in Path(submission_dir).iterdir():
                print(f"   - {item.name} ({item.stat().st_size} bytes)")
        except:
            print("   Directory is empty or doesn't exist")
        
        return 1


if __name__ == "__main__":
    # Run async main
    exit_code = asyncio.run(main())
    sys.exit(exit_code)