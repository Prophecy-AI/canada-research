import json
from pathlib import Path

def analyze_experiments():
    with open('experiments_output.json', 'r') as f:
        data = json.load(f)
    
    print("\n" + "="*80)
    print("ANALYSIS: Planner Choices vs Best Practices")
    print("="*80)
    
    for comp_id, comp_data in data.items():
        plan = comp_data.get('Plan', [])
        if isinstance(plan, str):
            continue
        
        eda = comp_data.get('EDA', '')
        
        print(f"\n{'='*80}")
        print(f"Competition: {comp_id}")
        print(f"{'='*80}")
        
        data_type = "unknown"
        if "Image" in eda or "image" in eda:
            data_type = "image"
        elif "Tabular" in eda or "tabular" in eda or "CSV" in eda:
            data_type = "tabular"
        elif "Text" in eda or "NLP" in eda:
            data_type = "text"
        elif "Audio" in eda or "audio" in eda:
            data_type = "audio"
        
        print(f"Data type: {data_type}")
        print(f"Experiments: {len(plan)}")
        
        for exp in plan:
            exp_id = exp.get('id', '?')
            strategy = exp.get('strategy', '?')
            model = exp.get('model', exp.get('models', '?'))
            
            print(f"\n  {exp_id}:")
            print(f"    Strategy: {strategy}")
            print(f"    Model: {model}")
            
            if strategy == "fastai_vision":
                hyperparams = exp.get('hyperparameters', {})
                epochs = hyperparams.get('epochs', '?')
                size = hyperparams.get('size', '?')
                lr = hyperparams.get('lr', '?')
                split_pct = hyperparams.get('split_pct', '?')
                
                print(f"    Hyperparameters:")
                print(f"      epochs: {epochs}")
                print(f"      size: {size}")
                print(f"      lr: {lr}")
                print(f"      split_pct: {split_pct}")
                
                issues = []
                improvements = []
                
                if split_pct == 0.01:
                    issues.append("⚠️  Very small validation set (1%) - might overfit")
                    improvements.append("→ Try 5-10% validation for better generalization estimate")
                
                if size == 128:
                    improvements.append("→ Try size=224 for better feature learning (gold uses 128, but larger might help)")
                
                if epochs <= 5:
                    improvements.append("→ Try 7-10 epochs with early stopping")
                
                if 'tta' not in hyperparams or not hyperparams.get('tta'):
                    improvements.append("→ Add TTA (test-time augmentation) for +2-5% boost")
                
                if 'mixup' not in hyperparams:
                    improvements.append("→ Consider mixup augmentation for better regularization")
                
                for issue in issues:
                    print(f"    {issue}")
                for imp in improvements:
                    print(f"    {imp}")
            
            elif strategy == "gradient_boosting":
                hyperparams = exp.get('hyperparameters', {})
                n_estimators = hyperparams.get('n_estimators', '?')
                learning_rate = hyperparams.get('learning_rate', '?')
                
                print(f"    Hyperparameters:")
                print(f"      n_estimators: {n_estimators}")
                print(f"      learning_rate: {learning_rate}")
                
                improvements = []
                if n_estimators <= 500:
                    improvements.append("→ Try n_estimators=1000-2000 for better performance")
                if learning_rate >= 0.05:
                    improvements.append("→ Try lr=0.01-0.03 with more trees")
                
                for imp in improvements:
                    print(f"    {imp}")


if __name__ == "__main__":
    analyze_experiments()

