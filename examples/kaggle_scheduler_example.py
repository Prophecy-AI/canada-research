"""
Example: Using TaskScheduler with Kaggle Competition Agent

Demonstrates how an agent can dynamically schedule tasks based on:
- Time remaining in epoch
- Task complexity and value
- Predicted durations

Scenario: 60-minute epoch with multiple modeling approaches
"""

import asyncio
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_v5.task_scheduler import TaskScheduler, TaskPriority, TaskComplexity
from agent_v5.tools.estimate_duration import EstimateTaskDurationTool


# Simulate task execution functions
async def load_data():
    """Load competition data"""
    print("   📂 Loading train.csv and test.csv...")
    await asyncio.sleep(2)  # Simulate 2s load time
    return {"rows": 100000, "columns": 50}


async def basic_preprocessing():
    """Quick preprocessing - handle missing values, basic encoding"""
    print("   🧹 Basic preprocessing: missing values, label encoding...")
    await asyncio.sleep(5)
    return {"features_created": 5}


async def advanced_feature_engineering():
    """Complex feature engineering - interactions, aggregations"""
    print("   🔧 Advanced feature engineering: interactions, aggregations...")
    await asyncio.sleep(20)
    return {"features_created": 50}


async def train_simple_model():
    """Train simple model (e.g., logistic regression)"""
    print("   🤖 Training logistic regression...")
    await asyncio.sleep(10)
    return {"model": "logistic", "cv_score": 0.75}


async def train_lightgbm():
    """Train LightGBM with default params"""
    print("   🌲 Training LightGBM (default params)...")
    await asyncio.sleep(30)
    return {"model": "lightgbm", "cv_score": 0.82}


async def train_xgboost():
    """Train XGBoost with default params"""
    print("   🚀 Training XGBoost (default params)...")
    await asyncio.sleep(40)
    return {"model": "xgboost", "cv_score": 0.83}


async def hyperparameter_tuning():
    """Tune best model"""
    print("   ⚙️  Hyperparameter tuning (Optuna, 50 trials)...")
    await asyncio.sleep(180)  # 3 minutes
    return {"best_score": 0.87}


async def train_neural_network():
    """Train neural network"""
    print("   🧠 Training neural network (10 epochs)...")
    await asyncio.sleep(120)  # 2 minutes
    return {"model": "nn", "cv_score": 0.80}


async def ensemble_models():
    """Ensemble top models"""
    print("   🎯 Ensembling top 3 models...")
    await asyncio.sleep(15)
    return {"ensemble_score": 0.88}


async def create_submission():
    """Generate submission file"""
    print("   📝 Creating submission.csv...")
    await asyncio.sleep(3)
    return {"submission_path": "submission.csv"}


async def scenario_1_plenty_of_time():
    """
    Scenario 1: 60 minute epoch, 1 GB dataset
    Result: Should run everything except maybe hyperparameter tuning
    """
    print("\n" + "=" * 70)
    print("SCENARIO 1: Plenty of Time (60 min budget, 1 GB data)")
    print("=" * 70 + "\n")

    # Initialize scheduler with 60 minute budget
    scheduler = TaskScheduler(
        time_budget_seconds=60 * 60,  # 60 minutes
        estimate_tool=EstimateTaskDurationTool(workspace_dir="/tmp")
    )

    # Add tasks in order of typical workflow
    # Task 1: Load data (CRITICAL)
    scheduler.add_task(
        task_id="load",
        name="Load Data",
        execute_fn=load_data,
        priority=TaskPriority.CRITICAL,
        complexity=TaskComplexity.QUICK_WIN,
        value_score=1.0,  # Must have
        duration_min=1,
        duration_typical=2,
        duration_max=5
    )

    # Task 2: Basic preprocessing (CRITICAL)
    scheduler.add_task(
        task_id="basic_prep",
        name="Basic Preprocessing",
        execute_fn=basic_preprocessing,
        priority=TaskPriority.CRITICAL,
        complexity=TaskComplexity.QUICK_WIN,
        value_score=0.9,
        depends_on=["load"],
        duration_min=3,
        duration_typical=5,
        duration_max=10
    )

    # Task 3: Simple model (HIGH - quick baseline)
    scheduler.add_task(
        task_id="simple_model",
        name="Simple Model (Logistic Regression)",
        execute_fn=train_simple_model,
        priority=TaskPriority.HIGH,
        complexity=TaskComplexity.QUICK_WIN,
        value_score=0.6,  # Lower score but fast
        depends_on=["basic_prep"],
        duration_min=5,
        duration_typical=10,
        duration_max=20
    )

    # Task 4: Advanced features (HIGH)
    scheduler.add_task(
        task_id="adv_features",
        name="Advanced Feature Engineering",
        execute_fn=advanced_feature_engineering,
        priority=TaskPriority.HIGH,
        complexity=TaskComplexity.EFFICIENT,
        value_score=0.8,
        depends_on=["basic_prep"],
        duration_min=10,
        duration_typical=20,
        duration_max=40
    )

    # Task 5: LightGBM (HIGH)
    scheduler.add_task(
        task_id="lgbm",
        name="LightGBM Model",
        execute_fn=train_lightgbm,
        priority=TaskPriority.HIGH,
        complexity=TaskComplexity.EFFICIENT,
        value_score=0.85,
        depends_on=["adv_features"],
        duration_min=20,
        duration_typical=30,
        duration_max=60
    )

    # Task 6: XGBoost (MEDIUM)
    scheduler.add_task(
        task_id="xgb",
        name="XGBoost Model",
        execute_fn=train_xgboost,
        priority=TaskPriority.MEDIUM,
        complexity=TaskComplexity.EFFICIENT,
        value_score=0.85,
        depends_on=["adv_features"],
        duration_min=30,
        duration_typical=40,
        duration_max=80
    )

    # Task 7: Neural Network (LOW - exploratory)
    scheduler.add_task(
        task_id="nn",
        name="Neural Network",
        execute_fn=train_neural_network,
        priority=TaskPriority.LOW,
        complexity=TaskComplexity.EXPENSIVE,
        value_score=0.7,  # Uncertain value
        depends_on=["adv_features"],
        duration_min=60,
        duration_typical=120,
        duration_max=300
    )

    # Task 8: Hyperparameter tuning (LOW - only if time permits)
    scheduler.add_task(
        task_id="tuning",
        name="Hyperparameter Tuning",
        execute_fn=hyperparameter_tuning,
        priority=TaskPriority.LOW,
        complexity=TaskComplexity.EXPENSIVE,
        value_score=0.9,  # High value but takes forever
        depends_on=["lgbm", "xgb"],
        duration_min=120,
        duration_typical=180,
        duration_max=600
    )

    # Task 9: Ensemble (HIGH)
    scheduler.add_task(
        task_id="ensemble",
        name="Ensemble Models",
        execute_fn=ensemble_models,
        priority=TaskPriority.HIGH,
        complexity=TaskComplexity.QUICK_WIN,
        value_score=0.95,  # Usually improves score
        depends_on=["lgbm", "xgb"],  # Needs at least 2 models
        duration_min=10,
        duration_typical=15,
        duration_max=30
    )

    # Task 10: Submission (CRITICAL)
    scheduler.add_task(
        task_id="submission",
        name="Create Submission",
        execute_fn=create_submission,
        priority=TaskPriority.CRITICAL,
        complexity=TaskComplexity.QUICK_WIN,
        value_score=1.0,  # Must have
        depends_on=["ensemble"],
        duration_min=2,
        duration_typical=3,
        duration_max=5
    )

    # Execute with adaptive scheduling
    results = await scheduler.execute_all(adaptive=True, safety_margin=1.2)

    print("\n" + scheduler.get_summary())
    return results


async def scenario_2_time_crunch():
    """
    Scenario 2: 15 minute epoch, 10 GB dataset
    Result: Should prioritize quick wins, skip expensive tasks
    """
    print("\n" + "=" * 70)
    print("SCENARIO 2: Time Crunch (15 min budget, 10 GB data)")
    print("=" * 70 + "\n")

    # Initialize scheduler with only 15 minutes
    scheduler = TaskScheduler(
        time_budget_seconds=15 * 60,  # 15 minutes
        estimate_tool=EstimateTaskDurationTool(workspace_dir="/tmp")
    )

    # Same tasks but with scaled durations for larger data
    # (10GB vs 1GB = ~5x slower based on our size multiplier)

    scheduler.add_task(
        task_id="load",
        name="Load Data (10 GB)",
        execute_fn=load_data,
        priority=TaskPriority.CRITICAL,
        complexity=TaskComplexity.QUICK_WIN,
        value_score=1.0,
        duration_min=5,
        duration_typical=10,  # 5x slower
        duration_max=25
    )

    scheduler.add_task(
        task_id="basic_prep",
        name="Basic Preprocessing (10 GB)",
        execute_fn=basic_preprocessing,
        priority=TaskPriority.CRITICAL,
        complexity=TaskComplexity.QUICK_WIN,
        value_score=0.9,
        depends_on=["load"],
        duration_min=15,
        duration_typical=25,  # 5x slower
        duration_max=50
    )

    scheduler.add_task(
        task_id="simple_model",
        name="Simple Model",
        execute_fn=train_simple_model,
        priority=TaskPriority.HIGH,
        complexity=TaskComplexity.QUICK_WIN,
        value_score=0.6,
        depends_on=["basic_prep"],
        duration_min=25,
        duration_typical=50,
        duration_max=100
    )

    # Skip advanced features - too expensive
    scheduler.add_task(
        task_id="adv_features",
        name="Advanced Features (10 GB)",
        execute_fn=advanced_feature_engineering,
        priority=TaskPriority.MEDIUM,  # Downgraded
        complexity=TaskComplexity.EXPENSIVE,
        value_score=0.8,
        depends_on=["basic_prep"],
        duration_min=50,
        duration_typical=100,
        duration_max=200
    )

    scheduler.add_task(
        task_id="lgbm",
        name="LightGBM",
        execute_fn=train_lightgbm,
        priority=TaskPriority.HIGH,
        complexity=TaskComplexity.EFFICIENT,
        value_score=0.85,
        depends_on=["basic_prep"],  # Use basic features only
        duration_min=100,
        duration_typical=150,
        duration_max=300
    )

    # Skip expensive models
    scheduler.add_task(
        task_id="xgb",
        name="XGBoost",
        execute_fn=train_xgboost,
        priority=TaskPriority.LOW,  # Downgraded
        complexity=TaskComplexity.EXPENSIVE,
        value_score=0.85,
        depends_on=["basic_prep"],
        duration_min=150,
        duration_typical=200,
        duration_max=400
    )

    scheduler.add_task(
        task_id="submission",
        name="Create Submission",
        execute_fn=create_submission,
        priority=TaskPriority.CRITICAL,
        complexity=TaskComplexity.QUICK_WIN,
        value_score=1.0,
        depends_on=["lgbm"],  # Just use best single model
        duration_min=2,
        duration_typical=3,
        duration_max=5
    )

    # Execute with adaptive scheduling
    results = await scheduler.execute_all(adaptive=True, safety_margin=1.2)

    print("\n" + scheduler.get_summary())
    return results


async def scenario_3_mid_epoch_replan():
    """
    Scenario 3: Started with 30 min, half-way through realize we're behind
    Result: Should dynamically re-prioritize and skip low-value tasks
    """
    print("\n" + "=" * 70)
    print("SCENARIO 3: Mid-Epoch Re-planning (started 30 min, now 10 min left)")
    print("=" * 70 + "\n")

    # Simulate being 20 minutes into a 30 minute epoch
    scheduler = TaskScheduler(
        time_budget_seconds=30 * 60,
        estimate_tool=EstimateTaskDurationTool(workspace_dir="/tmp")
    )

    # Manually advance the clock to simulate 20 minutes elapsed
    scheduler.started_at = time.time() - (20 * 60)  # Started 20 min ago

    print(f"⏰ Simulating mid-epoch state: 20 min elapsed, 10 min remaining\n")

    # Add remaining tasks
    scheduler.add_task(
        task_id="lgbm",
        name="LightGBM (in progress)",
        execute_fn=train_lightgbm,
        priority=TaskPriority.HIGH,
        complexity=TaskComplexity.EFFICIENT,
        value_score=0.85,
        duration_min=20,
        duration_typical=30,
        duration_max=60
    )

    scheduler.add_task(
        task_id="xgb",
        name="XGBoost",
        execute_fn=train_xgboost,
        priority=TaskPriority.MEDIUM,
        complexity=TaskComplexity.EFFICIENT,
        value_score=0.85,
        duration_min=30,
        duration_typical=40,
        duration_max=80
    )

    scheduler.add_task(
        task_id="nn",
        name="Neural Network",
        execute_fn=train_neural_network,
        priority=TaskPriority.LOW,
        complexity=TaskComplexity.EXPENSIVE,
        value_score=0.7,
        duration_min=60,
        duration_typical=120,
        duration_max=300
    )

    scheduler.add_task(
        task_id="ensemble",
        name="Ensemble",
        execute_fn=ensemble_models,
        priority=TaskPriority.HIGH,
        complexity=TaskComplexity.QUICK_WIN,
        value_score=0.95,
        depends_on=["lgbm"],  # Just need 1 model minimum
        duration_min=10,
        duration_typical=15,
        duration_max=30
    )

    scheduler.add_task(
        task_id="submission",
        name="Create Submission",
        execute_fn=create_submission,
        priority=TaskPriority.CRITICAL,
        complexity=TaskComplexity.QUICK_WIN,
        value_score=1.0,
        depends_on=["lgbm"],  # Can submit with just one model
        duration_min=2,
        duration_typical=3,
        duration_max=5
    )

    # Execute - should skip NN and XGBoost, focus on finishing LGBM + submission
    results = await scheduler.execute_all(adaptive=True, safety_margin=1.2)

    print("\n" + scheduler.get_summary())
    return results


async def main():
    """Run all scenarios"""
    print("\n" + "🎯" * 35)
    print("Task Scheduler Examples for Kaggle Competitions")
    print("🎯" * 35)

    # Scenario 1: Plenty of time
    await scenario_1_plenty_of_time()

    # Scenario 2: Time crunch
    await scenario_2_time_crunch()

    # Scenario 3: Mid-epoch replan
    await scenario_3_mid_epoch_replan()

    print("\n" + "=" * 70)
    print("✅ All scenarios complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
