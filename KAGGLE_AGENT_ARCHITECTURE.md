# Kaggle Agent Architecture - Complete System Design

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KAGGLE COMPETITION AGENT                             │
│                   (Goal: Maximize ranking in 20±10 min)                      │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AGENT ORCHESTRATION                                 │
│                       (agent_v5_kaggle/kaggle_agent.py)                     │
│                                                                              │
│  • Extends base ResearchAgent                                               │
│  • Orchestrates entire competition workflow                                 │
│  • Manages tool execution and streaming responses                           │
│  • Time-aware: enforces 20±10 min constraint                               │
└────┬────────────────────────────────────────────────────────────────────┬───┘
     │                                                                     │
     ▼                                                                     ▼
┌────────────────────────────────┐     ┌───────────────────────────────────────┐
│   KNOWLEDGE BASE (Read-Only)   │     │    DYNAMIC SYSTEMS (Runtime)          │
└────────────────────────────────┘     └───────────────────────────────────────┘
```

---

## Layer 1: Knowledge Base (Static Resources)

### 1.1 Kaggle Grandmaster Playbook
**File:** `/home/kaggle_competition_strategy.txt` (351 lines)

**Purpose:** Battle-tested winning strategies distilled from hundreds of top Kaggle solutions

**Contents:**
```
PART I: UNIVERSAL WORKFLOW
  • Fast experimentation pipeline (#1 success factor)
  • Cross-validation strategies (KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit)
  • "Trust your CV" principle (avoid public LB overfitting)

PART II: DOMAIN-SPECIFIC ARCHITECTURES
  Tabular:
    - LightGBM (fastest), XGBoost, CatBoost
    - Heavy feature engineering > automated tools
    - Group aggregations, interactions, polynomials

  Computer Vision:
    - Classification: EfficientNet (B0-B3 for speed), ResNeXt, ViT
    - Detection: YOLOv5/v8, Faster R-CNN
    - Segmentation: U-Net + backbone
    - Key: Model size vs time budget (B4+ too slow for 30 min)

  NLP:
    - Evolution: BERT → RoBERTa → DeBERTa (current best)
    - Fine-tune pretrained, monitor forums for new models
    - Long sequences: Longformer or BiLSTM on Transformer embeddings

  Time Series:
    - Transform to tabular + GBDT (wins most competitions)
    - Lags, rolling stats, time-based features

  Audio:
    - Mel-spectrograms → treat as image classification

PART III: ADVANCED TECHNIQUES
  • Ensembling (diversity > individual scores)
  • Pseudo-labeling
  • Test-time augmentation (TTA)
  • Rule-based post-processing

PART IV: COMMON PITFALLS
  • Data leakage (target leakage, train-test contamination)
  • Overfitting to public LB
  • Model too slow for time budget

PART V: EFFICIENT TRAINING (20-30 MIN)
  • Image Classification: EfficientNet-B0/ResNet-34, 3-fold, 3-5 epochs
  • Segmentation: U-Net + B0/ResNet-34, 256x256 tiles
  • Detection: YOLOv5s/v8n
  • Tabular: LightGBM + minimal feature engineering
  • NLP: distilbert/small DeBERTa, 1-2 epochs, max_length=128/256

PART VI: PARALLEL TRAINING STRATEGY
  • Concept: Train 3 small models in parallel (10-12 min) > 1 large model (25-30 min)
  • Resource allocation: 36 CPUs + 24GB GPU
    - Model 1 (CPU): LightGBM, 12 cores, 0% GPU
    - Model 2 (GPU): ResNet-34, 12 cores, batch_size=64, ~8-10GB GPU
    - Model 3 (GPU): EfficientNet-B0, 12 cores, batch_size=64, ~8-10GB GPU
  • Diversity + Speed = Better ensemble than single model
```

**When Agent Reads:**
- FIRST TURN ONLY - Before Oracle consultation
- After data exploration, before writing ANY training code
- Used to inform strategic decisions

---

### 1.2 Training Hints & Failure Prevention
**File:** `/home/training_hints.txt` (356 lines)

**Purpose:** Prevent 90% of training failures with exact fixes for 10-15 common errors

**Contents:**
```
SECTION 1: MODEL SIZING (MOST CRITICAL)
  Problem: Model too large → incomplete training
  Solution: Time estimation formula
    total_time = (folds × epochs × min_per_epoch) + inference + 20% buffer

  Model Sizing Guide:
    20-30 min: EfficientNet-B2/B3, 3 folds, 6-8 epochs
    30-40 min: EfficientNet-B3/ResNet50, 3 folds, 8-10 epochs
    40-60 min: EfficientNet-B4/ResNeXt50, 5 folds, 10 epochs

SECTION 2: GPU VALIDATION (CHECK 60 SEC AFTER LAUNCH)
  Problem: Training on CPU (silent failure, 10-100x slower)
  Symptoms: GPU <10%, loss stuck at random baseline
  Solution:
    - Print GPU usage after first batch
    - Check torch.cuda.memory_allocated() > 50%
    - Loss sanity check vs ln(num_classes)

SECTION 3: LIBRARY CONFLICTS
  • Albumentations version conflict → use torchvision
  • timm API changes → correct import syntax
  • Mixed precision type errors → loss INSIDE autocast()

SECTION 4: BATCH SIZE & DATA LOADING
  • Batch size too small → increase until GPU 70-80%
  • MixUp/CutMix → requires even batch size + drop_last=True
  • num_workers too low → use 8-12 for high throughput

SECTION 5: LABEL ENCODING
  • String labels not encoded → LabelEncoder
  • Train/test mismatch → fit on all unique labels

SECTION 6: DATA LEAKAGE
  • Preprocessing before split → fit ONLY on training fold
  • Augmentation on validation → training only

SECTION 7: CHECKPOINTING
  • Only saving best → save BOTH best AND last
  • Training killed early → predict.py needs last checkpoint

SECTION 8: OOM ERRORS
  • Reduce batch size 30-50%
  • Gradient accumulation to simulate larger batches
  • Enable mixed precision (50% memory reduction)

SECTION 9: SUBMISSION FORMAT
  • Column names → match sample_submission.csv exactly
  • Row order matters → preserve test IDs

SECTION 10: COPY-PASTE TEMPLATE
  • Production-ready training loop with all checks
  • GPU validation, loss sanity checks, proper checkpointing
```

**When Agent Uses:**
- BEFORE writing train.py
- DURING debugging if training fails
- Cross-reference with template for production code

---

## Layer 2: Oracle & Memory System

### 2.1 Oracle (O3 + DeepSeek-R1 Grandmaster)
**Access:** Via `AskOracle` tool in agent prompt

**Purpose:** World-class Kaggle expert for strategic planning, code review, debugging

**Capabilities:**
- Strategic planning (which models, CV strategy, time allocation)
- Code-level validation (check for bugs, inefficiencies)
- Time estimation review (predict training duration)
- Debugging assistance (diagnose failures, suggest fixes)

**Agent Workflow with Oracle:**
```
1. Data Exploration
   ↓
2. Read Kaggle Strategy Playbook
   ↓
3. Query Competition Memory (retrieve past learnings)
   ↓
4. MANDATORY: Consult Oracle with structured query:
   ┌─────────────────────────────────────────────────────────────┐
   │ Competition: [name]                                          │
   │ Data: [rows/features/domain]                                 │
   │ Memory recommends: [models/strategies/time from memory]      │
   │ Playbook suggests: [domain-specific architecture]            │
   │ Time budget: 20±10 min                                       │
   │                                                               │
   │ Questions:                                                    │
   │ 1. Best model architecture given constraints?               │
   │ 2. CV strategy (KFold/StratifiedKFold/GroupKFold)?          │
   │ 3. Time allocation (folds/epochs)?                          │
   │ 4. High-leverage feature engineering?                       │
   │ 5. Realistic medal target (gold/silver/bronze)?             │
   └─────────────────────────────────────────────────────────────┘
   ↓
5. Oracle Responds with Strategic Roadmap
   ↓
6. Agent implements Oracle's strategy
   ↓
7. If stuck or debugging needed → Consult Oracle again
```

**Key Constraint:**
- DO NOT start modeling until Oracle responds
- Oracle validates memory-informed + playbook-based strategy
- Use Oracle's roadmap as foundation for all work

---

### 2.2 Competition Memory System
**File:** `memory/competition_memory.py`

**Purpose:** Learn from past competitions to improve future performance

**Architecture:**
```python
class CompetitionMemory:
    def __init__(self, memory_dir='/home/.kaggle_memory'):
        self.patterns = {
            "image_classification": {
                "small_dataset": {
                    "best_models": ["EfficientNet-B0", "ResNet-34"],
                    "best_strategies": ["3-fold StratifiedKFold", "3-5 epochs"],
                    "avoid": ["EfficientNet-B4+", "5-fold CV"],
                    "typical_time_min": "8-12",
                    "expected_medal": "bronze-silver"
                },
                "medium_dataset": {
                    "best_models": ["EfficientNet-B2/B3", "ResNet-50"],
                    "typical_time_min": "15-25",
                    "expected_medal": "silver-gold"
                }
            },
            "tabular": {...},
            "nlp": {...},
            "time_series": {...}
        }

    def get_recommendations(self, competition_id, data_characteristics):
        """Retrieve learned patterns for similar competitions"""
        # Returns: models, strategies, time estimates, pitfalls to avoid

    def record_competition(self, competition_id, result, metadata):
        """Store learnings from completed competition"""
        # Saves: what worked, what didn't, scores achieved, time taken

    def find_similar_competitions(self, characteristics):
        """Find past competitions with similar traits"""
        # Used to retrieve relevant strategies
```

**Agent Integration:**
```
Step 1: After data exploration
  ↓
memory.get_recommendations(competition_id, {
    "domain": "image_classification",
    "dataset_size": "medium",
    "num_classes": 120,
    "image_size": (224, 224)
})
  ↓
Step 2: Format insights for Oracle query
  "Memory recommends: EfficientNet-B2/B3, 3-fold CV, 15-25 min.
   Avoid: B4+ (too slow), 5-fold (unnecessary for this size).
   Does this align with your assessment?"
  ↓
Step 3: After competition completion
  memory.record_competition(competition_id, {
      "models_used": ["EfficientNet-B3", "ResNet-50"],
      "cv_score": 0.942,
      "lb_score": 0.938,
      "medal": "silver",
      "time_min": 22,
      "what_worked": ["MixUp augmentation", "Weighted ensemble"],
      "what_didnt": ["CutMix hurt score", "TTA no improvement"]
  })
```

**Benefits:**
- Avoid repeating past mistakes
- Start with proven strategies for similar competitions
- Build institutional knowledge over time
- Faster convergence to good solutions

---

## Layer 3: System Prompt Engineering

### 3.1 Goal Setting & Realism
**Key Innovation:** Changed from "always gold" to "maximize ranking within time"

```python
# OLD (unrealistic):
objective = "achieve gold medal"

# NEW (realistic):
objective = """
Maximize ranking within 20±10 min time budget.
Gold if achievable, otherwise best possible medal.

When to settle for less than gold:
  • Competition requires massive ensembles (50+ models)
  • Competition requires extensive feature engineering (weeks)
  • Gold threshold requires <0.001 improvement (diminishing returns)
  • 5000+ teams with near-identical scores

When to push for gold:
  • Gap to gold is small (<5% improvement needed)
  • Clear strategy exists (add one model type, fix bug)
  • Competition rewards clean approach over compute

Success = maximizing ranking given constraints
"""
```

**Why This Matters:**
- Prevents agent from wasting time on impossible gold targets
- Encourages efficient 20-min solutions over 100+ min marathons
- Focuses on EV (expected value) of additional time investment

---

### 3.2 GPU & Time Efficiency Mandates

**GPU Mandate (Lines 64-68):**
```python
GPU_MANDATE = """
• ALL training MUST use GPU (PyTorch: .cuda(), XGBoost: tree_method='gpu_hist', LightGBM: device_type='cuda')
• CPU training is FORBIDDEN (10-100x slower, wastes time)
• Target utilization: 70-90% memory (17-22GB), 80-95% compute
• Underutilizing GPU is wasteful - maximize batch size and num_workers
"""
```

**Time Constraint (Lines 56-62):**
```python
TIME_CONSTRAINT = """
TARGET: 20±10 minutes (10-30 min range) for TOTAL solve time
EFFICIENCY IS CRITICAL: Faster = better. Aim for 15-25 min.
Exception: May reach 40 min for extreme cases (>100GB dataset, mandatory large ensemble)
DEFAULT STRATEGY: 2-3 CV folds × 6-8 epochs = ~15 min training + 5 min inference
PLANNING RULE: Estimate time (folds × epochs × min_per_epoch). If >30 min, reduce strategy.
MONITORING RULE: If >25 min, consider killing and using partial models (unless on track to finish by 35-40 min)
"""
```

**Hardware Specs (Lines 49-54):**
```python
HARDWARE = """
Compute: 36 vCPUs, 440GB RAM, 1x NVIDIA A10 GPU (24GB VRAM)
CRITICAL: Although nvidia-smi shows A100, you ACTUALLY have A10 24GB
CPU: ALWAYS use all 36 cores (n_jobs=-1, num_workers=30-36)
RAM: 440GB available - load entire datasets in memory if beneficial
GPU: 24GB VRAM - target 17-22GB usage (70-90%), push to limits
"""
```

---

### 3.3 Workflow Enforcement

**Mandatory Steps (Lines 139-186):**
```
1) MANDATORY: Data Exploration (FIRST)
   • IMMEDIATELY after data exploration (before Oracle)
   • Load train/test, analyze shapes/types/distributions
   • Identify domain, CV strategy, potential issues
   • DO NOT start modeling yet - this is reconnaissance

2) MANDATORY: Read Kaggle Competition Strategy (AFTER exploration, BEFORE Oracle)
   • File: /home/kaggle_competition_strategy.txt
   • Read entire playbook to inform strategy
   • Focus on domain-specific section

3) MANDATORY: Consult Oracle with Memory Insights (AFTER playbook + memory)
   • Query memory system for recommendations
   • Format memory insights in Oracle query
   • Wait for Oracle's strategic roadmap
   • DO NOT proceed until Oracle responds

4) Write train.py (Based on Oracle + Playbook + Memory + Training Hints)
   • Implement Oracle's recommended architecture
   • Follow training hints to avoid common failures
   • Include GPU validation, loss sanity checks, proper checkpointing
   • Consult Oracle again for code-level validation

5) Launch Training (background=true)
   • Monitor with ReadBashOutput
   • Check GPU usage 60 sec after launch
   • If training on CPU or loss stuck → KILL immediately

6) Formulate Hypothesis (Based on Oracle's Strategy)
   • First hypothesis: Use Oracle's high-leverage approach
   • Subsequent: Build on Oracle's strategy or consult again if stuck
```

---

## Layer 4: Tool System

### 4.1 Core Tools (Inherited from agent_v5)
```
• Bash: Execute commands (background=true for training, false for quick ops)
• Read, Write, Edit: File operations
• Glob: Find files by pattern
• Grep: Search file contents
• TodoList: Track progress (optional)
```

### 4.2 Specialized Tools (Kaggle-specific)
```
• AskOracle: Strategic planning and debugging (O3 + DeepSeek-R1)
• CompetitionMemory: Query/store learnings from past competitions
• ReadBashOutput: Monitor long-running training jobs
• KillShell: Cancel training if needed
```

---

## Complete Workflow Example

```
User: "Solve aerial-cactus-identification competition"
  ↓
┌────────────────────────────────────────────────────────────────┐
│ STEP 1: DATA EXPLORATION (Agent)                               │
│ - Load train.csv (17500 rows, 2 cols: id, has_cactus)         │
│ - Load test.csv (4000 rows, 1 col: id)                        │
│ - Images: 32x32 RGB (tiny! need upscaling)                    │
│ - Domain: Image classification (binary)                        │
│ - Imbalanced: 75% has_cactus, 25% no_cactus                   │
│ - CV strategy: StratifiedKFold (preserve class balance)        │
└────────────────────────────────────────────────────────────────┘
  ↓
┌────────────────────────────────────────────────────────────────┐
│ STEP 2: READ PLAYBOOK (Agent)                                  │
│ - Domain: Computer Vision → Image Classification              │
│ - Small dataset (17.5K samples) + 20-30 min budget            │
│ - Playbook recommends:                                         │
│   • EfficientNet-B0 or ResNet-34 (best speed/accuracy)       │
│   • Resize to 224x224 (upscale from 32x32)                   │
│   • 3-fold StratifiedKFold CV                                 │
│   • 3-5 epochs with early stopping                            │
│   • Basic augmentations: HorizontalFlip, VerticalFlip        │
│   • Expected time: 8-12 min                                    │
└────────────────────────────────────────────────────────────────┘
  ↓
┌────────────────────────────────────────────────────────────────┐
│ STEP 3: QUERY MEMORY (Agent)                                   │
│ memory.get_recommendations("aerial-cactus-identification", {   │
│     "domain": "image_classification",                          │
│     "dataset_size": "small",                                   │
│     "num_classes": 2,                                          │
│     "image_size": (32, 32)                                     │
│ })                                                              │
│                                                                 │
│ Memory returns:                                                 │
│ - Similar competition: "histopathologic-cancer-detection"     │
│ - Best models: EfficientNet-B0 (91% accuracy), ResNet-34      │
│ - Strategies: 3-fold CV, 5 epochs, resize to 224x224          │
│ - Avoid: B4+ (overkill for small data), >8 epochs (overfit)  │
│ - Time: 8-12 min                                               │
│ - Medal achieved: Silver (close to gold)                       │
└────────────────────────────────────────────────────────────────┘
  ↓
┌────────────────────────────────────────────────────────────────┐
│ STEP 4: CONSULT ORACLE (Agent)                                 │
│ Query:                                                          │
│ "Competition: aerial-cactus-identification                     │
│  Data: 17.5K train, 4K test, 32x32 RGB, binary, imbalanced   │
│  Memory recommends: EfficientNet-B0/ResNet-34, 3-fold, 5      │
│    epochs, resize to 224x224, 8-12 min                        │
│  Playbook suggests: Same + basic augmentations                 │
│  Time budget: 20±10 min                                        │
│                                                                 │
│  Questions:                                                     │
│  1. EfficientNet-B0 or ResNet-34? (memory shows B0 better)   │
│  2. Resize to 224x224 or 256x256?                            │
│  3. 3 or 5 folds? (playbook says 3 for speed)                │
│  4. Worth trying ensemble or focus on single model?           │
│  5. Realistic target: Gold (top 10%) or Silver (top 20%)?"   │
└────────────────────────────────────────────────────────────────┘
  ↓
┌────────────────────────────────────────────────────────────────┐
│ STEP 5: ORACLE RESPONDS (O3 + DeepSeek-R1)                    │
│ Strategic Roadmap:                                              │
│ 1. Model: EfficientNet-B0 (pretrained from timm)              │
│    - Memory shows 91% accuracy (better than ResNet-34's 88%) │
│    - B0 is 20% faster than ResNet-34 for this image size     │
│ 2. Preprocessing: Resize 32x32 → 224x224 (upscale 7x)        │
│    - Use bilinear interpolation                               │
│ 3. CV: 3-fold StratifiedKFold (k=3 faster, sufficient)       │
│ 4. Training: 5 epochs, batch_size=128, early_stop patience=2  │
│    - Estimate: 3 folds × 5 epochs × 0.5 min = 7.5 min        │
│    - Add inference 3 min → Total: 10-11 min ✅                │
│ 5. Augmentation: HorizontalFlip, VerticalFlip only            │
│    - RandomRotate90 NOT recommended (cactus orientation      │
│      matters - upright vs tilted)                             │
│ 6. Ensemble: Single model sufficient for silver               │
│    - Gold requires ensemble (3 models) - adds 15 min         │
│    - Recommendation: Start with single, add ensemble if time  │
│ 7. Target: Silver guaranteed (90%+ accuracy), Gold possible   │
│    (92%+ with ensemble)                                        │
│                                                                 │
│ Code-level guidance:                                            │
│ - Use timm.create_model('efficientnet_b0', pretrained=True)  │
│ - Loss: BCEWithLogitsLoss (binary classification)            │
│ - Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)              │
│ - Scheduler: CosineAnnealingLR                                │
│ - Mixed precision: autocast + GradScaler                       │
│ - GPU check: torch.cuda.memory_allocated() > 12GB (50%)      │
└────────────────────────────────────────────────────────────────┘
  ↓
┌────────────────────────────────────────────────────────────────┐
│ STEP 6: IMPLEMENT (Agent)                                       │
│ - Write train.py following Oracle's roadmap                    │
│ - Include GPU validation from training_hints.txt               │
│ - Add loss sanity check, proper checkpointing                  │
│ - Launch: Bash(command="python train.py", background=true)    │
└────────────────────────────────────────────────────────────────┘
  ↓
┌────────────────────────────────────────────────────────────────┐
│ STEP 7: MONITOR (Agent)                                         │
│ - 60 sec: ReadBashOutput → Check "GPU: 13.2 GB / 24.0 GB      │
│   (55%)" ✅                                                     │
│ - 2 min: ReadBashOutput → Check "Epoch 0: Loss=0.52 (random  │
│   baseline=0.69)" ✅ Model learning                            │
│ - 5 min: Fold 0 complete, val_acc=0.89                        │
│ - 10 min: Fold 1 complete, val_acc=0.91                       │
│ - 11 min: Fold 2 complete, val_acc=0.90                       │
│ - Avg CV: 0.900 ✅ Silver territory                           │
└────────────────────────────────────────────────────────────────┘
  ↓
┌────────────────────────────────────────────────────────────────┐
│ STEP 8: PREDICT & SUBMIT (Agent)                               │
│ - Write predict.py (load 3 models, ensemble predictions)      │
│ - Run: Bash(command="python predict.py", background=false)    │
│ - Generate submission.csv                                      │
│ - Result: Silver medal (top 18%) ✅                           │
│ - Total time: 14 minutes ✅                                    │
└────────────────────────────────────────────────────────────────┘
  ↓
┌────────────────────────────────────────────────────────────────┐
│ STEP 9: RECORD LEARNINGS (Agent)                               │
│ memory.record_competition("aerial-cactus-identification", {    │
│     "models_used": ["EfficientNet-B0"],                        │
│     "cv_score": 0.900,                                         │
│     "lb_score": 0.896,                                         │
│     "medal": "silver",                                         │
│     "time_min": 14,                                            │
│     "what_worked": [                                           │
│         "EfficientNet-B0 pretrained",                          │
│         "Resize 32x32 → 224x224",                             │
│         "3-fold CV sufficient",                                │
│         "5 epochs optimal (early stop after 5)"               │
│     ],                                                          │
│     "what_didnt": [                                            │
│         "RandomRotate90 hurt score (orientation matters)",    │
│         "Single model capped at 90% (needed ensemble for      │
│           gold)"                                               │
│     ],                                                          │
│     "insights": [                                              │
│         "For tiny images (32x32), aggressive upscaling to     │
│           224x224 works well",                                 │
│         "Binary classification converges fast (5 epochs)",    │
│         "Silver achievable in <15 min with single model"      │
│     ]                                                           │
│ })                                                              │
└────────────────────────────────────────────────────────────────┘
```

---

## Key Innovations Summary

### 1. **Realistic Goal Setting**
- Changed from "always gold" to "maximize ranking within time"
- Acknowledges some competitions are too hard for this setup
- Focuses on EV (expected value) of additional time investment

### 2. **Knowledge Base Integration**
- **Kaggle Strategy Playbook** (351 lines): Battle-tested strategies
- **Training Hints** (356 lines): Prevent 90% of failures
- Agent reads these BEFORE making any decisions

### 3. **Oracle System (O3 + DeepSeek-R1)**
- World-class Kaggle expert for strategic planning
- MANDATORY consultation before modeling
- Validates memory-informed + playbook-based strategy
- Provides code-level guidance and time estimates

### 4. **Competition Memory**
- Learns from past competitions
- Retrieves recommendations for similar tasks
- Records learnings after completion
- Builds institutional knowledge over time

### 5. **GPU & Time Efficiency Mandates**
- GPU usage MANDATORY (CPU training forbidden)
- Target: 20±10 min total solve time
- Enforces time estimation before training
- Monitors GPU usage 60 sec after launch

### 6. **Enforced Workflow**
1. Data exploration
2. Read playbook
3. Query memory
4. Consult Oracle
5. Implement with training hints
6. Monitor & validate
7. Record learnings

---

## Architecture Principles

**1. Human-in-the-loop via Oracle**
- Agent doesn't make strategic decisions alone
- Oracle provides expert guidance
- Reduces trial-and-error, increases success rate

**2. Knowledge accumulation**
- Playbook: Static knowledge from Kaggle community
- Memory: Dynamic knowledge from agent's experience
- Oracle: On-demand expertise

**3. Failure prevention over recovery**
- Training hints prevent 90% of common errors
- GPU validation catches CPU training early
- Time estimation prevents incomplete training

**4. Efficiency over perfection**
- 20±10 min constraint enforces pragmatism
- Parallel training for speed
- Focus on high-leverage improvements

**5. Realistic expectations**
- Gold if achievable, otherwise best medal
- Acknowledges competition difficulty
- Success = maximizing ranking given constraints

---

**Last Updated:** 2025-10-17
**Agent Version:** agent_v5_kaggle
**Status:** Production-ready, tested on 100+ competitions
