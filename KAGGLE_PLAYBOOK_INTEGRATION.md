# Kaggle Grandmaster Playbook Integration

**Date:** 2025-10-16
**Status:** ✅ Complete

---

## Overview

Integrated comprehensive Kaggle Grandmaster Playbook into agent system to provide battle-tested strategies from hundreds of winning solutions.

**Key Changes:**
1. Added playbook file to Docker environment
2. Updated agent prompt to mandate reading playbook before strategy/training
3. Updated Oracle prompt to reference playbook for validation
4. Restructured workflow to incorporate playbook at critical decision points

---

## 1. Playbook File Location

### Source File
**Location:** `/Users/Yifan/canada-research/mle-bench/environment/Kaggle Competition Strategy Cheatsheet.txt`

### Docker Container Path
**Mapped to:** `/home/kaggle_grandmaster_playbook.txt`

### Dockerfile Change
**File:** `/Users/Yifan/canada-research/mle-bench/environment/Dockerfile` (Line 82)
```dockerfile
COPY "environment/Kaggle Competition Strategy Cheatsheet.txt" /home/kaggle_grandmaster_playbook.txt
```

---

## 2. Playbook Contents Summary

The playbook is a comprehensive 268-line synthesis of winning Kaggle strategies covering:

### Part I: Universal Kaggle Workflow
- **Section 1:** Foundational principles (persistence, iteration, community engagement)
- **Section 2:** Fast experimentation pipeline (GPU acceleration, rapid iteration)
- **Section 3:** Cross-validation mastery (K-Fold, Stratified, Group, Time-Series splits)

### Part II: Domain-Specific Tactics
- **Section 3:** Tabular competitions (GBDTs: LightGBM/XGBoost/CatBoost, feature engineering, hybrid ensembles)
- **Section 4:** Computer Vision (EfficientNet/ResNeXt/ViT, advanced augmentation: MixUp/CutMix, TTA)
- **Section 5:** Natural Language Processing (BERT/RoBERTa/DeBERTa, fine-tuning, knowledge distillation)
- **Section 6:** Specialized domains (time series, audio classification, recommender systems)

### Part III: Advanced Strategies
- **Section 7:** Ensembling (diversity principle, blending, hill climbing, stacking)
- **Section 8:** Performance optimization (pseudo-labeling, TTA, rule-based post-processing)
- **Section 9:** Avoiding pitfalls (data leakage: target leakage, train-test contamination)

---

## 3. Agent Prompt Changes

### File: `/Users/Yifan/canada-research/mle-bench/agents/agent_v5_kaggle/kaggle_agent.py`

### Added Section (Lines 66-84): "KAGGLE GRANDMASTER KNOWLEDGE BASE"

```python
**KAGGLE GRANDMASTER KNOWLEDGE BASE (CRITICAL - READ THIS FIRST):**
- **File location:** /home/kaggle_grandmaster_playbook.txt
- **MANDATORY: Read this file BEFORE writing ANY training script or making strategic decisions**
- **Contents:** Comprehensive synthesis of winning Kaggle strategies covering:
  • Universal workflow principles (fast experimentation, rigorous CV strategies)
  • Domain-specific architectures and tactics:
    - Tabular: GBDTs (LightGBM/XGBoost/CatBoost), heavy feature engineering, GBDT+NN ensembles
    - Computer Vision: EfficientNet/ResNeXt/ViT, advanced augmentation (MixUp/CutMix), TTA
    - NLP: Transformer models (BERT/RoBERTa/DeBERTa), fine-tuning strategies, knowledge distillation
    - Time Series: Transform to tabular + GBDTs, lag/window features, TimeSeriesSplit CV
  • Advanced techniques: Stacking, pseudo-labeling, TTA, rule-based post-processing
  • Common pitfalls: Data leakage (target leakage, train-test contamination), overfitting to public LB
- **Why critical:** This playbook contains battle-tested strategies from hundreds of winning solutions
- **When to reference:**
  1. BEFORE initial strategy planning (consult Oracle AFTER reading playbook)
  2. BEFORE writing train.py (choose appropriate model architecture for domain)
  3. BEFORE designing CV strategy (match data structure to CV type)
  4. When stuck or getting poor results (check if violating playbook principles)
```

### Updated Workflow Step 2 (Lines 123-129): "MANDATORY: Read Kaggle Grandmaster Playbook"

```python
2) **MANDATORY: Read Kaggle Grandmaster Playbook** (FIRST TURN ONLY - After data exploration, BEFORE Oracle)
   • **Read /home/kaggle_grandmaster_playbook.txt in full** - this is the foundation of all strategy
   • Identify your competition's domain (tabular/CV/NLP/time-series/audio/recsys)
   • Note the recommended architectures and techniques for your domain
   • Understand the universal principles (fast experimentation, rigorous CV)
   • Pay special attention to common pitfalls section (data leakage, overfitting to public LB)
   • **This reading is NON-NEGOTIABLE - it contains battle-tested strategies from hundreds of winning solutions**
```

### Updated Workflow Step 3 (Lines 131-148): "MANDATORY: Consult Oracle" - Now references playbook

```python
3) **MANDATORY: Consult Oracle for Gold-Medal Strategy** (FIRST TURN ONLY - After reading playbook)
   After reading playbook AND completing data exploration, call Oracle with structured query:

   "I've read the Kaggle Grandmaster Playbook. Based on it, I understand this is a [domain] competition.

   Competition: [name]. Task: [classification/regression/time-series/etc]. Metric: [RMSE/AUC/F1/etc].
   Data: Train [X rows, Y cols], Test [Z rows]. Features: [A numerical, B categorical, C text/image].
   Target: [balanced/imbalanced/range]. Missing: [patterns]. Notable: [temporal/spatial patterns if any].
   Resources: {{os.cpu_count()}} CPU cores, A10 GPU 24GB, [X]GB RAM.

   Playbook recommends: [architecture/technique from playbook for this domain]
   My initial plan: [your plan based on playbook]

   Validate my strategy and recommend optimizations for gold-medal performance in 20±10 min."

   • DO NOT proceed with ANY modeling until Oracle responds
   • Oracle validates your playbook-based strategy and provides refinements
   • Use Oracle's strategic roadmap as foundation for all work
```

### Updated Workflow Step 9 (Line 181): "Execute" - Re-read playbook before train.py

```python
9) **Execute**
   • **BEFORE writing train.py: Re-read relevant sections of /home/kaggle_grandmaster_playbook.txt for your domain**
   • Oracle has already provided a gold-medal strategy - execute that plan, not generic baselines
   • **GPU MANDATE: ALL training/inference scripts MUST use GPU...**
```

---

## 4. Oracle Prompt Changes

### File: `/Users/Yifan/canada-research/agent_v5/tools/oracle.py`

### Added Section (Lines 182-191): "KAGGLE GRANDMASTER KNOWLEDGE BASE"

```python
"**KAGGLE GRANDMASTER KNOWLEDGE BASE:**\n"
"The agent has access to /home/kaggle_grandmaster_playbook.txt - a comprehensive synthesis of "
"winning Kaggle strategies covering:\n"
"• Universal workflow (fast experimentation, rigorous CV)\n"
"• Domain-specific tactics (GBDTs for tabular, Transformers for NLP, CNNs/ViTs for vision)\n"
"• Advanced strategies (ensembling, pseudo-labeling, TTA)\n"
"• Common pitfalls (data leakage, overfitting to public LB)\n"
"When reviewing agent's strategy or code, reference these battle-tested techniques. If agent's "
"approach contradicts the playbook (e.g., using simple averaging instead of stacking, not using "
"appropriate CV strategy), point it out explicitly.\n\n"
```

**Impact:**
- Oracle now knows agent has access to playbook
- Oracle validates agent's strategy against playbook recommendations
- Oracle explicitly points out contradictions with playbook

---

## 5. Expected Workflow Changes

### Before Playbook Integration:
1. Agent explores data
2. Agent consults Oracle for strategy
3. Oracle provides generic recommendations
4. Agent writes training code
5. Agent may or may not use best practices

### After Playbook Integration:
1. Agent explores data
2. **Agent reads entire playbook (268 lines of battle-tested strategies)**
3. **Agent identifies competition domain and notes recommended architectures**
4. **Agent consults Oracle with playbook-informed initial plan**
5. Oracle validates plan against playbook and provides refinements
6. **Agent re-reads relevant playbook sections before writing train.py**
7. Agent writes training code using playbook techniques
8. **Oracle validates code against playbook best practices**

---

## 6. Key Strategies Now Available to Agent

### Tabular Competitions
- **Primary approach:** LightGBM/XGBoost/CatBoost (not simple models)
- **Feature engineering:** Heavy manual feature engineering (interactions, polynomials, group aggregations)
- **Hybrid ensembles:** Combine GBDTs with neural networks (GRUs, TabNet)
- **CV strategy:** Stratified K-Fold for classification, Group K-Fold for grouped data

### Computer Vision
- **Architectures:** EfficientNet (B0-B7), ResNeXt, Vision Transformers (ViT/BEiT/Dino)
- **Augmentation:** MixUp, CutMix (beyond basic flips/rotations)
- **Advanced:** Test-Time Augmentation (TTA), CLAHE preprocessing for medical imaging
- **CV strategy:** Stratified K-Fold with proper augmentation in training only

### Natural Language Processing
- **Models:** DeBERTa > RoBERTa > BERT (evolution of performance)
- **Strategy:** Fine-tune pre-trained models (not train from scratch)
- **Advanced:** Knowledge distillation, pseudo-labeling, handling long sequences (Longformer)
- **CV strategy:** Stratified K-Fold to maintain class balance

### Time Series
- **Transform to tabular:** Time-based features, lag features, window features
- **Models:** GBDTs (LightGBM/XGBoost) on engineered features
- **CV strategy:** TimeSeriesSplit (training on past to predict future)

### Common Pitfalls to Avoid
- **Data leakage:** Target leakage (features not available at prediction time)
- **Train-test contamination:** Fitting scalers on entire dataset before split
- **Overfitting to public LB:** Trust local CV, not public leaderboard
- **Wrong CV strategy:** Using K-Fold for time series or grouped data

---

## 7. Benefits

### Strategic Level
- **Battle-tested approaches:** Agent starts with proven winning strategies, not generic baselines
- **Domain expertise:** Agent knows which architectures dominate each domain
- **Avoid common mistakes:** Playbook explicitly warns about pitfalls

### Implementation Level
- **Correct CV strategies:** Agent matches CV type to data structure
- **Advanced techniques:** Agent knows about MixUp/CutMix/TTA/stacking/pseudo-labeling
- **Proper ensembling:** Agent understands diversity principle (not just averaging same model type)

### Time Efficiency
- **Less trial and error:** Agent starts with high-probability approaches
- **Oracle validation:** Oracle catches deviations from playbook best practices
- **Faster convergence:** Using proven architectures reduces wasted experiments

---

## 8. Example Agent Behavior

### Competition: Dog Breed Classification (Computer Vision)

**Old behavior (no playbook):**
1. Agent explores data → sees images
2. Agent consults Oracle → "use a CNN"
3. Agent writes ResNet-18 with basic augmentation
4. Results mediocre, wastes time iterating

**New behavior (with playbook):**
1. Agent explores data → sees images
2. **Agent reads playbook → identifies as Computer Vision competition**
3. **Agent notes:** "Playbook recommends EfficientNet/ResNeXt/ViT, MixUp/CutMix augmentation, TTA"
4. **Agent consults Oracle:** "Playbook recommends EfficientNet-B4 with MixUp. Is this optimal for dog breeds?"
5. Oracle validates → "Yes, EfficientNet-B4 good choice. Use batch_size=128 for A10 GPU."
6. **Agent re-reads CV section before writing train.py**
7. Agent writes train.py with:
   - EfficientNet-B4 (not ResNet-18)
   - MixUp augmentation (not just basic flips)
   - Stratified K-Fold CV (not random split)
   - TTA during inference (not single prediction)
8. Results strong from first attempt

---

## 9. Testing Recommendations

### Verify Playbook Access
```bash
# Inside container
ls -la /home/kaggle_grandmaster_playbook.txt
# Should show: -rw-r--r-- 1 root root [size] [date] /home/kaggle_grandmaster_playbook.txt
```

### Verify Agent Reads Playbook
Check agent logs for:
- "Reading /home/kaggle_grandmaster_playbook.txt"
- "Based on playbook, this is a [domain] competition"
- "Playbook recommends [technique]"

### Verify Oracle References Playbook
Check Oracle responses for:
- "Your playbook-based approach is correct"
- "This contradicts the playbook's recommendation of [X]"
- References to specific playbook sections

---

## 10. Files Modified

1. **`/Users/Yifan/canada-research/mle-bench/environment/Dockerfile`** (Line 82)
   - Added COPY command for playbook file

2. **`/Users/Yifan/canada-research/mle-bench/agents/agent_v5_kaggle/kaggle_agent.py`**
   - Added "KAGGLE GRANDMASTER KNOWLEDGE BASE" section (Lines 66-84)
   - Updated Step 2: "MANDATORY: Read Kaggle Grandmaster Playbook" (Lines 123-129)
   - Updated Step 3: Oracle consultation now references playbook (Lines 131-148)
   - Updated Step 9: Re-read playbook before train.py (Line 181)

3. **`/Users/Yifan/canada-research/agent_v5/tools/oracle.py`**
   - Added "KAGGLE GRANDMASTER KNOWLEDGE BASE" section (Lines 182-191)
   - Oracle now validates against playbook strategies

---

## Summary

**Status:** ✅ **COMPLETE**

The Kaggle Grandmaster Playbook is now fully integrated into the agent system:

- ✅ Playbook file added to Docker environment at `/home/kaggle_grandmaster_playbook.txt`
- ✅ Agent instructed to read playbook BEFORE strategy planning
- ✅ Agent instructed to re-read playbook BEFORE writing train.py
- ✅ Oracle prompt updated to reference and validate against playbook
- ✅ Workflow restructured to incorporate playbook at critical decision points

**Expected Impact:**
- Agent starts with battle-tested strategies instead of generic baselines
- Agent avoids common pitfalls (wrong CV strategy, data leakage, etc.)
- Agent uses domain-appropriate architectures (GBDTs for tabular, ViT for vision, DeBERTa for NLP)
- Oracle validates strategies against playbook best practices
- Faster convergence to winning solutions (less trial and error)

**The agent now has access to hundreds of winning solutions' worth of strategic knowledge! 🏆**
