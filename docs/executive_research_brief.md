# High-Value Prescriber Profile: Executive Research Brief

**Portfolio diversity drives high-value prescriber status. Top 10% prescribe 2.5× more drugs (7.4 vs 3.0) and generate 64.8% of $77.5B revenue.**

---

## THE ANSWER

### Top 3 Findings

**1. Portfolio Diversity is the Primary Driver** (d=2.0 [LARGE], OR=6.67)

- Top 10%: 7.4 unique drugs | Bottom 90%: 3.0 unique drugs
- Largest effect in study. Each +1 SD (≈2 drugs) = **6.7× higher odds** of high-value status
- High-value prescribers are **generalists**, not specialists

**2. Clinical Complexity Matters** (d=0.53 [MEDIUM], OR=0.75 [necessary])

- Top 10%: 5.0 conditions treated | Bottom 90%: 3.8 conditions
- High-value prescribers manage sicker patients with comorbidities
- Low complexity **predicts low-value status** (necessary characteristic)

**3. Declining Growth Paradox** (d=-0.70 [MEDIUM], OR=0.78)

- Top 10%: -1.97 Rx/month decline | Bottom 90%: -0.36 Rx/month
- High-value prescribers **declining 5× faster** (retention risk)
- Suggests market saturation, competitive pressure, or lifecycle effects

### Model Performance

- **ROC-AUC: 0.92** (excellent discrimination) | **Recall: 85%** (captures 85% of high-value) | **Precision: 39%** (requires secondary validation)
- Sample: 61,091 prescribers, $77.5B revenue, Jan 2023-Sep 2025

---

## COMPLETE FEATURE RANKING

**All 17 characteristics ranked by predictive importance (absolute coefficient):**

| Rank | Feature | Effect Size | Odds Ratio | Type | Category | Interpretation |
|------|---------|-------------|-----------|------|----------|----------------|
| **1** | **unique_drugs** | **d=2.0 [LARGE]** | **6.67** | **Sufficient** | Portfolio | **Primary driver** - portfolio generalists |
| 2 | medicaid_rate | d=0.19 | 2.05 | Sufficient | Payer Mix | High Medicaid = high-volume practice |
| 3 | unique_procedures | N/A | 1.87 | Sufficient | Clinical | Broad procedural repertoire |
| 4 | commercial_rate | d=0.04 | 1.85 | Sufficient | Payer Mix | Higher commercial proportion |
| 5 | medicare_rate | d=-0.08 | 1.60 | Sufficient | Payer Mix | Higher Medicare proportion |
| 6 | unique_products_paid | N/A | 0.68 | Necessary | Financial | Pharma payment diversity |
| 7 | avg_claim_amount | d=0.42 [SMALL] | 1.39 | Sufficient | Financial | Higher Rx cost ($8,446 vs $6,325) |
| 8 | unique_conditions | **d=0.53 [MEDIUM]** | 0.75 | Necessary | Clinical | **Comorbidity mgmt** (5.0 vs 3.8) |
| 9 | payment_count | N/A | 1.30 | Sufficient | Financial | More pharma transactions |
| 10 | growth_rate_monthly | **d=-0.70 [MEDIUM]** | 0.78 | Necessary | Temporal | **Declining faster** (-1.97 vs -0.36) |
| 11 | avg_days_supply | N/A | 0.83 | Necessary | Temporal | Days supply per Rx |
| 12 | supply_90day_rate | d=0.001 | 0.85 | Necessary | Temporal | 90-day refills (no difference) |
| 13 | procedure_rate | d=0.09 | 0.87 | Necessary | Clinical | Procedural vs diagnostic |
| 14 | total_payments | N/A | 0.94 | Weak | Financial | Total pharma payment amount |
| 15 | certification_count | d=0.11 | 0.94 | Weak | Credentials | Board certifications (weak) |
| 16 | receives_payments | N/A | 1.02 | None | Financial | Binary payment receipt (negligible) |
| 17 | brand_rate | N/A | 1.00 | None | Portfolio | 100% both groups (no variance) |

**Legend:**

- [LARGE] effect (d>=0.8) | [MEDIUM] (d>=0.5) | [SMALL] (d>=0.2) | N/A = Negligible (d<0.2)
- **Sufficient:** Presence predicts high-value (OR>1) | **Necessary:** Absence predicts low-value (OR<1)

---

## EVIDENCE

### Segment Comparison

|  | **Top 10%** | **Bottom 90%** | **Ratio** |
|---|------------|---------------|----------|
| **Prescribers** | 6,110 | 54,981 | — |
| **Total Revenue** | $50.2B (64.8%) | $27.3B (35.2%) | — |
| **Mean Revenue** | $8.2M | $497K | **16.5×** |
| **Median Revenue** | $4.8M | $253K | 19.2× |
| **Mean Prescriptions** | 1,078 | 85 | 12.7× |
| **Mean Unique Drugs** | 7.4 | 3.0 | **2.5×** |
| **Revenue per Rx** | $8,446 | $6,325 | 1.3× |
| **Threshold (90th %ile)** | ≥$2.67M | — | — |

### Logistic Regression: Sufficient vs Necessary

**SUFFICIENT Characteristics** (presence → high-value):

1. **unique_drugs (OR=6.67)** - Each +1 SD = 6.7× odds increase. **Strongest predictor.**
   - Example: Prescriber 2 SD above mean (≈7.6 drugs) has **45× higher odds** than mean
2. **medicaid_rate (OR=2.05)** - High Medicaid proportion doubles odds (indicates high-volume practice)
3. **unique_procedures (OR=1.87)** - Broader procedural repertoire increases odds 87%
4. **commercial_rate (OR=1.85)** - Higher commercial payer mix increases odds 85%
5. **medicare_rate (OR=1.60)** - Higher Medicare proportion increases odds 60%

**NECESSARY Characteristics** (absence → low-value):

1. **unique_products_paid (OR=0.68)** - Low pharma payment diversity **reduces odds 32%**
2. **unique_conditions (OR=0.75)** - Low clinical complexity reduces odds 25%
3. **growth_rate_monthly (OR=0.78)** - Faster decline reduces odds 22%
4. **avg_days_supply (OR=0.83)** - Lower days supply reduces odds 17%
5. **supply_90day_rate (OR=0.85)** - Lower 90-day usage reduces odds 15%

**NOT PREDICTIVE:**

- **brand_rate:** 100% in both groups (no variance, OR=1.0)
- **receives_payments:** Negligible effect (OR=1.02, only 2% increase)
- **certification_count:** Weak (OR=0.94)
- **total_payments:** Weak negative (OR=0.94)

### Statistical Validation

- [x] **FDR correction:** Benjamini-Hochberg (α=0.05) applied to 17 tests - 16 passed
- [x] **Large sample:** n=61,091 ensures adequate statistical power
- [x] **Train/test split:** 80/20 (42,914 train, 10,729 test) with standardized features
- [x] **Effect sizes reported:** Cohen's d, Cramér's V, odds ratios (not just p-values)
- [x] **Model validated:** ROC-AUC 0.92, Confusion matrix: 84.7% recall (1,005/1,187 true positives)

**Confusion Matrix:**

|  | Predicted Bottom 90% | Predicted Top 10% |
|---|---------------------|------------------|
| **Actual Bottom 90%** | 7,947 (83%) | 1,595 (17%) |
| **Actual Top 10%** | 182 (15%) | 1,005 (85%) |

---

## ACTION

### What To Do

**Immediate - Target These Prescribers:**

1. **Portfolio generalists:** ≥5 unique drugs prescribed (2 SD above mean = 7.6 drugs ideal)
2. **High Medicaid rate:** ≥25% Medicaid proportion (signals high-volume practice)
3. **Clinical complexity:** ≥4 unique conditions treated (comorbidity management)

**Short-Term - Investigate Urgently:**

1. **Why are high-value prescribers declining faster?** (-1.97 vs -0.36 Rx/month)
   - Competitive dynamics? Patient churn? Market saturation? Lifecycle effects?
   - **Action:** Longitudinal cohort analysis stratified by prescriber tenure
   - **Impact:** Retention > acquisition (high-value segment at risk)

2. **Payment-product alignment matters (but weakly)**
   - Top 10%: 24.6% alignment | Bottom 90%: 13.9% alignment (d=0.36, small effect)
   - Don't over-invest in payment strategies (OR≈1.0 for binary receipt)

**Avoid:**

- [x] **Payment-based targeting:** Receives_payments OR=1.02 (negligible)
- [x] **Credential screening:** Certifications d=0.11 (negligible effect)
- [x] **Specialty stratification:** 99% missing data (can't use)
- [x] **One-dimensional approach:** 17 characteristics, not one magic metric

### What This Doesn't Answer

**Causal Questions:**

- Does portfolio diversity **cause** high-value status, or do high-volume practices naturally accumulate diverse portfolios? (Observational design, can't infer causality)

**Multi-Tier Segmentation:**

- Analysis is binary (top 10% vs bottom 90%). Need **reverse clustering** for high/medium/low tiers and optimal cutoffs.

**Temporal Stability:**

- Cross-sectional snapshot (Jan 2023-Sep 2025). Need **longitudinal validation** to assess stability over time.

**Generalizability:**

- Dataset: Specialty biologics (rheumatology/dermatology) in U.S. market
- May not generalize to other therapeutic areas (oncology, cardiology) or geographies

### Next Steps

**1. Complete Task 2: Reverse Clustering** (high/medium/low value discrimination)

- Work backward from value tiers to find optimal feature combinations
- Compare forward vs reverse clustering approaches
- Develop tier-specific targeting strategies

**2. Longitudinal Analysis: Address Declining Growth**

- Cohort analysis stratified by prescriber tenure (control for lifecycle effects)
- Competitive dynamics assessment (share erosion vs market maturity)
- Retention program design for high-value segment

**3. External Validation**

- Replicate findings in independent dataset (other therapeutic areas, time periods, geographies)
- Assess temporal stability (train on 2023 data, validate on 2024-2025)

**4. Advanced Modeling**

- Non-linear models (gradient boosting, random forest) to capture interactions
- Feature interactions (e.g., unique_drugs × medicaid_rate)
- Address class imbalance (SMOTE, class weights) to improve precision

---

## CONTEXT

### Methods (One Paragraph)

Analyzed 61,091 prescribers generating $77.5B from 4 datasets (rx_claims, med_claims, provider_payments, providers_bio), Jan 2023-Sep 2025. Top 10% defined as ≥$2.67M revenue (90th percentile). Engineered 17 features across 5 domains (portfolio, financial, clinical, payer mix, temporal). Statistical tests: Mann-Whitney U for continuous variables, chi-square for categorical, Pearson/Spearman for correlations, with Benjamini-Hochberg FDR correction (α=0.05). Effect sizes: Cohen's d (small=0.2, medium=0.5, large=0.8), Cramér's V, odds ratios. Logistic regression: 80/20 train-test split, standardized features (z-scores), ROC-AUC 0.92. Full methodology: `docs/statistical_validation_report.md`

### Data Quality Limitations

**Critical Gaps:**

- **Specialty 99% missing:** Can't stratify by specialty ("Unknown" = 99.6% top 10%, 96.3% bottom 90%)
- **No patient IDs:** Couldn't analyze multi-drug regimen complexity per patient
- **Branded drugs only:** 100% brand rate (zero variance, uninformative)

**Sample Coverage:**

- 86% matched with medical claims (52,791/61,091) for condition complexity analysis
- Pharma payment data: 89.6% top 10%, 81.5% bottom 90% receive payments

**Temporal Scope:**

- Single time window (Jan 2023-Sep 2025)
- Cross-sectional design limits causal inference
- Growth rates calculated via linear regression on monthly Rx counts

---

## KEY TAKEAWAYS

**For Quick Readers (30-second scan):**

1. **Portfolio diversity (7.4 vs 3.0 drugs, OR=6.67)** is the strongest predictor
2. **Top 10% generate 64.8% of revenue** with 16× higher mean revenue ($8.2M vs $497K)
3. **High-value prescribers declining faster** (-1.97 vs -0.36 Rx/month) → retention urgency
4. **Model achieves 0.92 AUC** with 85% recall (excellent screening tool)
5. **Target generalists** with ≥5 drugs + high Medicaid + ≥4 conditions treated

**For Deep Readers:**

- See full statistical validation report: `docs/statistical_validation_report.md` (60+ pages)
- All 17 features with tests, effect sizes, FDR corrections, confusion matrices, appendices

---

**Document Version:** 1.0 | **Date:** October 3, 2025 | **Team:** Canada Research Project

**Full Report:** `docs/statistical_validation_report.md` | **Notebook:** `research/high_value_prescriber_analysis.ipynb`

**Contact:** canada-research@example.com
