# High-Value Prescriber Profiling Analysis

Comprehensive statistical analysis identifying characteristics that distinguish top 10% revenue-generating prescribers from the bottom 90%.

## Key Findings

- **Sample**: 61,091 prescribers generating $77.5B total revenue
- **Top 10% Threshold**: ≥$2.67M annual revenue (6,110 prescribers, 64.8% of revenue)
- **Primary Driver**: Portfolio diversity (7.4 vs 3.0 unique drugs, Cohen's d=2.0, OR=6.67)
- **Predictive Performance**: ROC-AUC 0.92, Recall 85%
- **17 Characteristics Validated**: Across 5 domains with rigorous statistical testing

**Quick Read**: See [Executive Research Brief](docs/executive_research_brief.md) (3 pages)
**Full Report**: See [Statistical Validation Report](docs/statistical_validation_report.md) (60+ pages)

---

## Repository Structure

```
canada-research/
├── README.md                                    # This file
├── research/
│   └── high_value_prescriber_analysis.ipynb    # Main analysis notebook (78 cells)
├── outputs/
│   ├── timeseries_prescriber_monthly.parquet   # Time-series dataset
│   └── modeling_dataset.csv                     # Feature matrix for modeling
├── docs/
│   ├── executive_research_brief.md              # 3-page executive summary
│   ├── statistical_validation_report.md         # Full research paper (60+ pages)
│   ├── methodology.md                           # Data extraction & statistical framework
│   ├── data_specifications.md                   # Table schemas & SQL queries
│   ├── assumptions_and_limitations.md           # Study constraints & caveats
│   └── feature_definitions.md                   # Complete feature reference
├── streamlit_app/                               # Interactive dashboard (separate deliverable)
│   ├── app.py                                   # Main Streamlit application
│   ├── utils/                                   # Data loading, charting, metrics
│   └── requirements.txt                         # Python dependencies
└── modeling_dataset_preparation.ipynb           # Feature engineering pipeline
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Access to BigQuery project: `unique-bonbon-472921-q8`
- Jupyter Notebook
- Required packages:
  ```bash
  pip install polars>=0.19 scipy>=1.11 scikit-learn>=1.3 matplotlib seaborn jupyter
  ```

### Running the Analysis

1. **Clone repository:**
   ```bash
   git clone https://github.com/[username]/canada-research.git
   cd canada-research
   ```

2. **Open main analysis notebook:**
   ```bash
   jupyter notebook research/high_value_prescriber_analysis.ipynb
   ```

3. **Run cells sequentially** (Sections 0-7):
   - Section 0: Setup & imports
   - Section 1: Define top 10% revenue generators
   - Sections 2-5: Comparative analysis (portfolio, financial, clinical, geographic)
   - Section 6: Advanced characteristics (velocity, refill patterns, complexity)
   - Section 7: Logistic regression (necessary vs sufficient features)

4. **View outputs:**
   - Statistical test results printed inline
   - Visualizations displayed in notebook
   - Summary tables with effect sizes
   - Logistic regression coefficients & odds ratios

### Streamlit Dashboard (Optional)

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

**Features**: Individual prescriber dashboard, comparative analytics, data explorer

---

## Data Sources

Analysis integrates 4 healthcare datasets from BigQuery:

| Dataset | Description | Key Fields | Records |
|---------|-------------|------------|---------|
| **rx_claims** | Prescription claims | NPI, NDC, total_paid_amt, service_date, payer | ~50M claims |
| **med_claims** | Medical claims | NPI, diagnosis_code, procedure_code | ~800K claims |
| **provider_payments** | Pharma payments (Open Payments) | NPI, payment_amount, associated_product | ~5M payments |
| **providers_bio** | Provider biographical data | NPI, certifications, education, awards | ~800K providers |

**Time Range**: January 2023 - September 2025

**Linkage**: All datasets linked via prescriber NPI (National Provider Identifier)

---

## Methodology Summary

**Design**: Cross-sectional comparative analysis with predictive modeling

**Segmentation**: 90th percentile revenue threshold ($2.67M)
- Top 10%: n=6,110 (≥$2.67M revenue)
- Bottom 90%: n=54,981 (<$2.67M revenue)

**Features**: 17 characteristics across 5 domains
1. **Portfolio** (3): unique_drugs, brand_rate, revenue_per_rx
2. **Financial** (4): receives_payments, total_payments, payment_count, unique_products_paid
3. **Clinical** (7): unique_conditions, unique_procedures, procedure_rate, certifications, education, awards, memberships
4. **Payer Mix** (3): commercial_rate, medicare_rate, medicaid_rate
5. **Temporal** (4): growth_rate_monthly, supply_90day_rate, avg_days_supply, prescription_count

**Statistical Tests**:
- Continuous: Mann-Whitney U (non-parametric, appropriate for skewed distributions)
- Categorical: Chi-square test of independence
- Correlations: Pearson (linear) & Spearman (monotonic)
- **Multiple comparison correction**: Benjamini-Hochberg FDR (α=0.05)
- **Effect sizes**: Cohen's d, Cramér's V, odds ratios

**Predictive Model**:
- Algorithm: Logistic regression (L2 regularization)
- Train/test split: 80/20 (42,914 train, 10,729 test)
- Feature standardization: Z-scores (mean=0, std=1)
- Performance: ROC-AUC 0.92, Recall 85%, Precision 39%

**See full methodology**: [docs/methodology.md](docs/methodology.md)

---

## Key Results

### Top 3 Findings

1. **Portfolio Diversity is Primary Driver** (Cohen's d=2.0, OR=6.67)
   - Top 10%: 7.4 unique drugs | Bottom 90%: 3.0 drugs
   - Largest effect in study (large magnitude)
   - Each +1 SD = 6.7× higher odds of high-value status

2. **Clinical Complexity Matters** (Cohen's d=0.53, OR=0.75)
   - Top 10%: 5.0 conditions treated | Bottom 90%: 3.8 conditions
   - Medium effect size
   - Necessary characteristic (low complexity predicts low-value)

3. **Declining Growth Paradox** (Cohen's d=-0.70, OR=0.78)
   - Top 10%: -1.97 Rx/month decline | Bottom 90%: -0.36 Rx/month
   - High-value prescribers declining 5× faster
   - Medium effect (retention urgency)

### Logistic Regression: Sufficient vs Necessary

**SUFFICIENT** (presence → high-value):
1. unique_drugs (OR=6.67) - Strongest predictor
2. medicaid_rate (OR=2.05)
3. unique_procedures (OR=1.87)
4. commercial_rate (OR=1.85)
5. medicare_rate (OR=1.60)

**NECESSARY** (absence → low-value):
1. unique_products_paid (OR=0.68)
2. unique_conditions (OR=0.75)
3. growth_rate_monthly (OR=0.78)
4. avg_days_supply (OR=0.83)
5. supply_90day_rate (OR=0.85)

### Revenue Concentration

- **Top 10%** (6,110 prescribers): $50.2B (64.8% of total), mean $8.2M, median $4.8M
- **Bottom 90%** (54,981 prescribers): $27.3B (35.2% of total), mean $497K, median $253K
- **Revenue ratio**: 16.5× higher mean revenue in top 10%

---

## Documentation

| Document | Description | Length | Audience |
|----------|-------------|--------|----------|
| [Executive Research Brief](docs/executive_research_brief.md) | Ultra-terse 3-page summary | 3 pages | Busy technical peers |
| [Statistical Validation Report](docs/statistical_validation_report.md) | Full academic-style research paper | 60+ pages | Researchers, stakeholders |
| [Methodology](docs/methodology.md) | Data extraction & statistical framework | 10 pages | Analysts replicating work |
| [Data Specifications](docs/data_specifications.md) | Table schemas & SQL queries | 8 pages | Data engineers |
| [Assumptions & Limitations](docs/assumptions_and_limitations.md) | Study constraints & caveats | 5 pages | Critical reviewers |
| [Feature Definitions](docs/feature_definitions.md) | Complete feature reference | 6 pages | Feature engineers |

---

## Reproducibility

### Code Availability

- **Main notebook**: `research/high_value_prescriber_analysis.ipynb` (78 cells, fully documented)
- **Feature engineering**: `modeling_dataset_preparation.ipynb` (9-step pipeline)
- **Dashboard**: `streamlit_app/` (modular architecture, 5 files)

### Data Access

- BigQuery project: `unique-bonbon-472921-q8`
- Tables: `HCP.rx_claims`, `HCP.med_claims`, `HCP.provider_payments`, `HCP.providers_bio`
- **Contact**: canada-research@example.com for data sharing agreements

### Software Environment

```
Python 3.11+
polars==0.19+
scipy==1.11+
scikit-learn==1.3+
matplotlib==3.7+
seaborn==0.12+
jupyter
```

### Replication Steps

1. Query BigQuery tables using SQL from `docs/data_specifications.md`
2. Run feature engineering pipeline: `modeling_dataset_preparation.ipynb`
3. Execute analysis notebook: `research/high_value_prescriber_analysis.ipynb`
4. Validate results match reported effect sizes (±5% tolerance for sampling variation)

**Expected runtime**: ~30 minutes (BigQuery queries ~15min, analysis ~15min)

---

## Limitations

### Data Quality

- **Specialty 99% missing**: Can't stratify by clinical specialty
- **No patient IDs**: Couldn't analyze multi-drug regimen complexity per patient
- **Branded drugs only**: 100% brand rate (zero variance)

### Study Design

- **Cross-sectional**: Single time window (Jan 2023-Sep 2025), no causal inference
- **Observational**: Can't establish causality (e.g., does diversity cause high-value or vice versa?)
- **Single therapeutic area**: Specialty biologics (rheumatology/dermatology) may not generalize

### Model

- **Class imbalance**: 10% positive class reduces precision (39%)
- **No external validation**: Validated on 20% holdout from same dataset
- **Linear assumptions**: Logistic regression may miss non-linear interactions

**See full limitations**: [docs/assumptions_and_limitations.md](docs/assumptions_and_limitations.md)

---

## Future Work

### Immediate Next Steps

1. **Complete Task 2: Reverse Clustering Algorithm**
   - Discriminate high/medium/low value tiers (not just binary)
   - Identify optimal feature combinations for classification
   - Compare forward vs reverse clustering approaches

2. **Investigate Declining Growth Paradox**
   - Longitudinal cohort analysis stratified by prescriber tenure
   - Competitive dynamics assessment (market maturation vs share erosion)
   - Retention program design

3. **External Validation**
   - Replicate findings in other therapeutic areas (oncology, cardiology)
   - Temporal validation (train 2023, test 2024-2025)
   - Geographic validation (other regions/countries)

### Advanced Analytics

4. **Non-linear Models**
   - Gradient boosting, random forests to capture interactions
   - Feature interactions (unique_drugs × medicaid_rate)
   - Address class imbalance (SMOTE, class weights)

5. **Causal Inference**
   - Propensity score matching to estimate treatment effects
   - Instrumental variables for payment impact
   - Difference-in-differences for policy changes

6. **Patient-Level Analysis**
   - Acquire patient-identifiable data
   - Analyze regimen complexity, persistence, adherence
   - Patient-level outcomes (hospitalization, costs)

---

## Citation

If using this work, please cite:

```
High-Value Prescriber Profiling Analysis (2025)
Canada Research Project
https://github.com/[username]/canada-research
```

**For academic use**: See [Statistical Validation Report](docs/statistical_validation_report.md) for full methodology and references.

---

## Contact

- **Project**: Canada Research - High-Value Prescriber Analytics
- **Email**: canada-research@example.com
- **Repository**: https://github.com/[username]/canada-research
- **Issues**: https://github.com/[username]/canada-research/issues

---

## License

[Specify license - MIT, Apache 2.0, or Proprietary]

---

## Acknowledgments

- Data Engineering Team (BigQuery infrastructure)
- Analytics Team (statistical consultation)
- [Add other acknowledgments]

---

**Last Updated**: October 3, 2025 | **Version**: 1.0
