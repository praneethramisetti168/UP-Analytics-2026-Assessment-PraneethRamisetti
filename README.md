# APS Failure at Scania Trucks — Predictive Maintenance Classifier

**Candidate:** Praneeth Ramisetti  
**Assessment:** UP Analytics 2026 — Round 1 Screener  
**Repository:** `UP-Analytics-2026-Assessment-PraneethRamisetti`

---

## Problem Summary

Scania trucks' Air Pressure System (APS) failures cause costly breakdowns, towing, and delivery delays. This project builds a binary classifier to predict APS failures from 170 anonymized sensor readings, prioritizing **recall** (minimizing missed failures) while managing false positives (unnecessary inspections).

---

## Methodology

### Data
| Split | Rows | APS Failures | Failure Rate |
|-------|------|-------------|-------------|
| Train | 29,618 | 502 | 1.69% |
| Test  | 16,000 | 375 | 2.34% |

Label mapping: `pos` → 1 (APS failure), `neg` → 0 (non-APS failure)

### Preprocessing Pipeline
1. **Missing values**: Parsed `"na"` strings as NaN. Dropped 13 columns with >70% missingness. Median-imputed remaining missing values (robust to outliers and skew).
2. **Missingness indicators**: Added binary flags for columns with 20–70% missing data, encoding the "was this reading available?" signal as a feature.
3. **Row-level aggregates**: Added per-row null count/fraction, mean, std, min, max — capturing sensor health at the observation level.
4. **Scaling**: StandardScaler applied after imputation for models sensitive to scale.

### Validation Strategy
- **Primary split**: Provided train/test CSVs treated as the primary evaluation horizon.
- **Internal validation**: First 80% of training rows used as "earlier data", last 20% as "later data" — simulating time-based ordering without an explicit timestamp column.

### Models Trained
| Model | Type | Imbalance Handling |
|-------|------|-------------------|
| Logistic Regression | Baseline | `class_weight='balanced'` |
| Random Forest | Ensemble | `class_weight='balanced'` |
| XGBoost | Gradient Boosting | `scale_pos_weight` |
| **LightGBM** ✓ | Gradient Boosting | `scale_pos_weight` + early stopping |

**Final model: LightGBM** — chosen for best PR-AUC, speed, and native handling of missing values.

### Threshold Selection
Threshold was selected by maximizing F1 on the validation PR curve, balancing the operational cost asymmetry: **a missed APS failure costs ~10–50× more than an unnecessary inspection**.

---

## Key Findings

| Metric | Value |
|--------|-------|
| **Test PR-AUC** | **0.8933** ← primary metric |
| **Test ROC-AUC** | 0.9936 |
| **Recall @ threshold** | 0.7707 (289 / 375 APS failures caught) |
| **Precision @ threshold** | 0.8653 |
| **F1 Score** | 0.8152 |
| **Optimal threshold** | 0.7147 |
| **True Positives** | 289 |
| **False Negatives** | 86 (missed failures) |
| **False Positives** | 45 (unnecessary inspections) |
| **High-Risk bucket failure rate** | 85.3% actual APS failures in High tier |

- **Top failure drivers**: Sensors in the `ee`, `cr`, `cs`, `ag`, `ay` families showed highest SHAP importance.
- **Risk segmentation**: High-risk trucks (predicted probability ≥ 0.70) had significantly elevated actual failure rates, validating the risk tiering.
- **False negatives concentrate** at probabilities just below the threshold (0.30–0.50), suggesting borderline cases that could benefit from additional sensor monitoring.

---

## Recommendations

1. **Deploy High-risk alert**: Immediately inspect any truck with model probability ≥ 0.70. This high-risk bucket captures the majority of APS failures at manageable inspection volume.
2. **Medium-risk follow-up**: Schedule trucks with probability 0.40–0.70 for inspection within 48 hours — balances coverage with operational load.
3. **Threshold tuning in production**: If inspection capacity is constrained, raise threshold to 0.75. If zero missed failures is critical, lower to 0.50 and accept more false positives.
4. **Data quality**: 169/170 sensor columns had missing values. Improving sensor telemetry reliability (especially the 13 dropped columns) would likely improve recall further.
5. **Temporal monitoring**: Retrain the model quarterly as fleet composition and sensor drift may shift feature distributions over time.

---

## Assumptions & Limitations

- **No timestamp column**: Time-based split was approximated using row ordering. If rows are shuffled in the source, this approximation may not hold.
- **Anonymized features**: Sensor names (e.g., `aa_000`, `cr_000`) are anonymized; domain interpretation of top features is not possible without Scania's data dictionary.
- **Static model**: The classifier is trained offline. Real-time streaming inference would require an additional data pipeline.
- **SMOTE not applied**: To avoid leakage, SMOTE was not used (imbalance handled via `scale_pos_weight`). SMOTE within cross-validation folds could marginally improve recall.
- **MySQL Workbench**: Assessment mentioned MySQL access; the provided data was CSV-only. Database integration was not implemented.

---

## Setup & Reproduction

```bash
# 1. Clone the repo
git clone https://github.com/<your-handle>/UP-Analytics-2026-Assessment-PraneethRamisetti
cd UP-Analytics-2026-Assessment-PraneethRamisetti

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full analysis (generates model artifacts + predictions)
python run_analysis.py

# 4. Launch the API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# → Open http://localhost:8000/docs for interactive Swagger UI

# 5. (Optional) Open the notebook
jupyter notebook notebook.ipynb
```

---

## File Structure

```
├── aps_failure_training_set.csv   ← Training data (29,618 rows)
├── aps_failure_test_set.csv       ← Test data (16,000 rows)
├── notebook.ipynb                 ← Full analysis notebook
├── run_analysis.py                ← Standalone analysis script
├── requirements.txt               ← Pinned dependencies
├── api/
│   ├── main.py                    ← FastAPI application
│   ├── schemas.py                 ← Pydantic request/response models
│   └── model_artifacts/           ← Saved model, preprocessor, metadata
├── outputs/
│   ├── figures/                   ← All visualization PNGs
│   ├── test_predictions.csv       ← Final predictions on test set
│   ├── test_predictions.xlsx      ← Same in Excel format
│   ├── model_comparison.csv       ← Model comparison table
│   └── kpi_summary.csv            ← KPI metrics table
└── screenshots/                   ← API screenshots for submission
```

---

## AI Usage Disclosure

- **GitHub Copilot / AI assistance**: Used for boilerplate code generation and docstring drafting. All modeling decisions, feature engineering logic, threshold selection rationale, and analytical insights reflect the author's own understanding.
- **Libraries**: scikit-learn, LightGBM, XGBoost, SHAP, imbalanced-learn, FastAPI, pandas, numpy, matplotlib, seaborn.
- **References**: 
  - LightGBM docs: https://lightgbm.readthedocs.io
  - SHAP: https://shap.readthedocs.io
  - Scania APS dataset: UCI ML Repository / Kaggle
  - FastAPI docs: https://fastapi.tiangolo.com

---

*Submitted by Praneeth Ramisetti — July 2026*
