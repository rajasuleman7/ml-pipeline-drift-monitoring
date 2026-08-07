# End-to-End ML Pipeline with Drift Monitoring

A fully reproducible ML pipeline covering **data validation → feature engineering → Optuna hyperparameter tuning → SHAP explainability**, with an **Evidently-powered drift monitoring** layer that detects covariate shift on new data batches using KS-test fallback when needed.

---

## Features

### Pipeline
- **Data validation** — schema checks, missing rate thresholds, value range validation, allowed-value enforcement
- **Feature engineering** — log-transform detection on skewed features, polynomial interaction terms, configurable via scikit-learn `BaseEstimator`
- **Model comparison** — Gradient Boosting, Random Forest, Logistic Regression benchmarked
- **Optuna tuning** — TPE sampler with configurable trial budget; tunes `n_estimators`, `max_depth`, `learning_rate`, `subsample` per model type
- **SHAP explainability** — `TreeExplainer` for tree models with Gini importance fallback

### Drift Monitoring
- **Evidently integration** — `DataDriftPreset`, `DataQualityPreset`, `TargetDriftPreset` out of the box
- **KS-test fallback** — `scipy.stats.ks_2samp` per-feature when Evidently is unavailable
- **Drift simulation** — `simulate_drift()` injects controlled covariate shift for testing
- **Drift report** — JSON summary with drifted features, drift share, and action recommendation
- **17 unit tests** — validation, feature engineering, and drift detection all covered

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| ML | scikit-learn 1.3+ |
| Hyperparameter Tuning | Optuna 3.5+ (TPE sampler) |
| Explainability | SHAP (`TreeExplainer`) |
| Drift Detection | Evidently + scipy KS-test fallback |
| Data | pandas, numpy |
| Testing | pytest (17 tests) |

---

## Project Structure

```
ml-pipeline-drift-monitoring/
├── src/
│   ├── pipeline/
│   │   ├── data_validation.py      # Schema checks, missing rates, range/allowed-value validation
│   │   ├── feature_engineering.py  # FeatureEngineer transformer + preprocessor builder
│   │   └── train.py                # 5-stage pipeline: validate → FE → Optuna → eval → SHAP
│   └── monitoring/
│       ├── drift_detector.py       # Evidently + KS-test drift detection, simulate_drift()
│       └── monitor.py              # CLI drift monitoring runner
├── tests/
│   ├── test_validation.py          # 6 data validation tests
│   ├── test_feature_engineering.py # 5 feature engineering tests
│   └── test_drift.py               # 6 drift detection tests
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/rajasuleman7/ml-pipeline-drift-monitoring.git
cd ml-pipeline-drift-monitoring
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Running

### Full training pipeline

```bash
cd src/pipeline
python train.py                  # 20 Optuna trials per model
python train.py --n_trials 50    # more thorough search
python train.py --quick          # 5 trials (fast demo)
```

### Drift monitoring on a new batch

```bash
cd src/monitoring
python monitor.py                         # no drift
python monitor.py --shift 2.0             # simulate mild covariate shift
python monitor.py --shift 5.0             # simulate severe drift
```

### Tests

```bash
pytest tests/ -v
```

---

## Sample Pipeline Output

```
============================================================
  END-TO-END ML PIPELINE WITH DRIFT MONITORING
============================================================

[1/5] Data validation...
  Validation PASSED — 63 checks, 0 errors

[2/5] Feature engineering...
  Original features: 20  →  After FE: 24
  Log-transformed: ['feature_3', 'feature_7']

[3/5] Optuna hyperparameter tuning (20 trials × 3 models)...
  gradient_boosting         best_auc=0.9712  trials=20
  random_forest             best_auc=0.9688  trials=20
  logistic_regression       best_auc=0.9142  trials=20

  Best: gradient_boosting  auc=0.9712

[4/5] Training best model and evaluating...
  accuracy=0.9275  f1=0.9264  roc_auc=0.9801

[5/5] SHAP feature importance...
  Method: shap
  proc_3               0.1842  ████████████████████████████
  proc_7               0.1631  █████████████████████████
  proc_0               0.1124  █████████████████
```

---

## Sample Drift Report

```json
{
  "timestamp": "2024-11-15T10:30:00Z",
  "n_reference": 1000,
  "n_current": 200,
  "drifted_features": ["feature_0", "feature_1", "feature_2"],
  "drift_share": 0.6,
  "dataset_drift": true,
  "method": "ks_test"
}
```
