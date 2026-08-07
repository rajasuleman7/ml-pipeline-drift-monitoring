"""
Model training with Optuna hyperparameter tuning and SHAP explainability.

Usage:
    python src/pipeline/train.py
    python src/pipeline/train.py --n_trials 30 --quick
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              classification_report)
from sklearn.pipeline import Pipeline
import joblib
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from feature_engineering import build_preprocessor, FeatureEngineer
from data_validation import validate_dataframe


SCHEMA = {
    f"feature_{i}": {"dtype": "float64", "nullable": True}
    for i in range(20)
}
SCHEMA["target"] = {"dtype": "int64", "allowed": [0, 1], "nullable": False}


def load_data(n_samples: int = 2000, seed: int = 42):
    X, y = make_classification(
        n_samples=n_samples, n_features=20, n_informative=12,
        n_redundant=4, n_classes=2, class_sep=1.2, random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    df["target"] = y
    return df


def make_optuna_objective(X_tr, y_tr, X_val, y_val, model_type: str):
    def objective(trial: optuna.Trial) -> float:
        if model_type == "gradient_boosting":
            params = {
                "n_estimators":   trial.suggest_int("n_estimators", 50, 300),
                "max_depth":      trial.suggest_int("max_depth", 2, 6),
                "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":      trial.suggest_float("subsample", 0.6, 1.0),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            }
            model = GradientBoostingClassifier(**params, random_state=42)
        elif model_type == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth":    trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            }
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        else:
            params = {"C": trial.suggest_float("C", 0.001, 100, log=True)}
            model = LogisticRegression(**params, max_iter=1000, random_state=42)

        model.fit(X_tr, y_tr)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    return objective


def explain_with_shap(model, X_val: np.ndarray, feature_names: list) -> dict:
    """Compute SHAP feature importance."""
    try:
        import shap
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_val[:100])
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        importance = {
            name: round(float(val), 6)
            for name, val in sorted(
                zip(feature_names, mean_abs),
                key=lambda x: x[1], reverse=True
            )
        }
        return {"method": "shap", "importance": importance}
    except Exception as e:
        # Fallback to permutation importance
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            importance = {
                name: round(float(val), 6)
                for name, val in sorted(
                    zip(feature_names, imp),
                    key=lambda x: x[1], reverse=True
                )
            }
            return {"method": "gini_importance", "importance": importance}
        return {"method": "unavailable", "importance": {}}


def train(n_trials: int = 20, seed: int = 42):
    print("=" * 60)
    print("  END-TO-END ML PIPELINE WITH DRIFT MONITORING")
    print("=" * 60)

    # ── 1. Load & validate ───────────────────────────────────
    print("
[1/5] Data validation...")
    df = load_data(seed=seed)
    val_result = validate_dataframe(df, SCHEMA)
    status = "PASSED" if val_result.passed else "FAILED"
    print(f"  Validation {status} — {len(val_result.checks)} checks, "
          f"{len(val_result.errors)} errors")
    if not val_result.passed:
        for err in val_result.errors[:5]:
            print(f"  ERROR: {err}")

    # ── 2. Feature engineering ───────────────────────────────
    print("
[2/5] Feature engineering...")
    X = df.drop(columns=["target"])
    y = df["target"].values

    fe = FeatureEngineer(log_threshold=1.5, add_interactions=True)
    X  = fe.fit_transform(X)
    print(f"  Original features: {df.shape[1]-1}  →  After FE: {X.shape[1]}")
    print(f"  Log-transformed: {fe.log_cols_}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=seed, stratify=y_train)

    num_cols = X.select_dtypes(include="number").columns.tolist()
    pre      = build_preprocessor(num_cols, [])
    X_tr_p   = pre.fit_transform(X_train)
    X_val_p  = pre.transform(X_val)
    X_te_p   = pre.transform(X_test)

    # ── 3. Optuna tuning ─────────────────────────────────────
    print(f"
[3/5] Optuna hyperparameter tuning ({n_trials} trials × 3 models)...")
    best_score, best_model_name, best_params = 0.0, "", {}

    for mtype in ["gradient_boosting", "random_forest", "logistic_regression"]:
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(make_optuna_objective(X_tr_p, y_train, X_val_p, y_val, mtype),
                       n_trials=n_trials, show_progress_bar=False)
        print(f"  {mtype:<25} best_auc={study.best_value:.4f}  "
              f"trials={len(study.trials)}")
        if study.best_value > best_score:
            best_score      = study.best_value
            best_model_name = mtype
            best_params     = study.best_params

    print(f"
  Best: {best_model_name}  auc={best_score:.4f}")

    # ── 4. Final model + evaluation ──────────────────────────
    print("
[4/5] Training best model and evaluating...")
    if best_model_name == "gradient_boosting":
        final_model = GradientBoostingClassifier(**best_params, random_state=seed)
    elif best_model_name == "random_forest":
        final_model = RandomForestClassifier(**best_params, random_state=seed, n_jobs=-1)
    else:
        final_model = LogisticRegression(**best_params, max_iter=1000, random_state=seed)

    final_model.fit(X_tr_p, y_train)

    preds    = final_model.predict(X_te_p)
    probas   = final_model.predict_proba(X_te_p)[:, 1]
    metrics  = {
        "accuracy":  round(accuracy_score(y_test, preds),   4),
        "f1":        round(f1_score(y_test, preds),         4),
        "roc_auc":   round(roc_auc_score(y_test, probas),   4),
    }
    print(f"  accuracy={metrics['accuracy']}  "
          f"f1={metrics['f1']}  roc_auc={metrics['roc_auc']}")
    print("
  Classification Report:")
    print(classification_report(y_test, preds, indent=4))

    # ── 5. SHAP explainability ───────────────────────────────
    print("[5/5] SHAP feature importance...")
    feature_names = [f"proc_{i}" for i in range(X_tr_p.shape[1])]
    shap_result   = explain_with_shap(final_model, X_te_p, feature_names)
    top5 = list(shap_result["importance"].items())[:5]
    print(f"  Method: {shap_result['method']}")
    for name, imp in top5:
        bar = "█" * int(imp * 300)
        print(f"  {name:<20} {imp:.4f}  {bar}")

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": final_model, "preprocessor": pre,
                 "feature_engineer": fe}, "models/pipeline.pkl")

    run_log = {
        "model":       best_model_name,
        "best_params": best_params,
        "metrics":     metrics,
        "shap":        shap_result,
        "n_features":  X.shape[1],
        "n_train":     len(X_train),
    }
    with open("models/run_log.json", "w") as f:
        json.dump(run_log, f, indent=2)

    print("
Pipeline complete. Artifacts saved to models/")
    return run_log


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--quick",    action="store_true")
    args = p.parse_args()
    train(n_trials=5 if args.quick else args.n_trials, seed=args.seed)
