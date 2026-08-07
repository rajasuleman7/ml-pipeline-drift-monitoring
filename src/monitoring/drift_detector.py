"""
Data drift detection using Evidently.
Monitors feature distributions, target drift, and data quality on new batches.
Falls back to KS-test-based detection if Evidently is unavailable.
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
    HAS_EVIDENTLY = True
except ImportError:
    HAS_EVIDENTLY = False

from scipy import stats as scipy_stats


@dataclass
class DriftReport:
    timestamp:        str
    n_reference:      int
    n_current:        int
    drifted_features: list[str] = field(default_factory=list)
    drift_scores:     dict      = field(default_factory=dict)
    dataset_drift:    bool      = False
    drift_share:      float     = 0.0
    quality_issues:   list[str] = field(default_factory=list)
    method:           str       = "ks_test"

    def to_dict(self) -> dict:
        return {
            "timestamp":        self.timestamp,
            "n_reference":      self.n_reference,
            "n_current":        self.n_current,
            "drifted_features": self.drifted_features,
            "drift_share":      round(self.drift_share, 3),
            "dataset_drift":    self.dataset_drift,
            "quality_issues":   self.quality_issues,
            "method":           self.method,
        }


def detect_drift(reference: pd.DataFrame,
                 current:   pd.DataFrame,
                 target_col: Optional[str] = None,
                 drift_threshold: float = 0.05,
                 report_path: Optional[str] = None) -> DriftReport:
    """
    Detect data drift between reference and current datasets.
    Uses Evidently if available; falls back to KS-test.
    """
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()

    if HAS_EVIDENTLY:
        return _evidently_drift(reference, current, target_col,
                                drift_threshold, ts, report_path)
    return _ks_drift(reference, current, drift_threshold, ts)


def _evidently_drift(ref, cur, target_col, threshold, ts, report_path) -> DriftReport:
    metrics = [DataDriftPreset(), DataQualityPreset()]
    if target_col and target_col in ref.columns and target_col in cur.columns:
        metrics.append(TargetDriftPreset())

    report = Report(metrics=metrics)
    report.run(reference_data=ref, current_data=cur)
    result_dict = report.as_dict()

    # Parse Evidently results
    drifted, scores = [], {}
    try:
        drift_data = result_dict["metrics"][0]["result"]
        for col, info in drift_data.get("drift_by_columns", {}).items():
            p_val = info.get("drift_score", 1.0)
            scores[col] = p_val
            if info.get("drift_detected", False):
                drifted.append(col)
    except (KeyError, IndexError):
        pass

    if report_path:
        report.save_html(report_path)

    share = len(drifted) / max(len(ref.columns), 1)
    return DriftReport(
        timestamp=ts, n_reference=len(ref), n_current=len(cur),
        drifted_features=drifted, drift_scores=scores,
        dataset_drift=share > 0.3, drift_share=share, method="evidently",
    )


def _ks_drift(ref: pd.DataFrame, cur: pd.DataFrame,
              threshold: float, ts: str) -> DriftReport:
    """KS-test fallback drift detection."""
    drifted, scores = [], {}
    num_cols = ref.select_dtypes(include="number").columns

    for col in num_cols:
        if col not in cur.columns:
            continue
        stat, p_val = scipy_stats.ks_2samp(
            ref[col].dropna().values,
            cur[col].dropna().values,
        )
        scores[col] = round(float(p_val), 6)
        if p_val < threshold:
            drifted.append(col)

    share = len(drifted) / max(len(num_cols), 1)
    return DriftReport(
        timestamp=ts, n_reference=len(ref), n_current=len(cur),
        drifted_features=drifted, drift_scores=scores,
        dataset_drift=share > 0.3, drift_share=share, method="ks_test",
    )


def simulate_drift(df: pd.DataFrame, cols: list[str],
                   shift: float = 2.0, seed: int = 1) -> pd.DataFrame:
    """Simulate covariate shift on specified columns for testing."""
    rng     = np.random.default_rng(seed)
    drifted = df.copy()
    for col in cols:
        if col in drifted.select_dtypes(include="number").columns:
            drifted[col] = drifted[col] + rng.normal(shift, 0.5, len(drifted))
    return drifted
