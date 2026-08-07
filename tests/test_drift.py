"""Tests for drift detection."""
import sys, os
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from monitoring.drift_detector import detect_drift, simulate_drift, DriftReport


def make_ref(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"f{i}": rng.normal(0, 1, n) for i in range(5)})

def test_no_drift_detected():
    ref = make_ref(200, seed=0)
    cur = make_ref(100, seed=1)
    report = detect_drift(ref, cur)
    assert isinstance(report, DriftReport)
    assert not report.dataset_drift   # same distribution, no drift

def test_drift_detected_after_shift():
    ref = make_ref(200, seed=0)
    cur = simulate_drift(make_ref(100, seed=1), ["f0","f1","f2"], shift=5.0)
    report = detect_drift(ref, cur)
    assert len(report.drifted_features) > 0

def test_report_has_counts():
    ref = make_ref(200)
    cur = make_ref(100)
    report = detect_drift(ref, cur)
    assert report.n_reference == 200
    assert report.n_current   == 100

def test_simulate_drift_changes_values():
    df      = make_ref(100)
    drifted = simulate_drift(df, ["f0"], shift=10.0)
    assert not np.allclose(df["f0"].values, drifted["f0"].values)

def test_drift_share_in_range():
    ref = make_ref(200)
    cur = make_ref(100)
    report = detect_drift(ref, cur)
    assert 0.0 <= report.drift_share <= 1.0

def test_to_dict_serialisable():
    import json
    ref = make_ref(100)
    cur = make_ref(50)
    report = detect_drift(ref, cur)
    json.dumps(report.to_dict())
