"""Tests for feature engineering."""
import sys, os
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "pipeline"))
from feature_engineering import FeatureEngineer, build_preprocessor


def make_df(n=100):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "a": rng.exponential(10, n),
        "b": rng.normal(5, 2, n),
        "c": rng.uniform(0, 1, n),
    })

def test_fit_transform_returns_df():
    fe = FeatureEngineer()
    out = fe.fit_transform(make_df())
    assert isinstance(out, pd.DataFrame)

def test_more_columns_after_fe():
    df = make_df()
    fe = FeatureEngineer(add_interactions=True)
    out = fe.fit_transform(df)
    assert out.shape[1] >= df.shape[1]

def test_log_cols_detected():
    df = make_df()
    fe = FeatureEngineer(log_threshold=0.5)
    fe.fit(df)
    assert len(fe.log_cols_) > 0

def test_no_interactions():
    df = make_df()
    fe = FeatureEngineer(add_interactions=False)
    out = fe.fit_transform(df)
    interaction_cols = [c for c in out.columns if "_x_" in c]
    assert len(interaction_cols) == 0

def test_preprocessor_output_shape():
    df = make_df(100)
    cols = list(df.columns)
    pre = build_preprocessor(cols, [])
    out = pre.fit_transform(df)
    assert out.shape[0] == 100
    assert out.shape[1] == len(cols)
