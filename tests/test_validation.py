"""Tests for data validation."""
import sys, os
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "pipeline"))
from data_validation import validate_dataframe, ValidationResult


SCHEMA = {
    "age":    {"dtype": "float64", "min": 0, "max": 120, "nullable": False},
    "income": {"dtype": "float64", "min": 0},
    "label":  {"dtype": "int64", "allowed": [0, 1], "nullable": False},
}

def make_df(**kwargs):
    base = {"age": [25, 35, 45], "income": [50000, 60000, 70000], "label": [0, 1, 0]}
    base.update(kwargs)
    return pd.DataFrame(base)

def test_valid_df_passes():
    result = validate_dataframe(make_df(), SCHEMA)
    assert result.passed

def test_missing_column_fails():
    df = make_df().drop(columns=["age"])
    result = validate_dataframe(df, SCHEMA)
    assert not result.passed
    assert any("age" in e for e in result.errors)

def test_value_below_min_fails():
    df = make_df(age=[-1, 35, 45])
    result = validate_dataframe(df, SCHEMA)
    assert not result.passed

def test_disallowed_label_fails():
    df = make_df(label=[0, 1, 2])
    result = validate_dataframe(df, SCHEMA)
    assert not result.passed

def test_high_missing_rate_fails():
    df = make_df(age=[None, None, None])
    result = validate_dataframe(df, SCHEMA, max_missing_rate=0.1)
    assert not result.passed

def test_checks_populated():
    result = validate_dataframe(make_df(), SCHEMA)
    assert len(result.checks) > 0
