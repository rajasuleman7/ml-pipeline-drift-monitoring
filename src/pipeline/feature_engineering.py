"""
Feature engineering pipeline.
Builds numeric features, lag features, interaction terms, and encoding.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Applies domain-agnostic feature engineering:
    - Log transform on skewed positive features
    - Polynomial interaction terms for top feature pairs
    - Ratio features between correlated columns
    """

    def __init__(self, log_threshold: float = 2.0,
                 add_interactions: bool = True):
        self.log_threshold    = log_threshold
        self.add_interactions = add_interactions
        self.log_cols_: list  = []
        self.skewness_:  dict = {}

    def fit(self, X: pd.DataFrame, y=None):
        num_cols = X.select_dtypes(include="number").columns
        for col in num_cols:
            if (X[col] > 0).all():
                skew = float(X[col].skew())
                self.skewness_[col] = skew
                if abs(skew) > self.log_threshold:
                    self.log_cols_.append(col)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Log transform skewed features
        for col in self.log_cols_:
            if col in X.columns and (X[col] > 0).all():
                X[f"{col}_log"] = np.log1p(X[col])

        # Interaction terms for first 3 numeric columns
        if self.add_interactions:
            num_cols = X.select_dtypes(include="number").columns.tolist()[:3]
            for i in range(len(num_cols)):
                for j in range(i+1, len(num_cols)):
                    X[f"{num_cols[i]}_x_{num_cols[j]}"] = (
                        X[num_cols[i]] * X[num_cols[j]]
                    )
        return X


def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """Standard preprocessing pipeline for train/test."""
    from sklearn.preprocessing import OrdinalEncoder
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    steps = [("num", num_pipe, num_cols)]
    if cat_cols:
        steps.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(steps)
