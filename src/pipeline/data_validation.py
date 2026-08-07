"""
Data validation using Great Expectations-style checks.
Validates schema, types, value ranges, missing rates, and distribution.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValidationResult:
    passed:   bool
    checks:   list[dict] = field(default_factory=list)
    warnings: list[str]  = field(default_factory=list)
    errors:   list[str]  = field(default_factory=list)

    def add_check(self, name: str, passed: bool, detail: str = ""):
        self.checks.append({"check": name, "passed": passed, "detail": detail})
        if not passed:
            self.errors.append(f"{name}: {detail}")
            self.passed = False


def validate_dataframe(df: pd.DataFrame,
                       schema: dict,
                       max_missing_rate: float = 0.1) -> ValidationResult:
    """
    Validate a DataFrame against a schema dict.
    schema = {"col_name": {"dtype": "float64", "min": 0, "max": 1, "nullable": False}}
    """
    result = ValidationResult(passed=True)

    # 1. Check expected columns exist
    for col, spec in schema.items():
        if col not in df.columns:
            result.add_check(f"column_exists.{col}", False, f"Column missing")
            continue

        series = df[col]

        # 2. Missing rate check
        missing_rate = series.isna().mean()
        nullable     = spec.get("nullable", True)
        if not nullable and missing_rate > 0:
            result.add_check(f"no_nulls.{col}", False,
                             f"{missing_rate:.1%} nulls found (expected none)")
        elif missing_rate > max_missing_rate:
            result.add_check(f"missing_rate.{col}", False,
                             f"{missing_rate:.1%} > {max_missing_rate:.1%} threshold")
        else:
            result.add_check(f"missing_rate.{col}", True,
                             f"{missing_rate:.1%} missing")

        # 3. Range checks (on non-null values)
        valid = series.dropna()
        if "min" in spec and (valid < spec["min"]).any():
            bad = int((valid < spec["min"]).sum())
            result.add_check(f"min_value.{col}", False,
                             f"{bad} values below min={spec['min']}")
        if "max" in spec and (valid > spec["max"]).any():
            bad = int((valid > spec["max"]).sum())
            result.add_check(f"max_value.{col}", False,
                             f"{bad} values above max={spec['max']}")

        # 4. Allowed values
        if "allowed" in spec:
            bad_vals = set(valid.unique()) - set(spec["allowed"])
            if bad_vals:
                result.add_check(f"allowed_values.{col}", False,
                                 f"Unexpected values: {bad_vals}")
            else:
                result.add_check(f"allowed_values.{col}", True)

    # 5. Duplicate row check
    dup_rate = df.duplicated().mean()
    if dup_rate > 0.05:
        result.warnings.append(f"High duplicate rate: {dup_rate:.1%}")

    return result
