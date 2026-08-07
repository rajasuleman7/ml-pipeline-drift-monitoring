"""
Drift monitoring runner — generates drift reports on new data batches.

Usage:
    python src/monitoring/monitor.py --batch_size 200 --shift 2.0
    python src/monitoring/monitor.py  # no drift
"""

import os
import sys
import json
import argparse
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))

from pipeline.train import load_data
from monitoring.drift_detector import detect_drift, simulate_drift


def run_monitoring(batch_size: int = 200,
                   shift: float = 0.0,
                   output_dir: str = "data"):
    print("=" * 55)
    print("  DRIFT MONITORING REPORT")
    print("=" * 55)

    df = load_data(n_samples=2000, seed=42)
    reference = df.drop(columns=["target"]).iloc[:1000]
    current   = df.drop(columns=["target"]).iloc[1000:1000 + batch_size]

    if shift > 0:
        drift_cols = [f"feature_{i}" for i in range(5)]
        current = simulate_drift(current, cols=drift_cols, shift=shift)
        print(f"  Simulated drift on: {drift_cols} (shift={shift})")

    report = detect_drift(reference, current)

    print(f"
  Method:          {report.method}")
    print(f"  Reference size:  {report.n_reference}")
    print(f"  Current size:    {report.n_current}")
    print(f"  Drifted features:{len(report.drifted_features)}")
    print(f"  Drift share:     {report.drift_share:.1%}")
    print(f"  Dataset drift:   {'YES ⚠' if report.dataset_drift else 'NO ✓'}")

    if report.drifted_features:
        print(f"
  Drifted columns:")
        for col in report.drifted_features[:10]:
            print(f"    - {col}")

    if report.quality_issues:
        print(f"
  Quality issues:")
        for issue in report.quality_issues:
            print(f"    - {issue}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "drift_report.json")
    with open(out_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"
  Report saved: {out_path}")

    if report.dataset_drift:
        print("
  ACTION REQUIRED: Significant drift detected.")
        print("  Consider retraining the model on recent data.")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int,   default=200)
    p.add_argument("--shift",      type=float, default=0.0,
                   help="Simulate covariate shift of this magnitude")
    args = p.parse_args()
    run_monitoring(batch_size=args.batch_size, shift=args.shift)
