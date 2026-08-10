"""Train and persist the baseline models from a labelled CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from complaint_intelligence.modeling import train_models
from complaint_intelligence.sample_data import generate_sample_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, help="CSV with text, topic, sentiment and urgency columns")
    parser.add_argument("--output", type=Path, default=Path("models/baseline.joblib"))
    args = parser.parse_args()
    df = pd.read_csv(args.data) if args.data else generate_sample_data("data/complaints.csv")
    bundle = train_models(df)
    bundle.save(args.output)
    print(f"Saved {len(bundle.models)} models to {args.output}")
    for target, scores in bundle.metrics.items():
        print(f"{target:10s} macro-F1={scores['macro_f1']:.3f} accuracy={scores['accuracy']:.3f}")


if __name__ == "__main__":
    main()

