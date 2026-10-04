"""Score a CSV with a previously trained model pipeline."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from credit_risk.data import engineer_features, validate_applications
from credit_risk.models import select_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV containing the raw financial features")
    parser.add_argument("--model", type=Path, required=True, help="Path to a .joblib model from artifacts/models")
    parser.add_argument("--output", type=Path, default=Path("artifacts/predictions.csv"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not args.input.exists() or not args.model.exists():
        parser.error("Both --input and --model must point to existing files.")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    raw = validate_applications(pd.read_csv(args.input))
    model = joblib.load(args.model)
    probabilities = model.predict_proba(select_features(engineer_features(raw)))[:, 1]
    output = raw.copy()
    output["estimated_default_probability"] = probabilities
    output["model_flag_at_threshold"] = (probabilities >= args.threshold).astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} scored rows to {args.output}")
    print("Model flags are educational outputs, not credit decisions.")


if __name__ == "__main__":
    main()
