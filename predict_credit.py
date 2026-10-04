"""Score unlabeled German Credit-format rows with a trained pipeline."""

import argparse
from pathlib import Path

import joblib

from credit_risk.german import engineer_german_features, load_german_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Whitespace-delimited file with 20 attributes")
    parser.add_argument("--model", type=Path, required=True, help="Path to a German model .joblib file")
    parser.add_argument("--output", type=Path, default=Path("artifacts/german_credit/predictions.csv"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not args.input.exists() or not args.model.exists():
        parser.error("Both --input and --model must point to existing files.")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    applications = load_german_data(args.input, require_target=False)
    model = joblib.load(args.model)
    probabilities = model.predict_proba(engineer_german_features(applications))[:, 1]
    output = applications.copy()
    output["estimated_bad_credit_probability"] = probabilities
    output["model_flag_at_threshold"] = (probabilities >= args.threshold).astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} scored rows to {args.output}")
    print("Model flags are an educational output, not a credit decision.")


if __name__ == "__main__":
    main()
