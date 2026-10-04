"""Train credit-risk baselines and report holdout classification metrics."""

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split

from credit_risk.data import TARGET, engineer_features, validate_applications
from credit_risk.models import build_models, select_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/credit_applications.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.1 <= args.test_size <= 0.5:
        parser.error("--test-size must be between 0.1 and 0.5")
    if not args.data.exists():
        parser.error(f"Dataset not found: {args.data}. Run generate_sample_data.py first.")

    applications = validate_applications(pd.read_csv(args.data), require_target=True)
    engineered = engineer_features(applications)
    x = select_features(engineered)
    y = applications[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(exist_ok=True)
    metrics = []
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(13, 4))

    for axis, (name, model) in zip(axes_cm, build_models(args.seed).items()):
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        metrics.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1_score": f1_score(y_test, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probabilities),
            }
        )
        RocCurveDisplay.from_predictions(y_test, probabilities, name=name, ax=ax_roc)
        ConfusionMatrixDisplay.from_predictions(
            y_test, predictions, display_labels=["No default", "Default"], ax=axis, colorbar=False
        )
        axis.set_title(name.replace("_", " ").title())
        joblib.dump(model, args.output_dir / "models" / f"{name}.joblib")

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax_roc.set_title("ROC curves (stratified holdout)")
    fig_roc.tight_layout()
    fig_roc.savefig(args.output_dir / "roc_curves.png", dpi=160)
    plt.close(fig_roc)
    fig_cm.tight_layout()
    fig_cm.savefig(args.output_dir / "confusion_matrices.png", dpi=160)
    plt.close(fig_cm)

    results = pd.DataFrame(metrics).sort_values("roc_auc", ascending=False)
    results.to_csv(args.output_dir / "metrics.csv", index=False)
    print(f"Holdout metrics saved to {args.output_dir / 'metrics.csv'}")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Plots and fitted models saved under {args.output_dir}")
    print("Metrics describe this dataset only; they are not a lending approval recommendation.")


if __name__ == "__main__":
    main()
