"""Train and evaluate classifiers on the Statlog German Credit data."""

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

from credit_risk.german import (
    GERMAN_TARGET_COLUMN,
    build_german_models,
    engineer_german_features,
    load_german_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to the original german.data file")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/german_credit"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.data.exists():
        parser.error(f"Dataset not found: {args.data}")
    if not 0.1 <= args.test_size <= 0.5:
        parser.error("--test-size must be between 0.1 and 0.5")

    applications = load_german_data(args.data)
    features = engineer_german_features(applications)
    target = applications[GERMAN_TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=args.test_size, random_state=args.seed, stratify=target
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = args.output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    metrics = []
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(13, 4))

    for axis, (name, model) in zip(axes_cm, build_german_models(args.seed).items()):
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        false_negatives = ((y_test == 1) & (predictions == 0)).sum()
        false_positives = ((y_test == 0) & (predictions == 1)).sum()
        total_error_cost = 5 * false_negatives + false_positives
        metrics.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision_bad_credit": precision_score(y_test, predictions, zero_division=0),
                "recall_bad_credit": recall_score(y_test, predictions, zero_division=0),
                "f1_bad_credit": f1_score(y_test, predictions, zero_division=0),
                "roc_auc_bad_credit": roc_auc_score(y_test, probabilities),
                "total_error_cost_5_to_1": int(total_error_cost),
                "mean_error_cost_per_case": total_error_cost / len(y_test),
            }
        )
        RocCurveDisplay.from_predictions(y_test, probabilities, name=name, ax=ax_roc)
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            display_labels=["Good credit", "Bad credit"],
            ax=axis,
            colorbar=False,
        )
        axis.set_title(name.replace("_", " ").title())
        joblib.dump(model, models_dir / f"{name}.joblib")

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax_roc.set_title("ROC curves (stratified holdout)")
    fig_roc.tight_layout()
    fig_roc.savefig(args.output_dir / "roc_curves.png", dpi=160)
    plt.close(fig_roc)
    fig_cm.tight_layout()
    fig_cm.savefig(args.output_dir / "confusion_matrices.png", dpi=160)
    plt.close(fig_cm)

    results = pd.DataFrame(metrics).sort_values("mean_error_cost_per_case")
    results.to_csv(args.output_dir / "metrics.csv", index=False)
    print(f"Holdout metrics saved to {args.output_dir / 'metrics.csv'}")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Fitted pipelines and plots saved under {args.output_dir}")
    print("This small benchmark is educational; it is not a validated lending model.")


if __name__ == "__main__":
    main()
