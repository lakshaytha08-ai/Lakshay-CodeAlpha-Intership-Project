# Creditworthiness Classification

A small Python project that compares Logistic Regression, Decision Tree, and Random Forest classifiers using financial-history features. It reports accuracy, precision, recall, F1-score, and ROC-AUC on a stratified holdout set, and saves ROC/confusion-matrix plots and fitted pipelines.

> **Important:** The included data is synthetic and exists only to demonstrate the workflow. This project is not validated for lending or other consequential decisions. Do not use its scores to approve, deny, price, or rank real applicants. Real credit models require appropriate legal review, representative data, privacy controls, calibration, fairness analysis, explainability, and ongoing monitoring.

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Generate demo data and train

```powershell
python generate_sample_data.py
python train.py
```

The generated CSV is at `data/credit_applications.csv`. Training saves holdout metrics to `artifacts/metrics.csv`, plots to `artifacts/`, and model pipelines to `artifacts/models/`. The test partition is held out from fitting and preprocessing. All three models are reported; the test set is not used to select a deployment model.

## Score applications

The input CSV must contain these raw numeric columns: `annual_income`, `monthly_debt`, `monthly_expenses`, `credit_limit`, `current_balance`, `on_time_payments`, `late_payments`, and `account_age_months`. The target column `defaulted` is only required for training, with `0` meaning no observed default and `1` meaning default.

For a quick demo, create a second CSV from unlabeled demo rows and score it:

```powershell
python -c "import pandas as pd; d=pd.read_csv('data/credit_applications.csv'); d.drop(columns='defaulted').head(10).to_csv('data/unlabeled_demo.csv', index=False)"
python predict.py --input data/unlabeled_demo.csv --model artifacts/models/random_forest.joblib
```

The output contains an estimated default probability and a configurable model flag (default threshold `0.5`). The flag is only a technical demonstration, not a credit decision.

## Features and evaluation

Feature engineering adds debt-to-income ratio, credit utilization, late-payment rate, on-time-payment rate, and monthly cashflow ratio. Median imputation and (for Logistic Regression) standardization are fitted inside each pipeline using training rows only. The holdout metrics are useful for learning model evaluation; they do not establish real-world performance.

## Train on Statlog German Credit

The original `german.data` file contains 20 whitespace-delimited attributes and a class (`1` = good, `2` = bad). The project maps bad credit to the positive class, one-hot encodes categorical attributes, scales numeric attributes, and adds a credit-amount-per-duration-month feature. The dataset documentation specifies an asymmetric error cost: labeling a bad-credit case as good costs 5, while labeling a good-credit case as bad costs 1. The evaluation reports this cost alongside classification metrics.

```powershell
python train_german_credit.py --data "path\to\german.data"
```

The model pipelines, `metrics.csv`, and plots are saved separately under `artifacts/german_credit/`. To score a whitespace-delimited file containing 20 unlabeled attributes:

```powershell
python predict_german_credit.py --input "path\to\unlabeled.data" --model artifacts/german_credit/models/random_forest.joblib
```

This dataset has only 1,000 historical cases. Its results are a classroom benchmark, not evidence of current population performance or a model suitable for real credit decisions.

Run the tests with:

```powershell
python -m pytest
```
