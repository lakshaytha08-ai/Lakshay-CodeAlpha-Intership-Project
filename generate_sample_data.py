"""Create a deterministic, synthetic dataset for a local project demo."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_sample_data(rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    annual_income = np.clip(rng.lognormal(10.75, 0.55, rows), 18000, 250000)
    monthly_income = annual_income / 12
    monthly_expenses = monthly_income * rng.uniform(0.35, 0.82, rows)
    monthly_debt = monthly_income * rng.beta(1.7, 5.0, rows) * 1.25
    credit_limit = rng.uniform(1500, 30000, rows)
    current_balance = credit_limit * rng.beta(1.6, 2.8, rows) * 1.15
    account_age_months = rng.integers(6, 241, rows)
    payment_history_length = np.maximum(6, np.minimum(account_age_months, 60))

    debt_to_income = monthly_debt / monthly_income
    utilization = current_balance / credit_limit
    payment_late_probability = np.clip(
        0.025 + 0.22 * debt_to_income + 0.10 * np.maximum(utilization - 0.5, 0), 0.02, 0.48
    )
    late_payments = rng.binomial(payment_history_length, payment_late_probability)
    on_time_payments = payment_history_length - late_payments
    cashflow_ratio = (monthly_income - monthly_expenses) / monthly_income

    default_log_odds = (
        -2.15
        + 2.2 * debt_to_income
        + 1.45 * np.maximum(utilization - 0.55, 0)
        + 3.0 * (late_payments / payment_history_length)
        - 1.1 * cashflow_ratio
        - 0.0025 * account_age_months
    )
    default_probability = 1 / (1 + np.exp(-default_log_odds))
    defaulted = rng.binomial(1, default_probability)

    return pd.DataFrame(
        {
            "annual_income": np.round(annual_income, 2),
            "monthly_debt": np.round(monthly_debt, 2),
            "monthly_expenses": np.round(monthly_expenses, 2),
            "credit_limit": np.round(credit_limit, 2),
            "current_balance": np.round(current_balance, 2),
            "on_time_payments": on_time_payments,
            "late_payments": late_payments,
            "account_age_months": account_age_months,
            "defaulted": defaulted,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/credit_applications.csv"))
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rows < 20:
        parser.error("--rows must be at least 20")
    data = make_sample_data(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Wrote {len(data)} synthetic rows to {args.output}")
    print(f"Default rate: {data['defaulted'].mean():.1%}")


if __name__ == "__main__":
    main()
