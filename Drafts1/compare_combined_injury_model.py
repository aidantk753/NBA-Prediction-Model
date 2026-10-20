import numpy as np
import pandas as pd

from evaluate_injuries import build_dataset, PERIODS
from compare_injury_models import reason_category

SMOOTHING = 30
OUTPUT = "injury_combined_comparison.csv"


def brier(actual, predicted):
    return float(np.mean(
        (np.asarray(actual) - np.asarray(predicted)) ** 2
    ))


def evaluate(data, name, train_start, train_end, test_start, test_end):
    train = data[
        data["game_day"].between(train_start, train_end)
    ].copy()

    test = data[
        data["game_day"].between(test_start, test_end)
    ].copy()

    # Parent model: status + reason
    status_stats = train.groupby("status")["played"].agg(
        ["sum", "count"]
    )

    status_probs = (
        (status_stats["sum"] + 1)
        / (status_stats["count"] + 2)
    ).to_dict()

    reason_stats = train.groupby(
        ["status", "reason_category"]
    )["played"].agg(["sum", "count"])

    reason_probs = {}

    for key, row in reason_stats.iterrows():
        prior = status_probs[key[0]]
        reason_probs[key] = (
            row["sum"] + SMOOTHING * prior
        ) / (row["count"] + SMOOTHING)

    # Combined model: add previous status and gap.
    history_stats = train.groupby(
        ["status", "reason_category",
         "previous_status", "gap_category"]
    )["played"].agg(["sum", "count"])

    combined_probs = {}

    for key, row in history_stats.iterrows():
        parent_key = (key[0], key[1])
        prior = reason_probs[parent_key]

        combined_probs[key] = (
            row["sum"] + SMOOTHING * prior
        ) / (row["count"] + SMOOTHING)

    def predict(row):
        parent_key = (row.status, row.reason_category)
        parent = reason_probs.get(
            parent_key, status_probs.get(row.status, 0.5)
        )

        combined_key = (
            row.status, row.reason_category,
            row.previous_status, row.gap_category
        )

        combined = combined_probs.get(combined_key, parent)
        return parent, combined

    predictions = test.apply(
        predict, axis=1, result_type="expand"
    )

    test["reason_probability"] = predictions[0]
    test["combined_probability"] = predictions[1]
    test["period"] = name

    return test


def main():
    data = build_dataset()
    data["reason_category"] = data["reason"].apply(reason_category)

    results = pd.concat(
        [evaluate(data, *period) for period in PERIODS],
        ignore_index=True
    )

    previous = pd.read_csv("injury_model_comparison.csv")
    previous["game_day"] = pd.to_datetime(previous["game_day"])

    # Verify that the reason-only model matches the prior experiment.
    matched = results.merge(
        previous[[
            "nba_id", "game_day", "period", "reason_probability"
        ]],
        on=["nba_id", "game_day", "period"],
        suffixes=("", "_previous"),
        validate="one_to_one"
    )

    if len(matched) != len(previous):
        raise ValueError("Prediction counts do not match")

    if not np.allclose(
        matched["reason_probability"],
        matched["reason_probability_previous"]
    ):
        raise ValueError("Reason-only predictions do not match")

    print("\n========== MODEL COMPARISON ==========")

    for period, subset in [
        ("Jan-Feb", results[results["period"] == "Jan-Feb"]),
        ("Mar-Apr", results[results["period"] == "Mar-Apr"]),
        ("Combined", results)
    ]:
        reason_score = brier(
            subset["played"], subset["reason_probability"]
        )
        combined_score = brier(
            subset["played"], subset["combined_probability"]
        )

        print(f"\n{period}: {len(subset)} observations")
        print(f"Status + reason:    {reason_score:.5f}")
        print(f"Combined model:     {combined_score:.5f}")
        print(
            f"Relative improvement: "
            f"{100 * (reason_score - combined_score) / reason_score:.2f}%"
        )

    print("\n========== BY STATUS ==========")

    for status, subset in results.groupby("status"):
        print(f"\n{status}: {len(subset)} observations")
        print(
            "Reason-only Brier: "
            f"{brier(subset['played'], subset['reason_probability']):.5f}"
        )
        print(
            "Combined Brier:    "
            f"{brier(subset['played'], subset['combined_probability']):.5f}"
        )

    results.to_csv(OUTPUT, index=False)
    print(f"\nSaved results to {OUTPUT}")


if __name__ == "__main__":
    main()
