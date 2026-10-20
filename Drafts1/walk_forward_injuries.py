import pandas as pd
import numpy as np

from evaluate_injuries import build_dataset
from compare_injury_models import reason_category

SMOOTHING = 30

WINDOWS = [
    ("Jan", "2024-12-31", "2025-01-01", "2025-01-31"),
    ("Feb", "2025-01-31", "2025-02-01", "2025-02-28"),
    ("Mar", "2025-02-28", "2025-03-01", "2025-03-31"),
    ("Apr", "2025-03-31", "2025-04-01", "2025-04-13"),
]


def fit_predict(train, test):
    status_stats = train.groupby("status")["played"].agg(
        ["sum", "count"]
    )

    status_probs = (
        (status_stats["sum"] + 1) /
        (status_stats["count"] + 2)
    ).to_dict()

    reason_stats = train.groupby(
        ["status", "reason_category"]
    )["played"].agg(["sum", "count"])

    reason_probs = {}

    for (status, reason), row in reason_stats.iterrows():
        parent = status_probs[status]

        reason_probs[(status, reason)] = (
            row["sum"] + SMOOTHING * parent
        ) / (row["count"] + SMOOTHING)

    questionable = train[
        train["status"] == "Questionable"
    ]

    history_stats = questionable.groupby(
        ["status", "reason_category", "previous_status", "gap_category"],
        dropna=False
    )["played"].agg(["sum", "count"])

    history_probs = {}

    for key, row in history_stats.iterrows():
        parent = reason_probs.get(
            (key[0], key[1]),
            status_probs[key[0]]
        )

        history_probs[key] = (
            row["sum"] + SMOOTHING * parent
        ) / (row["count"] + SMOOTHING)

    predictions = []

    for row in test.itertuples():
        status = row.status
        reason = row.reason_category

        probability = reason_probs.get(
            (status, reason),
            status_probs.get(status, 0.5)
        )

        if status == "Questionable":
            key = (
                status,
                reason,
                row.previous_status,
                row.gap_category
            )

            probability = history_probs.get(
                key, probability
            )

        predictions.append(float(probability))

    return predictions


def main():
    data = build_dataset()
    data["game_day"] = pd.to_datetime(data["game_day"])
    data["reason_category"] = data["reason"].apply(reason_category)

    results = []

    for period, train_end, test_start, test_end in WINDOWS:
        train = data[data["game_day"] <= train_end].copy()

        test = data[
            (data["game_day"] >= test_start) &
            (data["game_day"] <= test_end)
        ].copy()

        if train.empty or test.empty:
            print(f"{period}: insufficient data")
            continue

        test["probability"] = fit_predict(train, test)
        test["period"] = period

        results.append(test)

        brier = np.mean(
            (test["played"] - test["probability"]) ** 2
        )

        q = test[test["status"] == "Questionable"]

        print(f"\n========== {period} ==========")
        print("Training observations:", len(train))
        print("Evaluation observations:", len(test))
        print(f"Overall Brier: {brier:.5f}")

        if not q.empty:
            q_brier = np.mean(
                (q["played"] - q["probability"]) ** 2
            )
            print("Questionable observations:", len(q))
            print(f"Questionable Brier: {q_brier:.5f}")

    if not results:
        print("No evaluation results.")
        return

    combined = pd.concat(results, ignore_index=True)

    overall = np.mean(
        (combined["played"] - combined["probability"]) ** 2
    )

    q = combined[
        combined["status"] == "Questionable"
    ]

    print("\n========== FINAL WALK-FORWARD RESULTS ==========")
    print("Observations:", len(combined))
    print("Unique players:", combined["nba_id"].nunique())
    print(f"Overall Brier: {overall:.5f}")

    if not q.empty:
        print("Questionable observations:", len(q))
        print(
            "Questionable Brier:",
            f"{np.mean((q['played'] - q['probability']) ** 2):.5f}"
        )

    combined.to_csv(
        "injury_walk_forward_predictions.csv",
        index=False
    )

    print("\nSaved injury_walk_forward_predictions.csv")


if __name__ == "__main__":
    main()
