import pandas as pd
import numpy as np
from evaluate_injuries import build_dataset, PERIODS

SMOOTHING = 30
OUTPUT = "injury_model_comparison.csv"


def reason_category(reason):
    reason = str(reason).strip()
    if reason.startswith("Injury/Illness"):
        return "Injury/Illness"
    if reason.startswith("G League"):
        return "G League"
    if reason in ("-", "nan", ""):
        return "Other/Unspecified"
    return "Other"


def brier(actual, predicted):
    return float(np.mean(
        (np.asarray(actual) - np.asarray(predicted)) ** 2
    ))


def evaluate(data, name, train_start, train_end, test_start, test_end):
    training = data[
        data["game_day"].between(train_start, train_end)
    ].copy()

    testing = data[
        data["game_day"].between(test_start, test_end)
    ].copy()

    if training.empty or testing.empty:
        raise ValueError(f"Missing observations for {name}")

    status_stats = training.groupby("status")["played"].agg(
        ["sum", "count"]
    )

    status_probs = (
        (status_stats["sum"] + 1)
        / (status_stats["count"] + 2)
    ).to_dict()

    # Fit the status + reason model.
    reason_stats = training.groupby(
        ["status", "reason_category"]
    )["played"].agg(["sum", "count"])

    reason_probs = {}

    for (status, reason), row in reason_stats.iterrows():
        prior = status_probs[status]

        reason_probs[(status, reason)] = (
            row["sum"] + SMOOTHING * prior
        ) / (row["count"] + SMOOTHING)

    testing["reason_probability"] = [
        reason_probs.get(
            (row.status, row.reason_category),
            status_probs.get(row.status, 0.5)
        )
        for row in testing.itertuples()
    ]

    return testing[[
        "nba_id", "game_day", "status", "reason_category",
        "played", "reason_probability"
    ]].assign(period=name)


def main():
    data = build_dataset()
    data["reason_category"] = data["reason"].apply(reason_category)

    results = pd.concat(
        [evaluate(data, *period) for period in PERIODS],
        ignore_index=True
    )

    previous = pd.read_csv("injury_validation_predictions.csv")
    previous["game_day"] = pd.to_datetime(previous["game_day"])

    previous = previous[[
        "nba_id", "game_day", "period",
        "played", "status_probability", "history_probability"
    ]].copy()

    results = results.merge(
        previous,
        on=["nba_id", "game_day", "period"],
        how="inner",
        suffixes=("", "_previous"),
        validate="one_to_one"
    )

    if len(results) != len(previous):
        raise ValueError(
            f"Prediction alignment failed: "
            f"{len(results)} matched, {len(previous)} expected"
        )

    if not (
        results["played"] == results["played_previous"]
    ).all():
        raise ValueError("Appearance labels do not match")

    print("\n========== THREE-MODEL COMPARISON ==========")

    for period, subset in [
        ("Jan-Feb", results[results["period"] == "Jan-Feb"]),
        ("Mar-Apr", results[results["period"] == "Mar-Apr"]),
        ("Combined", results)
    ]:
        print(f"\n{period} ({len(subset)} observations)")
        print(
            f"Status only:     "
            f"{brier(subset['played'], subset['status_probability']):.5f}"
        )
        print(
            f"History + gap:   "
            f"{brier(subset['played'], subset['history_probability']):.5f}"
        )
        print(
            f"Status + reason: "
            f"{brier(subset['played'], subset['reason_probability']):.5f}"
        )

    print("\n========== BY INJURY STATUS ==========")

    for status, subset in results.groupby("status"):
        print(f"\n{status} ({len(subset)} observations)")
        print(
            f"Status only:     "
            f"{brier(subset['played'], subset['status_probability']):.5f}"
        )
        print(
            f"History + gap:   "
            f"{brier(subset['played'], subset['history_probability']):.5f}"
        )
        print(
            f"Status + reason: "
            f"{brier(subset['played'], subset['reason_probability']):.5f}"
        )

    results.to_csv(OUTPUT, index=False)
    print(f"\nSaved comparison to {OUTPUT}")


if __name__ == "__main__":
    main()
