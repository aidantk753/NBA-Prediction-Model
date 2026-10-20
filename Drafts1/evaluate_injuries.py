import pandas as pd
import numpy as np
from pathlib import Path

CACHE = Path("nba_cache")
OUTPUT = Path("injury_validation_predictions.csv")
SMOOTHING = 30

PERIODS = [
    ("Jan-Feb", "2024-10-22", "2024-12-31",
     "2025-01-01", "2025-02-28"),
    ("Mar-Apr", "2024-10-22", "2025-02-28",
     "2025-03-01", "2025-04-13"),
]


def gap_category(days):
    if pd.isna(days):
        return "None"
    if days <= 2:
        return "1-2 days"
    if days <= 7:
        return "3-7 days"
    return "8+ days"


def brier(actual, predicted):
    return np.mean(
        (np.asarray(actual) - np.asarray(predicted)) ** 2
    )


def load_appearances():
    appearances = []
    files = list(CACHE.glob("*_2024-25_games.csv"))

    print(f"Reading {len(files)} cached player files...")

    for file in files:
        player_id = int(file.name.split("_")[0])
        logs = pd.read_csv(file)

        if "GAME_DATE" not in logs.columns:
            continue

        dates = pd.to_datetime(
            logs["GAME_DATE"], errors="coerce"
        ).dt.normalize()

        for game_day in dates.dropna().unique():
            appearances.append((player_id, game_day))

    result = pd.DataFrame(
        appearances, columns=["nba_id", "game_day"]
    )

    return result.drop_duplicates()


def prepare_reports():
    reports = pd.read_csv("pregame_injuries_matched.csv")

    reports = reports[
        reports["nba_id"].notna()
        & (reports["reason"] != "Trade Pending")
    ].copy()

    reports["nba_id"] = reports["nba_id"].astype(int)
    reports["game_day"] = pd.to_datetime(
        reports["game_date"], format="%m/%d/%Y"
    ).dt.normalize()

    reports["snapshot_dt"] = pd.to_datetime(
        reports["snapshot_dt"]
    )

    reports = reports.sort_values(
        ["nba_id", "game_day", "snapshot_dt"]
    ).drop_duplicates(
        ["nba_id", "game_day"], keep="last"
    )

    reports = reports.sort_values(
        ["nba_id", "game_day"]
    ).reset_index(drop=True)

    reports["previous_status"] = (
        reports.groupby("nba_id")["status"].shift()
        .fillna("None")
    )

    previous_date = (
        reports.groupby("nba_id")["game_day"].shift()
    )

    reports["gap_category"] = (
        (reports["game_day"] - previous_date)
        .dt.days.apply(gap_category)
    )

    return reports


def build_dataset():
    reports = prepare_reports()
    appearances = load_appearances()

    cached_ids = set(appearances["nba_id"])

    reports = reports[
        reports["nba_id"].isin(cached_ids)
    ].copy()

    reports = reports.merge(
        appearances.assign(played=1),
        on=["nba_id", "game_day"],
        how="left",
        validate="one_to_one"
    )

    reports["played"] = reports["played"].fillna(0).astype(int)

    return reports


def fit_model(training):
    status_stats = training.groupby(
        "status"
    )["played"].agg(["sum", "count"])

    status_probs = (
        (status_stats["sum"] + 1)
        / (status_stats["count"] + 2)
    ).to_dict()

    questionable = training[
        training["status"] == "Questionable"
    ]

    previous_stats = questionable.groupby(
        "previous_status"
    )["played"].agg(["sum", "count"])

    previous_probs = {
        key: (row["sum"] + 1) / (row["count"] + 2)
        for key, row in previous_stats.iterrows()
    }

    history_stats = questionable.groupby(
        ["previous_status", "gap_category"]
    )["played"].agg(["sum", "count"])

    history_probs = {}

    for key, row in history_stats.iterrows():
        prior = previous_probs[key[0]]

        history_probs[key] = (
            row["sum"] + SMOOTHING * prior
        ) / (row["count"] + SMOOTHING)

    return status_probs, previous_probs, history_probs


def evaluate(data, name, train_start, train_end,
             test_start, test_end):

    training = data[
        data["game_day"].between(train_start, train_end)
    ].copy()

    testing = data[
        data["game_day"].between(test_start, test_end)
    ].copy()

    if training.empty or testing.empty:
        raise ValueError(f"No observations for {name}")

    status_probs, previous_probs, history_probs = (
        fit_model(training)
    )

    def predict(row):
        baseline = status_probs.get(row["status"], 0.5)

        if row["status"] != "Questionable":
            return baseline, baseline

        previous = row["previous_status"]
        gap = row["gap_category"]

        prior = previous_probs.get(previous, baseline)

        history = history_probs.get(
            (previous, gap), prior
        )

        return baseline, history

    predictions = testing.apply(
        predict, axis=1, result_type="expand"
    )

    testing["status_probability"] = predictions[0]
    testing["history_probability"] = predictions[1]
    testing["period"] = name

    questionable = testing[
        testing["status"] == "Questionable"
    ]

    print(f"\n========== {name} ==========")
    print(f"Training observations: {len(training)}")
    print(f"Testing observations: {len(testing)}")

    print(
        f"Overall status-only Brier: "
        f"{brier(testing['played'], testing['status_probability']):.5f}"
    )
    print(
        f"Overall history Brier: "
        f"{brier(testing['played'], testing['history_probability']):.5f}"
    )

    if not questionable.empty:
        print(
            f"Questionable status-only Brier: "
            f"{brier(questionable['played'], questionable['status_probability']):.5f}"
        )
        print(
            f"Questionable history Brier: "
            f"{brier(questionable['played'], questionable['history_probability']):.5f}"
        )

    return testing


def main():
    data = build_dataset()
    print(f"Eligible injury observations: {len(data)}")

    results = pd.concat(
        [evaluate(data, *period) for period in PERIODS],
        ignore_index=True
    )

    results["squared_error"] = (
        results["played"] - results["history_probability"]
    ) ** 2

    results["probability_band"] = pd.cut(
        results["history_probability"],
        bins=[0, .2, .4, .6, .8, 1.000001],
        labels=[
            "0-20%", "20-40%", "40-60%",
            "60-80%", "80-100%"
        ],
        include_lowest=True
    )

    print("\n========== COMBINED ==========")
    print(f"Observations: {len(results)}")

    print(
        f"Status-only Brier: "
        f"{brier(results['played'], results['status_probability']):.5f}"
    )
    print(
        f"History Brier: "
        f"{brier(results['played'], results['history_probability']):.5f}"
    )

    print("\n========== CALIBRATION ==========")
    print(
        results.groupby(
            "probability_band", observed=True
        ).agg(
            observations=("played", "size"),
            predicted=("history_probability", "mean"),
            actual=("played", "mean")
        ).to_string()
    )

    print("\n========== BY INJURY STATUS ==========")
    print(
        results.groupby("status").agg(
            observations=("played", "size"),
            predicted=("history_probability", "mean"),
            actual=("played", "mean"),
            brier=("squared_error", "mean")
        ).to_string()
    )

    results.to_csv(OUTPUT, index=False)
    print(f"\nSaved predictions to {OUTPUT}")


if __name__ == "__main__":
    main()
