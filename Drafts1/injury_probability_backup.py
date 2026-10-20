import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
TRAINING_END = pd.Timestamp("2025-04-13")


def gap_category(days):
    if days is None or pd.isna(days):
        return "None"
    if days <= 2:
        return "1-2 days"
    if days <= 7:
        return "3-7 days"
    return "8+ days"


def load_probabilities():
    status = pd.read_csv(
        MODEL_DIR / "injury_status_probabilities.csv"
    ).set_index("status")["probability"].to_dict()

    previous = pd.read_csv(
        MODEL_DIR / "injury_previous_status_probabilities.csv"
    ).set_index("previous_status")["probability"].to_dict()

    history = pd.read_csv(
        MODEL_DIR / "injury_history_probabilities.csv"
    )

    history_probs = {
        (str(row.previous_status), str(row.gap_category)):
        float(row.probability)
        for row in history.itertuples()
    }

    return status, previous, history_probs


def predict_appearance(
    player_id,
    game_date,
    injury_report_file="pregame_injuries_matched.csv"
):
    game_day = pd.Timestamp(game_date).normalize()

    if game_day <= TRAINING_END:
        raise ValueError(
            "This model was trained through 2025-04-13. "
            "Earlier predictions require a chronologically "
            "trained model to avoid data leakage."
        )

    status_probs, previous_probs, history_probs = load_probabilities()

    report_path = Path(injury_report_file)

    if not report_path.is_absolute():
        report_path = MODEL_DIR / report_path

    reports = pd.read_csv(report_path)

    reports["game_day"] = pd.to_datetime(
        reports["game_date"]
    ).dt.normalize()

    reports = reports[
        (reports["nba_id"] == int(player_id))
        & (reports["game_day"] <= game_day)
        & (reports["reason"] != "Trade Pending")
    ].copy()

    reports = reports.sort_values(
        ["game_day", "snapshot_dt"]
    ).drop_duplicates("game_day", keep="last")

    current = reports[
        reports["game_day"] == game_day
    ]

    if current.empty:
        return None

    current_status = str(current.iloc[-1]["status"])

    if current_status not in status_probs:
        return None

    probability = float(status_probs[current_status])

    if current_status != "Questionable":
        return probability

    previous = reports[
        reports["game_day"] < game_day
    ]

    if previous.empty:
        previous_status = "None"
        gap = "None"
    else:
        last = previous.iloc[-1]
        previous_status = str(last["status"])
        days = (game_day - last["game_day"]).days
        gap = gap_category(days)

    return float(
        history_probs.get(
            (previous_status, gap),
            previous_probs.get(
                previous_status,
                probability
            )
        )
    )


if __name__ == "__main__":
    print("Injury probability module loaded successfully.")
    print("Training cutoff:", TRAINING_END.date())
    print("Status probabilities:")

    status_probs, _, _ = load_probabilities()

    for status, probability in status_probs.items():
        print(f"  {status}: {probability:.2%}")
