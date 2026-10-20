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


def reason_category(reason):
    reason = str(reason).strip()

    if reason.startswith("Injury/Illness"):
        return "Injury/Illness"

    if reason.startswith("G League"):
        return "G League"

    if reason in ("-", "nan", ""):
        return "Other/Unspecified"

    return "Other"


def load_probabilities():
    status = pd.read_csv(
        MODEL_DIR / "hybrid_status_probabilities.csv"
    )

    reason = pd.read_csv(
        MODEL_DIR / "hybrid_reason_probabilities.csv"
    )

    history = pd.read_csv(
        MODEL_DIR / "hybrid_history_probabilities.csv",
        keep_default_na=False
    )

    status_probs = {
        str(row.status): float(row.probability)
        for row in status.itertuples()
    }

    reason_probs = {
        (str(row.status), str(row.reason_category)):
        float(row.probability)
        for row in reason.itertuples()
    }

    history_probs = {
        (
            str(row.status),
            str(row.reason_category),
            str(row.previous_status),
            str(row.gap_category)
        ): float(row.probability)
        for row in history.itertuples()
    }

    return status_probs, reason_probs, history_probs


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

    status_probs, reason_probs, history_probs = load_probabilities()

    report_path = Path(injury_report_file)

    if not report_path.is_absolute():
        report_path = MODEL_DIR / report_path

    reports = pd.read_csv(report_path)

    reports["game_day"] = pd.to_datetime(
        reports["game_date"],
        format="%m/%d/%Y"
    ).dt.normalize()

    reports["nba_id"] = pd.to_numeric(
        reports["nba_id"], errors="coerce"
    )

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

    current_row = current.iloc[-1]

    current_status = str(current_row["status"])
    current_reason = reason_category(current_row["reason"])

    if current_status not in status_probs:
        return None

    # Parent model: status + reason.
    probability = reason_probs.get(
        (current_status, current_reason),
        status_probs[current_status]
    )

    # History features are used only for Questionable.
    if current_status != "Questionable":
        return float(probability)

    previous = reports[
        reports["game_day"] < game_day
    ]

    if previous.empty:
        previous_status = "None"
        gap = "None"
    else:
        last = previous.iloc[-1]

        previous_status = str(last["status"])

        days = (
            game_day - last["game_day"]
        ).days

        gap = gap_category(days)

    history_key = (
        current_status,
        current_reason,
        previous_status,
        gap
    )

    return float(
        history_probs.get(history_key, probability)
    )


if __name__ == "__main__":
    print("Hybrid injury probability module loaded successfully.")
    print("Training cutoff:", TRAINING_END.date())

    status_probs, reason_probs, history_probs = load_probabilities()

    print("\nStatus probabilities:")

    for status, probability in status_probs.items():
        print(f"  {status}: {probability:.2%}")

    print("\nModel groups:")
    print("Status:", len(status_probs))
    print("Status + reason:", len(reason_probs))
    print("Questionable history:", len(history_probs))
