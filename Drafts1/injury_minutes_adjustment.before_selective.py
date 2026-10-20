import pandas as pd
from pathlib import Path
from functools import lru_cache

STATUS_MINUTES_ADJUSTMENTS = {
    "Questionable": -1.41,
    "Probable": -0.53,
    "Available": -0.40,
}

COUNTING_STATS = {
    "PTS", "REB", "AST", "STL", "BLK",
    "OREB", "DREB", "FGM", "FGA",
    "FG3M", "FG3A", "FTM", "FTA",
    "TOV", "PF"
}

REPORT_PATH = (
    Path(__file__).resolve().parent
    / "pregame_injuries_matched.csv"
)


@lru_cache(maxsize=1)
def load_reports():
    reports = pd.read_csv(REPORT_PATH)

    reports["game_day"] = pd.to_datetime(
        reports["game_date"], errors="coerce"
    ).dt.normalize()

    reports["nba_id"] = pd.to_numeric(
        reports["nba_id"], errors="coerce"
    )

    reports["snapshot_dt"] = pd.to_datetime(
        reports["snapshot_dt"], errors="coerce"
    )

    return reports


def project_minutes(game_logs, cutoff_date):
    game_day = pd.Timestamp(cutoff_date).normalize()

    games = game_logs.copy()

    games["GAME_DATE"] = pd.to_datetime(
        games["GAME_DATE"], errors="coerce"
    ).dt.normalize()

    games["MIN"] = pd.to_numeric(
        games["MIN"], errors="coerce"
    )

    previous = games[
        games["GAME_DATE"] < game_day
    ].sort_values("GAME_DATE")

    previous = previous.dropna(subset=["MIN"])

    recent = previous.tail(8)["MIN"]
    season = previous.iloc[:-8]["MIN"]

    if recent.empty:
        return None

    if season.empty:
        return float(recent.mean())

    return float(
        0.80 * recent.mean()
        + 0.20 * season.mean()
    )


def adjust_for_injury_minutes(
    prediction, game_logs, stat_category,
    cutoff_date, player_id
):
    if prediction is None or stat_category not in COUNTING_STATS:
        return prediction

    if not REPORT_PATH.exists() or "MIN" not in game_logs.columns:
        return prediction

    game_day = pd.Timestamp(cutoff_date).normalize()

    reports = load_reports()

    current = reports[
        (reports["nba_id"] == int(player_id))
        & (reports["game_day"] == game_day)
        & (reports["reason"] != "Trade Pending")
    ]

    if current.empty:
        return prediction

    current = current.sort_values("snapshot_dt")
    status = str(current.iloc[-1]["status"])

    adjustment = STATUS_MINUTES_ADJUSTMENTS.get(status)

    if adjustment is None:
        return prediction

    baseline_minutes = project_minutes(
        game_logs, cutoff_date
    )

    if baseline_minutes is None or baseline_minutes <= 0:
        return prediction

    adjusted_minutes = max(
        0.0,
        baseline_minutes + adjustment
    )

    return float(
        prediction
        * adjusted_minutes
        / baseline_minutes
    )
