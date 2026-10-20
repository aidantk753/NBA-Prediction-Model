import pandas as pd
import numpy as np
from pathlib import Path
from nba_cache_utils import get_cached_game_logs
from Weighted_model import weighted_model
from injury_minutes_adjustment import (
    STATUS_MINUTES_ADJUSTMENTS,
)

START = "2025-01-01"
END = "2025-04-13"

reports = pd.read_csv("pregame_injuries_matched.csv")
reports["game_day"] = pd.to_datetime(
    reports["game_date"], errors="coerce"
).dt.normalize()
reports["nba_id"] = pd.to_numeric(
    reports["nba_id"], errors="coerce"
)

reports = reports[
    reports["game_day"].between(START, END)
    & reports["nba_id"].notna()
    & reports["status"].isin(STATUS_MINUTES_ADJUSTMENTS)
].copy()

reports = (
    reports.sort_values("snapshot_dt")
    .drop_duplicates(["nba_id", "game_day"], keep="last")
)

results = []
errors = []

for player_id, player_reports in reports.groupby("nba_id"):
    player_id = int(player_id)

    try:
        logs = get_cached_game_logs(player_id, "2024-25")

        if logs is None or logs.empty:
            continue

        logs = logs.copy()
        logs["game_day"] = pd.to_datetime(
            logs["GAME_DATE"], errors="coerce"
        ).dt.normalize()

        actual_games = (
            logs.sort_values("game_day")
            .drop_duplicates("game_day", keep="last")
            .set_index("game_day")
        )

        for _, report in player_reports.iterrows():
            day = report["game_day"]

            if day not in actual_games.index:
                continue

            actual = actual_games.loc[day, "PTS"]

            if pd.isna(actual):
                continue

            date = day.strftime("%Y-%m-%d")

            try:
                weights = {
                    "recent": 0.5,
                    "season": 0.5,
                    "career": 0.0
                }

                original, _ = weighted_model(
                    logs, "PTS", date, player_id,
                    recent_games=5,
                    weights=weights
                )

                adjusted, _ = weighted_model(
                    logs, "PTS", date, player_id,
                    recent_games=5,
                    weights=weights,
                    injury_minutes=True
                )

                if original is None or adjusted is None:
                    continue

                results.append({
                    "nba_id": player_id,
                    "game_day": date,
                    "status": report["status"],
                    "actual_pts": float(actual),
                    "original": float(original),
                    "adjusted": float(adjusted),
                    "original_error": abs(original - actual),
                    "adjusted_error": abs(adjusted - actual),
                })

            except Exception as exc:
                errors.append((player_id, date, str(exc)))

    except Exception as exc:
        errors.append((player_id, "load", str(exc)))

df = pd.DataFrame(results)

if df.empty:
    print("No predictions generated.")
else:
    df.to_csv("injury_points_comparison.csv", index=False)

    def summary(group):
        old = group["original_error"].mean()
        new = group["adjusted_error"].mean()

        return pd.Series({
            "Games": len(group),
            "Original MAE": round(old, 4),
            "Adjusted MAE": round(new, 4),
            "Improvement": round(old - new, 4),
        })

    print("\nOVERALL RESULTS")
    print(summary(df).to_string())

    print("\nBY INJURY STATUS")
    print(
        df.groupby("status")
        .apply(summary, include_groups=False)
        .to_string()
    )

    print("\nSaved: injury_points_comparison.csv")

print(f"\nSkipped due to errors: {len(errors)}")

if errors:
    print("First five errors:")
    for error in errors[:5]:
        print(error)
