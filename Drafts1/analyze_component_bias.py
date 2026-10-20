import pandas as pd
import numpy as np
import os
from nba_cache_utils import CACHE_DIR

df = pd.read_csv("combined_points_predictions.csv")

df = df[
    (df["date"] >= "2026-03-01")
    & (df["date"] <= "2026-04-30")
].copy()

df = df.dropna(
    subset=["predicted_points", "actual"]
)

# Load actual minutes from cached NBA game logs
logs = []

for filename in os.listdir(CACHE_DIR):
    if not filename.endswith("_2025-26_games.csv"):
        continue

    game_log = pd.read_csv(
        os.path.join(CACHE_DIR, filename),
        usecols=["Player_ID", "GAME_DATE", "MIN"]
    )

    logs.append(game_log)

minutes = pd.concat(logs, ignore_index=True)

minutes = minutes.rename(columns={
    "Player_ID": "player_id",
    "MIN": "actual_minutes"
})

minutes["date"] = pd.to_datetime(
    minutes["GAME_DATE"],
    format="%b %d, %Y"
).dt.strftime("%Y-%m-%d")

# Verify that each player-date has only one result
if minutes.duplicated(["player_id", "date"]).any():
    raise ValueError("Duplicate player-date records in cached logs")

# Match actual minutes to predictions
df = df.merge(
    minutes[["player_id", "date", "actual_minutes"]],
    on=["player_id", "date"],
    how="left",
    validate="one_to_one"
)

df["minutes_error"] = df["MIN"] - df["actual_minutes"]

print("Games with actual minutes:", df["actual_minutes"].notna().sum())
print("Overall minutes bias:", df["minutes_error"].mean())
print("Minutes MAE:", df["minutes_error"].abs().mean())

# Load pregame scoring averages
history = pd.read_csv(
    "predictions_big.csv",
    usecols=["player_id", "date", "stat", "actual"]
)

history = history[history["stat"] == "PTS"].copy()
history = history.drop_duplicates(["player_id", "date"])
history = history.sort_values(["player_id", "date"])

history["pregame_ppg"] = (
    history.groupby("player_id")["actual"]
    .transform(lambda x: x.expanding().mean().shift(1))
)

df = df.merge(
    history[["player_id", "date", "pregame_ppg"]],
    on=["player_id", "date"],
    how="left",
    validate="one_to_one"
)

df["scoring_group"] = pd.cut(
    df["pregame_ppg"],
    bins=[0, 10, 20, 30, np.inf],
    labels=["0-9.9", "10-19.9", "20-29.9", "30+"],
    right=False
)

results = df.groupby("scoring_group", observed=True).agg(
    games=("minutes_error", "count"),
    minutes_bias=("minutes_error", "mean"),
    minutes_mae=("minutes_error", lambda x: x.abs().mean())
)

print("\n========== MINUTES BIAS BY SCORING GROUP ==========")
print(results.round(4).to_string())

df["adjusted_bias"] = df["predicted_points"] - df["actual"]

# Diagnostic: replace predicted minutes with actual minutes
df["points_with_actual_minutes"] = (
    df["predicted_points"]
    * df["actual_minutes"]
    / df["MIN"].replace(0, np.nan)
)

df["actual_minutes_points_error"] = (
    df["points_with_actual_minutes"] - df["actual"]
)

diagnostic = df.groupby("scoring_group", observed=True).agg(
    games=("actual_minutes_points_error", "count"),
    original_points_bias=("adjusted_bias", "mean")
) 

diagnostic["actual_minutes_bias"] = (
    df.groupby("scoring_group", observed=True)[
        "actual_minutes_points_error"
    ].mean()
)

print("\n========== MINUTES CONTRIBUTION TO POINTS BIAS ==========")
print(diagnostic.round(4).to_string())

df["original_abs_error"] = (
    df["predicted_points"] - df["actual"]
).abs()

df["actual_minutes_abs_error"] = (
    df["points_with_actual_minutes"] - df["actual"]
).abs()

mae_comparison = df.groupby(
    "scoring_group", observed=True
).agg(
    games=("actual", "count"),
    original_mae=("original_abs_error", "mean"),
    actual_minutes_mae=("actual_minutes_abs_error", "mean")
)

mae_comparison["improvement_pct"] = (
    (mae_comparison["original_mae"] - mae_comparison["actual_minutes_mae"])
    / mae_comparison["original_mae"] * 100
)

print("\n========== ACTUAL MINUTES MAE COMPARISON ==========")
print(mae_comparison.round(4).to_string())

# Group games by predicted minutes
df["minutes_group"] = pd.cut(
    df["MIN"],
    bins=[0, 10, 20, 30, np.inf],
    labels=["0-9.9", "10-19.9", "20-29.9", "30+"],
    right=False
)

# Calculate minutes prediction accuracy
minutes_results = df.groupby(
    "minutes_group", observed=True
).agg(
    games=("minutes_error", "count"),
    predicted_minutes=("MIN", "mean"),
    actual_minutes=("actual_minutes", "mean"),
    minutes_bias=("minutes_error", "mean"),
    minutes_mae=("minutes_error", lambda x: x.abs().mean()),
    minutes_rmse=("minutes_error", lambda x: np.sqrt(np.mean(x**2)))
)

print("\n========== MINUTES ACCURACY BY PROJECTED PLAYING TIME ==========")
print(minutes_results.round(4).to_string())