import os
import numpy as np
import pandas as pd
from nba_cache_utils import CACHE_DIR

# Previously optimized configurations:
# statistic: (file, recent-game window, recent weight)
CONFIGS = {
    "MIN": ("predictions_minutes.full.csv", 8, 0.8),
    "2PA_PER_MIN": ("predictions_attempts_test.csv", 20, 0.9),
    "3PA_PER_MIN": ("predictions_attempts_test.csv", 20, 0.9),
    "FTA_PER_MIN": ("predictions_attempts_test.csv", 20, 1.0),
    "2P_PCT": ("predictions_shooting_comparison_complete.csv", 20, 0.8),
    "3P_PCT": ("predictions_shooting_comparison_complete.csv", 20, 0.8),
    "FT_PCT": ("predictions_shooting_comparison_complete.csv", 20, 0.8),
}

KEYS = ["player_id", "date"]
SHOOTING = ["2P_PCT", "3P_PCT", "FT_PCT"]

# Load only the selected configuration from each file.
frames = []

for stat, (filename, window, recent) in CONFIGS.items():
    d = pd.read_csv(
        filename,
        usecols=KEYS + [
            "stat", "window", "recent",
            "season", "career", "mean"
        ]
    )

    d = d.loc[
        (d["stat"] == stat)
        & (d["window"] == window)
        & np.isclose(d["recent"], recent)
        & np.isclose(d["season"], 1 - recent)
        & np.isclose(d["career"], 0),
        KEYS + ["mean"]
    ].rename(columns={"mean": stat})

    if d.duplicated(KEYS).any():
        raise ValueError(f"Duplicate player-games: {stat}")

    print(f"{stat}: {len(d)} rows")
    frames.append(d.set_index(KEYS))

# Minutes defines the eligible player-games.
combined = frames[0]

for frame in frames[1:]:
    combined = combined.join(frame, how="left")

combined = combined.reset_index()
combined["date"] = pd.to_datetime(combined["date"])

# Build daily league shooting totals from cached game logs.
logs = []
for filename in os.listdir(CACHE_DIR):
    if not filename.endswith("_2025-26_games.csv"):
        continue

    g = pd.read_csv(
        os.path.join(CACHE_DIR, filename),
        usecols=[
            "Player_ID", "Game_ID", "GAME_DATE", "MATCHUP",
            "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA"
        ]
    )
    logs.append(g)

league = pd.concat(logs, ignore_index=True)
league["date"] = pd.to_datetime(
    league["GAME_DATE"], format="%b %d, %Y"
)
matchups = league[
    ["Player_ID", "date", "Game_ID", "MATCHUP"]
].copy()

matchups["opponent"] = matchups["MATCHUP"].str.split().str[-1]

matchups["home_away"] = np.where(
    matchups["MATCHUP"].str.contains("@", regex=False),
    "away",
    "home"
)

matchups = matchups.rename(columns={
    "Player_ID": "player_id",
    "Game_ID": "game_id"
})

matchups = matchups[
    ["player_id", "date", "game_id", "opponent", "home_away"]
]

if matchups.duplicated(["player_id", "date"]).any():
    raise ValueError("Duplicate player-date matchups")

league["2PM"] = league["FGM"] - league["FG3M"]
league["2PA"] = league["FGA"] - league["FG3A"]

daily = (
    league.groupby("date")[
        ["2PM", "2PA", "FG3M", "FG3A", "FTM", "FTA"]
    ]
    .sum()
    .sort_index()
)

# Shift by one day: no game uses its own results
# or any future results for its league fallback.
historical = daily.cumsum().shift(1)

historical["league_2P_PCT"] = (
    historical["2PM"] / historical["2PA"].replace(0, np.nan)
)
historical["league_3P_PCT"] = (
    historical["FG3M"] / historical["FG3A"].replace(0, np.nan)
)
historical["league_FT_PCT"] = (
    historical["FTM"] / historical["FTA"].replace(0, np.nan)
)

league_rates = historical[
    ["league_2P_PCT", "league_3P_PCT", "league_FT_PCT"]
]

# Match each prediction date to the latest earlier
# available league statistics.
combined = pd.merge_asof(
    combined.sort_values("date"),
    league_rates.reset_index().sort_values("date"),
    on="date",
    direction="backward"
)

for stat in SHOOTING:
    combined[f"{stat}_fallback"] = combined[stat].isna()
    combined[stat] = combined[stat].fillna(
        combined[f"league_{stat}"]
    )

# Preserve original model predictions
combined["baseline_points"] = combined["MIN"] * (
    2 * combined["2PA_PER_MIN"] * combined["2P_PCT"]
    + 3 * combined["3PA_PER_MIN"] * combined["3P_PCT"]
    + combined["FTA_PER_MIN"] * combined["FT_PCT"]
)

# Actual points, one row per player-game.
actual = pd.read_csv(
    "predictions_big.csv",
    usecols=KEYS + ["stat", "actual"]
)

actual = (
    actual[actual["stat"] == "PTS"]
    .drop_duplicates(KEYS)
    [KEYS + ["actual"]]
)

combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
matchups["date"] = matchups["date"].dt.strftime("%Y-%m-%d")

combined = combined.merge(
    matchups,
    on=["player_id", "date"],
    how="left",
    validate="one_to_one"
)

print(
    "Games missing opponent information:",
    combined["opponent"].isna().sum()
)

# Load matchup features
matchup_features = pd.read_csv("matchup_features.csv")

combined = combined.merge(
matchup_features[
    [
        "Game_ID",
        "opponent",
        "pregame_pace",
        "expected_matchup_pace",
        "pregame_league_2PA_per_100",
        "pregame_league_3PA_per_100",
        "pregame_league_FTA_per_100",
        "opponent_pregame_2PA_allowed_per_100",
        "opponent_pregame_3PA_allowed_per_100",
        "opponent_pregame_FTA_allowed_per_100",
    ]
],
    left_on=["game_id", "opponent"],
    right_on=["Game_ID", "opponent"],
    how="left",
    validate="many_to_one"
)

print(
    "Missing matchup features:",
    combined["expected_matchup_pace"].isna().sum()
)

print("Columns before adjustment:")
print(combined.columns.tolist())

combined["pace_factor"] = (
    combined["expected_matchup_pace"]
    / combined["pregame_pace"]
)

combined["3PA_defense_factor"] = (
    combined["opponent_pregame_3PA_allowed_per_100"]
    / combined["pregame_league_3PA_per_100"]
)

combined["3PA_adjustment_factor"] = (
    1
    + 1.3579 * (combined["pace_factor"] - 1)
    + 0.3986 * (combined["3PA_defense_factor"] - 1)
).clip(lower=0)

combined["2PA_defense_factor"] = (
    combined["opponent_pregame_2PA_allowed_per_100"]
    / combined["pregame_league_2PA_per_100"]
)

combined["FTA_defense_factor"] = (
    combined["opponent_pregame_FTA_allowed_per_100"]
    / combined["pregame_league_FTA_per_100"]
)

combined["2PA_adjustment_factor"] = (
    1
    + 0.7607 * (combined["pace_factor"] - 1)
    + 0.6950 * (combined["2PA_defense_factor"] - 1)
).clip(lower=0)

combined["FTA_adjustment_factor"] = (
    1
    + 0.9909 * (combined["pace_factor"] - 1)
    + 0.3923 * (combined["FTA_defense_factor"] - 1)
).clip(lower=0)

# Preserve baseline when matchup features are unavailable
# Preserve baseline when matchup features are unavailable
for stat in ["2PA", "3PA", "FTA"]:
    combined[f"{stat}_adjustment_factor"] = (
        combined[f"{stat}_adjustment_factor"].fillna(1.0)
    )

# Apply all three adjustments to the original baseline
combined["predicted_points"] = (
    combined["baseline_points"]

    # Two-point attempts
    + 2 * combined["MIN"]
    * combined["2PA_PER_MIN"]
    * combined["2P_PCT"]
    * (combined["2PA_adjustment_factor"] - 1)

    # Three-point attempts
    + 3 * combined["MIN"]
    * combined["3PA_PER_MIN"]
    * combined["3P_PCT"]
    * (combined["3PA_adjustment_factor"] - 1)

    # Free-throw attempts
    + combined["MIN"]
    * combined["FTA_PER_MIN"]
    * combined["FT_PCT"]
    * (combined["FTA_adjustment_factor"] - 1)
)
combined = combined.merge(actual, on=KEYS, how="left")

validation = combined[
    (combined["date"] >= "2026-03-01")
    & (combined["date"] <= "2026-04-30")
].copy()

valid = validation.dropna(
    subset=["baseline_points", "predicted_points", "actual"]
).copy()

baseline_errors = valid["baseline_points"] - valid["actual"]

print("Baseline MAE:", baseline_errors.abs().mean())
print("Baseline RMSE:", np.sqrt((baseline_errors ** 2).mean()))

error = valid["predicted_points"] - valid["actual"]

print("\nMARCH-APRIL RESULTS")
print("Eligible games:", len(validation))
print("Evaluated games:", len(valid))
print("Excluded games:", len(validation) - len(valid))
print("Players:", valid["player_id"].nunique())
print("Points MAE:", error.abs().mean())
print("Points RMSE:", np.sqrt((error ** 2).mean()))

print("\nLeague-average fallback usage:")
for stat in SHOOTING:
    print(
        stat,
        int(valid[f"{stat}_fallback"].sum()),
        "games"
    )

combined.to_csv(
    "combined_points_predictions.csv",
    index=False
)

print("\nSaved: combined_points_predictions.csv")
