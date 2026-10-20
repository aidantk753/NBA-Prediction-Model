
import os
import numpy as np
import pandas as pd
from nba_cache_utils import CACHE_DIR

# Load existing predictions
d = pd.read_csv("combined_points_predictions.csv")

d = d[
    (d["date"] >= "2026-03-01") &
    (d["date"] <= "2026-04-30")
].dropna(subset=[
    "predicted_points", "actual", "MIN",
    "2PA_PER_MIN", "3PA_PER_MIN", "FTA_PER_MIN",
    "2P_PCT", "3P_PCT", "FT_PCT"
]).copy()

# Load actual game results from cache
logs = []

for filename in os.listdir(CACHE_DIR):
    if not filename.endswith("_2025-26_games.csv"):
        continue

    g = pd.read_csv(
        os.path.join(CACHE_DIR, filename),
        usecols=[
            "Player_ID", "GAME_DATE", "MIN",
            "FGM", "FGA", "FG3M", "FG3A",
            "FTM", "FTA", "PTS"
        ]
    )
    logs.append(g)

actual = pd.concat(logs, ignore_index=True)

actual["date"] = pd.to_datetime(
    actual["GAME_DATE"]
).dt.strftime("%Y-%m-%d")

actual = actual.rename(columns={
    "Player_ID": "player_id",
    "MIN": "actual_minutes",
    "PTS": "actual_points"
})

actual["2PA"] = actual["FGA"] - actual["FG3A"]
actual["2PM"] = actual["FGM"] - actual["FG3M"]

actual = actual.drop_duplicates(["player_id", "date"])

d = d.merge(
    actual,
    on=["player_id", "date"],
    how="inner",
    validate="one_to_one"
)

# Validate actual scoring data
d["actual"] = pd.to_numeric(d["actual"], errors="raise")
d["actual_points"] = pd.to_numeric(d["actual_points"], errors="raise")

assert np.allclose(
    d["actual"].to_numpy(dtype=float),
    d["actual_points"].to_numpy(dtype=float)
), "Actual points do not match cached game logs"

# Actual attempts per minute
# Zero-minute games cannot define a per-minute rate.
d = d[d["actual_minutes"] > 0].copy()

for stat, attempts in [
    ("2PA_PER_MIN", "2PA"),
    ("3PA_PER_MIN", "FG3A"),
    ("FTA_PER_MIN", "FTA")
]:
    d[f"actual_{stat}"] = (
        d[attempts] / d["actual_minutes"]
    )

# Actual shooting percentages
# For zero attempts, retain the predicted percentage.
# Its contribution is zero when actual volume is zero.
for stat, makes, attempts in [
    ("2P_PCT", "2PM", "2PA"),
    ("3P_PCT", "FG3M", "FG3A"),
    ("FT_PCT", "FTM", "FTA")
]:
    d[f"actual_{stat}"] = np.where(
        d[attempts] > 0,
        d[makes] / d[attempts].replace(0, np.nan),
        d[stat]
    )

def predict(data, minutes=False, volume=False, efficiency=False):
    m = data["actual_minutes"] if minutes else data["MIN"]

    def component(stat, category):
        if category:
            return data[f"actual_{stat}"]
        return data[stat]

    return m * (
        2 * component("2PA_PER_MIN", volume)
          * component("2P_PCT", efficiency)
        + 3 * component("3PA_PER_MIN", volume)
          * component("3P_PCT", efficiency)
        + component("FTA_PER_MIN", volume)
          * component("FT_PCT", efficiency)
    )

scenarios = {
    "Original model": predict(d),
    "Perfect minutes": predict(d, minutes=True),
    "Perfect shot volume": predict(d, volume=True),
    "Perfect shooting efficiency": predict(d, efficiency=True),
    "Perfect minutes + volume": predict(d, minutes=True, volume=True),
    "Perfect minutes + efficiency": predict(d, minutes=True, efficiency=True),
    "Perfect volume + efficiency": predict(d, volume=True, efficiency=True),
    "All actual components": predict(
        d, minutes=True, volume=True, efficiency=True
    )
}

baseline_mae = (
    d["predicted_points"] - d["actual_points"]
).abs().mean()

print("Games analyzed:", len(d))
print("Original points MAE:", round(baseline_mae, 4))
print("\nORACLE ERROR DECOMPOSITION")
print(f'{"Scenario":34} {"MAE":>9} {"Change":>11}')

for name, prediction in scenarios.items():
    mae = (prediction - d["actual_points"]).abs().mean()
    change = baseline_mae - mae

    print(f"{name:34} {mae:9.4f} {change:+11.4f}")

print("\nPositive change = improvement over original MAE")
print("Negative change = worse than original MAE")
