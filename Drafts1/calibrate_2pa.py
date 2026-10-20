
import os
import numpy as np
import pandas as pd
from nba_cache_utils import CACHE_DIR

d = pd.read_csv("combined_points_predictions.csv")

# Load actual two-point attempts from cached game logs
logs = []

for filename in os.listdir(CACHE_DIR):
    if filename.endswith("_2025-26_games.csv"):
        g = pd.read_csv(
            os.path.join(CACHE_DIR, filename),
            usecols=[
                "Player_ID", "GAME_DATE", "FGA", "FG3A"
            ]
        )
        logs.append(g)

actual = pd.concat(logs, ignore_index=True)

actual["date"] = pd.to_datetime(
    actual["GAME_DATE"]
).dt.strftime("%Y-%m-%d")

actual["actual_2PA"] = actual["FGA"] - actual["FG3A"]

actual = actual.rename(columns={"Player_ID": "player_id"})

actual = actual[
    ["player_id", "date", "actual_2PA"]
].drop_duplicates(["player_id", "date"])

d = d.merge(
    actual,
    on=["player_id", "date"],
    how="left",
    validate="one_to_one"
)

d["actual_2PA"] = pd.to_numeric(
    d["actual_2PA"], errors="coerce"
)

d["predicted_2PA"] = d["MIN"] * d["2PA_PER_MIN"]

train = d[
    (d["date"] >= "2025-11-01") &
    (d["date"] <= "2026-02-28")
].dropna(subset=["predicted_2PA", "actual_2PA"])

test = d[
    (d["date"] >= "2026-03-01") &
    (d["date"] <= "2026-04-30")
].dropna(subset=[
    "predicted_points", "actual", "predicted_2PA",
    "2P_PCT"
]).copy()

# Fit actual 2PA = intercept + slope * predicted 2PA
slope, intercept = np.polyfit(
    train["predicted_2PA"],
    train["actual_2PA"],
    1
)

print("Training games:", len(train))
print("Validation games:", len(test))
print("Intercept:", round(intercept, 5))
print("Slope:", round(slope, 5))

# Correct only the two-point attempt component
test["calibrated_2PA"] = (
    intercept + slope * test["predicted_2PA"]
).clip(lower=0)

# Original two-point scoring contribution:
original_2P = (
    2 * test["predicted_2PA"] * test["2P_PCT"]
)

# Calibrated two-point scoring contribution:
calibrated_2P = (
    2 * test["calibrated_2PA"] * test["2P_PCT"]
)

test["calibrated_points"] = (
    test["predicted_points"]
    - original_2P
    + calibrated_2P
)

test["actual"] = pd.to_numeric(test["actual"])

original_mae = (
    test["predicted_points"] - test["actual"]
).abs().mean()

calibrated_mae = (
    test["calibrated_points"] - test["actual"]
).abs().mean()

print("\nMARCH-APRIL COMPARISON")
print("Original MAE:", round(original_mae, 5))
print("Calibrated MAE:", round(calibrated_mae, 5))
print("Improvement:", round(original_mae - calibrated_mae, 5))
print("Percentage improvement:",
      round(100 * (original_mae - calibrated_mae) / original_mae, 3), "%")

