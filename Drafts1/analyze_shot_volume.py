
import os
import numpy as np
import pandas as pd
from nba_cache_utils import CACHE_DIR

d = pd.read_csv("combined_points_predictions.csv")

d = d[
    (d["date"] >= "2026-03-01") &
    (d["date"] <= "2026-04-30")
].dropna(subset=[
    "predicted_points", "actual", "MIN",
    "2PA_PER_MIN", "3PA_PER_MIN", "FTA_PER_MIN",
    "2P_PCT", "3P_PCT", "FT_PCT"
]).copy()

logs = []

for filename in os.listdir(CACHE_DIR):
    if filename.endswith("_2025-26_games.csv"):
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

actual = actual.drop_duplicates(["player_id", "date"])

d = d.merge(
    actual,
    on=["player_id", "date"],
    how="inner",
    validate="one_to_one"
)

d["actual_points"] = pd.to_numeric(d["actual_points"])
d["actual_minutes"] = pd.to_numeric(d["actual_minutes"])

d = d[d["actual_minutes"] > 0].copy()

# Actual shot attempts per minute
for stat, attempts in [
    ("2PA_PER_MIN", "2PA"),
    ("3PA_PER_MIN", "FG3A"),
    ("FTA_PER_MIN", "FTA")
]:
    d[f"actual_{stat}"] = (
        d[attempts] / d["actual_minutes"]
    )

def predict(data, replace=None):
    def rate(stat):
        if stat == replace:
            return data[f"actual_{stat}"]
        return data[stat]

    return data["MIN"] * (
        2 * rate("2PA_PER_MIN") * data["2P_PCT"]
        + 3 * rate("3PA_PER_MIN") * data["3P_PCT"]
        + rate("FTA_PER_MIN") * data["FT_PCT"]
    )

baseline = (
    d["predicted_points"] - d["actual_points"]
).abs().mean()

print("Games:", len(d))
print("Baseline points MAE:", round(baseline, 4))

print("\nINDIVIDUAL SHOT-VOLUME ORACLE TESTS")

for stat in ["2PA_PER_MIN", "3PA_PER_MIN", "FTA_PER_MIN"]:
    prediction = predict(d, replace=stat)
    mae = (prediction - d["actual_points"]).abs().mean()

    rate_error = d[stat] - d[f"actual_{stat}"]

    print(f"\n{stat}")
    print("Rate MAE:", round(rate_error.abs().mean(), 5))
    print("Rate bias:", round(rate_error.mean(), 5))
    print("Points MAE with perfect rate:", round(mae, 4))
    print("Points MAE improvement:", round(baseline - mae, 4))


# Analyze two-point attempts by projected volume

d["predicted_2PA"] = d["MIN"] * d["2PA_PER_MIN"]
d["actual_2PA"] = d["2PA"]

d["2PA_error"] = d["predicted_2PA"] - d["actual_2PA"]
d["2PA_absolute_error"] = d["2PA_error"].abs()

d["volume_bracket"] = pd.cut(
    d["predicted_2PA"],
    bins=[-float("inf"), 3, 6, 10, 15, float("inf")],
    labels=["Under 3", "3–6", "6–10", "10–15", "15+"],
    right=False
)

results = d.groupby(
    "volume_bracket", observed=True
).agg(
    games=("2PA_error", "size"),
    predicted_avg=("predicted_2PA", "mean"),
    actual_avg=("actual_2PA", "mean"),
    mae=("2PA_absolute_error", "mean"),
    bias=("2PA_error", "mean")
)

print("\nTWO-POINT ATTEMPT ACCURACY BY PREDICTED VOLUME")
print(results.round(3).to_string())

# Finer calibration analysis: 1-attempt intervals

bins = list(range(0, 21)) + [float("inf")]

d["fine_bracket"] = pd.cut(
    d["predicted_2PA"],
    bins=bins,
    right=False
)

calibration = d.groupby(
    "fine_bracket", observed=True
).agg(
    games=("2PA_error", "size"),
    predicted_avg=("predicted_2PA", "mean"),
    actual_avg=("actual_2PA", "mean"),
    mae=("2PA_absolute_error", "mean"),
    bias=("2PA_error", "mean")
)

print("\n2PA CALIBRATION BY ONE-ATTEMPT INTERVAL")
print(calibration.round(3).to_string())