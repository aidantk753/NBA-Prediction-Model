import pandas as pd
import numpy as np

df = pd.read_csv("combined_points_predictions.csv")

df = df[
    (df["date"] >= "2026-03-01")
    & (df["date"] <= "2026-04-30")
].copy()

df = df.dropna(
    subset=["baseline_points", "predicted_points", "actual"]
)

# Load historical player scoring results
history = pd.read_csv(
    "predictions_big.csv",
    usecols=["player_id", "date", "stat", "actual"]
)

history = history[history["stat"] == "PTS"].copy()
history = history.drop_duplicates(["player_id", "date"])

# Sort each player's games chronologically
history = history.sort_values(["player_id", "date"])

# Average points from earlier games only
history["pregame_ppg"] = (
    history.groupby("player_id")["actual"]
    .transform(lambda x: x.expanding().mean().shift(1))
)

# Attach pregame scoring averages to our predictions
df = df.merge(
    history[["player_id", "date", "pregame_ppg"]],
    on=["player_id", "date"],
    how="left",
    validate="one_to_one"
)

# Categorize players by pregame points per game
bins = [0, 10, 20, 30, np.inf]
labels = ["0-9.9", "10-19.9", "20-29.9", "30+"]

df["scoring_group"] = pd.cut(
    df["pregame_ppg"],
    bins=bins,
    labels=labels,
    right=False
)

# Calculate prediction errors
df["baseline_error"] = df["baseline_points"] - df["actual"]
df["adjusted_error"] = df["predicted_points"] - df["actual"]

# Compare accuracy across scoring groups
results = df.groupby("scoring_group", observed=True).agg(
    games=("actual", "count"),
    baseline_mae=("baseline_error", lambda x: x.abs().mean()),
    adjusted_mae=("adjusted_error", lambda x: x.abs().mean()),
    baseline_rmse=("baseline_error", lambda x: np.sqrt(np.mean(x**2))),
    adjusted_rmse=("adjusted_error", lambda x: np.sqrt(np.mean(x**2)))
)

results["mae_improvement_pct"] = (
    (results["baseline_mae"] - results["adjusted_mae"])
    / results["baseline_mae"] * 100
)

print("\n========== ACCURACY BY SCORING GROUP ==========")
print(results.round(4).to_string())

print("\nGames without pregame PPG:", df["pregame_ppg"].isna().sum())

# Signed prediction errors
df["baseline_bias"] = (
    df["baseline_points"] - df["actual"]
)

df["adjusted_bias"] = (
    df["predicted_points"] - df["actual"]
)

# Average bias by scoring group
bias_results = df.groupby(
    "scoring_group", observed=True
).agg(
    games=("actual", "count"),
    baseline_bias=("baseline_bias", "mean"),
    adjusted_bias=("adjusted_bias", "mean")
)

print("\n========== PREDICTION BIAS BY SCORING GROUP ==========")
print(bias_results.round(4).to_string())