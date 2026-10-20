import pandas as pd
import os
from nba_cache_utils import CACHE_DIR
import numpy as np

d = pd.read_csv("points_with_matchup_features.csv")

d["predicted_3PA"] = d["MIN"] * d["3PA_PER_MIN"]

print(d[[
    "player_id",
    "game_id",
    "predicted_3PA",
    "3PA_defense_factor"
]] if "3PA_defense_factor" in d.columns else d[[
    "player_id",
    "game_id",
    "predicted_3PA"
]].head())

logs = []

for filename in os.listdir(CACHE_DIR):
    if filename.endswith("_2025-26_games.csv"):
        df = pd.read_csv(
            os.path.join(CACHE_DIR, filename),
            usecols=["Player_ID", "Game_ID", "FG3A"]
        )
        logs.append(df)

actuals = pd.concat(logs, ignore_index=True)

actuals = actuals.rename(columns={
    "Player_ID": "player_id",
    "Game_ID": "game_id",
    "FG3A": "actual_3PA"
})

d = d.merge(
    actuals,
    on=["player_id", "game_id"],
    how="left",
    validate="one_to_one"
)

d["3PA_error"] = d["actual_3PA"] - d["predicted_3PA"]

print("\nMatched actual attempts:")
print(d[[
    "predicted_3PA",
    "actual_3PA",
    "3PA_error"
]].head().to_string(index=False))

print("\nMissing actual attempts:", d["actual_3PA"].isna().sum())

d["3PA_defense_factor"] = (
    d["opponent_pregame_3PA_allowed_per_100"]
    / d["pregame_league_3PA_per_100"]
)

correlation = d["3PA_defense_factor"].corr(d["3PA_error"])

print("\nCorrelation between opponent 3PA allowance and prediction error:")
print(correlation)

print("\nPrediction error by opponent defensive factor:")
d["defense_group"] = pd.qcut(
    d["3PA_defense_factor"],
    q=5,
    duplicates="drop"
)

print(
    d.groupby("defense_group", observed=True)["3PA_error"]
    .agg(["count", "mean"])
    .to_string()
)

print("\nPace vs. 3PA prediction error:")
print(d["pace_factor"].corr(d["3PA_error"]) if "pace_factor" in d.columns
      else (d["expected_matchup_pace"] / d["pregame_pace"]).corr(d["3PA_error"]))

print("\nPace vs. opponent 3PA allowance:")
print(
    (d["expected_matchup_pace"] / d["pregame_pace"])
    .corr(d["3PA_defense_factor"])
)

d["pace_factor"] = (
    d["expected_matchup_pace"] / d["pregame_pace"]
)

X = np.column_stack([
    np.ones(len(d)),
    d["pace_factor"] - 1,
    d["3PA_defense_factor"] - 1
])

y = pd.to_numeric(d["3PA_error"], errors="coerce").to_numpy(dtype=float)

valid = np.isfinite(X).all(axis=1) & np.isfinite(y)

print("Regression observations:", valid.sum(), "of", len(d))

coefficients = np.linalg.lstsq(X[valid], y[valid], rcond=None)[0]

print("\nJoint regression coefficients:")
print(f"Intercept: {coefficients[0]:.4f}")
print(f"Pace coefficient: {coefficients[1]:.4f}")
print(f"Defense coefficient: {coefficients[2]:.4f}")

print("\nInvalid regression observations:")
print(
    d.loc[~valid, [
        "player_id",
        "game_id",
        "MIN",
        "3PA_PER_MIN",
        "predicted_3PA",
        "actual_3PA",
        "3PA_error"
    ]].head(15).to_string(index=False)
)

print("\nMissing predicted attempts:",
      pd.to_numeric(d["predicted_3PA"], errors="coerce").isna().sum())

# Chronological training/test split
train = d.loc[
    valid & (pd.to_datetime(d["date"]) < "2026-02-01")
].copy()

test = d.loc[
    valid & (pd.to_datetime(d["date"]) >= "2026-02-01")
].copy()

# Fit multiplicative coefficients on training data only
# actual - predicted = predicted * [alpha*(P-1) + beta*(D-1)]
X_train = np.column_stack([
    train["predicted_3PA"] * (train["pace_factor"] - 1),
    train["predicted_3PA"] * (train["3PA_defense_factor"] - 1)
])

y_train = pd.to_numeric(
    train["actual_3PA"]
).to_numpy(dtype=float) - train["predicted_3PA"].to_numpy(dtype=float)

alpha, beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

# Apply coefficients to test data
test["adjustment_factor"] = (
    1
    + alpha * (test["pace_factor"] - 1)
    + beta * (test["3PA_defense_factor"] - 1)
)

test["adjustment_factor"] = test["adjustment_factor"].clip(lower=0)

test["adjusted_3PA"] = (
    test["predicted_3PA"] * test["adjustment_factor"]
)

# Reconstruct original points prediction
test["baseline_points"] = test["predicted_points"]

# Calculate change in projected three-point attempts
test["3PA_change"] = (
    test["adjusted_3PA"] - test["predicted_3PA"]
)

# Adjust only the three-point scoring component
test["adjusted_points"] = (
    test["baseline_points"]
    + 3 * test["3P_PCT"] * test["3PA_change"]
)

# Compare against actual points
points_test = test.dropna(
    subset=["baseline_points", "adjusted_points", "actual"]
).copy()

actual_pts = pd.to_numeric(points_test["actual"]).to_numpy(dtype=float)
baseline_pts = points_test["baseline_points"].to_numpy(dtype=float)
adjusted_pts = points_test["adjusted_points"].to_numpy(dtype=float)

def points_metrics(predictions):
    errors = actual_pts - predictions
    return np.mean(np.abs(errors)), np.sqrt(np.mean(errors**2))

base_pts_mae, base_pts_rmse = points_metrics(baseline_pts)
adj_pts_mae, adj_pts_rmse = points_metrics(adjusted_pts)

print("\n========== 3PA-ADJUSTED POINTS BACKTEST ==========")
print(f"Test observations: {len(points_test)}")
print(f"Baseline points MAE: {base_pts_mae:.4f}")
print(f"Adjusted points MAE: {adj_pts_mae:.4f}")
print(f"Baseline points RMSE: {base_pts_rmse:.4f}")
print(f"Adjusted points RMSE: {adj_pts_rmse:.4f}")

actual = pd.to_numeric(test["actual_3PA"]).to_numpy(dtype=float)
baseline = test["predicted_3PA"].to_numpy(dtype=float)
adjusted = test["adjusted_3PA"].to_numpy(dtype=float)

def evaluate(predictions):
    errors = actual - predictions
    return np.mean(np.abs(errors)), np.sqrt(np.mean(errors**2))

base_mae, base_rmse = evaluate(baseline)
adj_mae, adj_rmse = evaluate(adjusted)

print("\n========== MULTIPLICATIVE 3PA BACKTEST ==========")
print(f"Training observations: {len(train)}")
print(f"Testing observations:  {len(test)}")

print("\nFitted coefficients:")
print(f"Alpha (pace):    {alpha:.4f}")
print(f"Beta (defense):  {beta:.4f}")

print("\nPerformance:")
print(f"Baseline MAE: {base_mae:.4f}")
print(f"Adjusted MAE: {adj_mae:.4f}")
print(f"Baseline RMSE: {base_rmse:.4f}")
print(f"Adjusted RMSE: {adj_rmse:.4f}")

print(f"\nMAE improvement: {(base_mae-adj_mae)/base_mae*100:.3f}%")
print(f"RMSE improvement: {(base_rmse-adj_rmse)/base_rmse*100:.3f}%")