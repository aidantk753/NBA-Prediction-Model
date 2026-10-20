import pandas as pd
import os
from nba_cache_utils import CACHE_DIR
import numpy as np
d = pd.read_csv("points_with_matchup_features.csv")

# February–April 2026 test period
test = d.loc[
    (pd.to_datetime(d["date"]) >= "2026-02-01")
    & (pd.to_datetime(d["date"]) < "2026-05-01")
].copy()

# Previously fitted coefficients
coefficients = {
    "2PA": (0.7607, 0.6950),
    "3PA": (1.3579, 0.3986),
    "FTA": (0.9909, 0.3923)
}

test["pace_factor"] = (
    test["expected_matchup_pace"] / test["pregame_pace"]
)

for stat, (alpha, beta) in coefficients.items():

    test[f"predicted_{stat}"] = (
        test["MIN"] * test[f"{stat}_PER_MIN"]
    )

    defense_factor = (
        test[f"opponent_pregame_{stat}_allowed_per_100"]
        / test[f"pregame_league_{stat}_per_100"]
    )

    adjustment_factor = (
        1
        + alpha * (test["pace_factor"] - 1)
        + beta * (defense_factor - 1)
    ).clip(lower=0)

    test[f"adjusted_{stat}"] = (
        test[f"predicted_{stat}"] * adjustment_factor
    )

test["combined_points"] = (
    2 * test["adjusted_2PA"] * test["2P_PCT"]
    + 3 * test["adjusted_3PA"] * test["3P_PCT"]
    + test["adjusted_FTA"] * test["FT_PCT"]
)

# Compare baseline and combined model on identical games
valid = test.dropna(
    subset=["predicted_points", "combined_points", "actual"]
).copy()

actual = valid["actual"].to_numpy(dtype=float)
baseline = valid["predicted_points"].to_numpy(dtype=float)
combined = valid["combined_points"].to_numpy(dtype=float)

def evaluate(predictions):
    errors = actual - predictions
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    return mae, rmse

baseline_mae, baseline_rmse = evaluate(baseline)
combined_mae, combined_rmse = evaluate(combined)

print("\n========== COMBINED ADJUSTMENT BACKTEST ==========")
print(f"Test observations: {len(valid)}")

print(f"\nBaseline MAE: {baseline_mae:.4f}")
print(f"Combined MAE: {combined_mae:.4f}")

print(f"\nBaseline RMSE: {baseline_rmse:.4f}")
print(f"Combined RMSE: {combined_rmse:.4f}")

print(f"\nMAE improvement: {(baseline_mae - combined_mae) / baseline_mae * 100:.3f}%")
print(f"RMSE improvement: {(baseline_rmse - combined_rmse) / baseline_rmse * 100:.3f}%")