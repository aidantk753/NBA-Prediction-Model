import pandas as pd
import numpy as np

d = pd.read_csv("points_with_matchup_features.csv")

d["pace_factor"] = (
    d["expected_matchup_pace"] / d["pregame_pace"]
)

for stat in ["2PA", "3PA", "FTA"]:
    d[f"{stat}_defense_factor"] = (
        d[f"opponent_pregame_{stat}_allowed_per_100"]
        / d[f"pregame_league_{stat}_per_100"]
    )

cols = [
    "pace_factor",
    "2PA_defense_factor",
    "3PA_defense_factor",
    "FTA_defense_factor"
]

print("\nFactor distributions:")
print(d[cols].describe().to_string())

print("\nMissing or infinite factors:")
print(d[cols].isna().sum())
print(np.isinf(d[cols]).sum())