import pandas as pd

predictions = pd.read_csv(
    "combined_points_predictions.csv",
    dtype={"game_id": str}
)

features = pd.read_csv(
    "matchup_features.csv",
    dtype={"Game_ID": str}
)

features = features.rename(columns={
    "Game_ID": "game_id"
})

merged = predictions.merge(
    features.drop(columns=["team"]),
    on=["game_id", "opponent"],
    how="left",
    validate="many_to_one"
)

feature_cols = [
    "expected_matchup_pace",
    "opponent_pregame_2PA_allowed_per_100",
    "opponent_pregame_3PA_allowed_per_100",
    "opponent_pregame_FTA_allowed_per_100"
]

print("Original rows:", len(predictions))
print("Merged rows:", len(merged))

print("\nMissing matchup features:")
print(merged[feature_cols].isna().sum())

print("\nSample:")
print(
    merged[
        ["date", "game_id", "opponent", "predicted_points"]
        + feature_cols
    ].dropna().head(5).to_string(index=False)
)

merged.to_csv(
    "points_with_matchup_features.csv",
    index=False
)

print("\nSaved points_with_matchup_features.csv")
