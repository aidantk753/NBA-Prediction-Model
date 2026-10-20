import pandas as pd
from nba_cache_utils import get_cached_game_logs
from injury_minutes_adjustment import project_minutes

df = pd.read_csv("injury_points_comparison.csv")

minutes_cache = {}

def get_minutes(row):
    player_id = int(row["nba_id"])
    date = row["game_day"]

    if player_id not in minutes_cache:
        minutes_cache[player_id] = get_cached_game_logs(
            player_id, "2024-25"
        )

    return project_minutes(minutes_cache[player_id], date)

df["projected_minutes"] = df.apply(get_minutes, axis=1)

df["minutes_group"] = pd.cut(
    df["projected_minutes"],
    bins=[0, 15, 25, 35, float("inf")],
    labels=["0-15", "15-25", "25-35", "35+"],
    include_lowest=True
)

def summarize(group):
    original = group["original_error"].mean()
    adjusted = group["adjusted_error"].mean()

    return pd.Series({
        "Games": len(group),
        "Original MAE": round(original, 4),
        "Adjusted MAE": round(adjusted, 4),
        "Improvement": round(original - adjusted, 4)
    })

print("\nBY PROJECTED MINUTES")
print(
    df.groupby("minutes_group", observed=True)
    .apply(summarize, include_groups=False)
    .to_string()
)

print("\nBY INJURY STATUS AND PROJECTED MINUTES")
print(
    df.groupby(["status", "minutes_group"], observed=True)
    .apply(summarize, include_groups=False)
    .to_string()
)

df.to_csv("injury_points_minutes_groups.csv", index=False)
print("\nSaved: injury_points_minutes_groups.csv")
