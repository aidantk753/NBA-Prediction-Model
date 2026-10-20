import pandas as pd
from nba_cache_utils import get_cached_game_logs
from injury_minutes_adjustment import project_minutes

df = pd.read_csv("injury_points_comparison.csv")

results = []
cache = {}

for player_id, group in df.groupby("nba_id"):
    player_id = int(player_id)

    try:
        if player_id not in cache:
            cache[player_id] = get_cached_game_logs(
                player_id, "2024-25"
            )

        logs = cache[player_id].copy()
        logs["game_day"] = pd.to_datetime(
            logs["GAME_DATE"], errors="coerce"
        ).dt.normalize()

        for _, row in group.iterrows():
            day = pd.Timestamp(row["game_day"]).normalize()

            actual_games = logs[logs["game_day"] == day]

            if actual_games.empty:
                continue

            actual_minutes = pd.to_numeric(
                actual_games.iloc[0]["MIN"],
                errors="coerce"
            )

            projected_minutes = project_minutes(
                logs, day.strftime("%Y-%m-%d")
            )

            if (
                pd.isna(actual_minutes)
                or projected_minutes is None
                or projected_minutes <= 0
            ):
                continue

            results.append({
                "nba_id": player_id,
                "game_day": day,
                "status": row["status"],
                "projected_minutes": projected_minutes,
                "actual_minutes": float(actual_minutes),
                "minutes_error": float(actual_minutes - projected_minutes)
            })

    except Exception as exc:
        print(f"Skipped player {player_id}: {exc}")

result = pd.DataFrame(results)

result["minutes_group"] = pd.cut(
    result["projected_minutes"],
    bins=[0, 15, 25, 35, float("inf")],
    labels=["0-15", "15-25", "25-35", "35+"],
    include_lowest=True
)

def summarize(group):
    return pd.Series({
        "Games": len(group),
        "Mean error": round(group["minutes_error"].mean(), 3),
        "Minutes MAE": round(group["minutes_error"].abs().mean(), 3),
        "Std dev": round(group["minutes_error"].std(), 3)
    })

print("\nOVERALL BY INJURY STATUS")
print(
    result.groupby("status")
    .apply(summarize, include_groups=False)
    .to_string()
)

print("\nBY INJURY STATUS AND PROJECTED MINUTES")
print(
    result.groupby(
        ["status", "minutes_group"],
        observed=True
    )
    .apply(summarize, include_groups=False)
    .to_string()
)

result.to_csv("actual_injury_minutes_errors.csv", index=False)

print(f"\nTotal evaluated games: {len(result)}")
print("Saved: actual_injury_minutes_errors.csv")
