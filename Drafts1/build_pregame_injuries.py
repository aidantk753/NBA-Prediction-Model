import pandas as pd

df = pd.read_csv("season_rows.csv")

# Convert report timestamps to datetimes
hours = pd.to_datetime(
    df["snapshot_time"], format="%I%p"
).dt.strftime("%H:%M")

df["snapshot_dt"] = pd.to_datetime(
    df["snapshot_date"] + " " + hours
)

# Convert scheduled tipoffs to Eastern Time datetimes
# NBA games at 12:00-05:30 are afternoon games.
game_hours = pd.to_datetime(
    df["game_time"], format="%I:%M"
)

hour = game_hours.dt.hour
hour = hour.where(hour >= 12, hour + 12)

# Hours 1-5 are afternoon; 6-11 are evening.
hour = hour.where(hour != 24, 12)

df["tipoff_dt"] = (
    pd.to_datetime(df["game_date"])
    + pd.to_timedelta(hour, unit="h")
    + pd.to_timedelta(game_hours.dt.minute, unit="m")
)

# Only reports available at least 2 hours before tipoff
df["cutoff_dt"] = df["tipoff_dt"] - pd.Timedelta(hours=2)

eligible = df[df["snapshot_dt"] <= df["cutoff_dt"]].copy()

# Select latest eligible report per game/team
keys = ["game_date", "matchup", "team"]

latest = eligible.groupby(keys)["snapshot_dt"].transform("max")

pregame = eligible[eligible["snapshot_dt"] == latest].copy()

pregame.to_csv("pregame_injuries.csv", index=False)

print("Pregame rows:", len(pregame))
print("Unique player-games:",
      pregame[pregame["player"].notna()][
          ["game_date", "team", "player"]
      ].drop_duplicates().shape[0])
print("Statuses:")
print(pregame["status"].value_counts())