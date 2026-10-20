import pandas as pd
import unicodedata
import re
from nba_api.stats.static import players
import glob
import os


def normalize(name):
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", name.lower())

df = pd.read_csv("pregame_injuries.csv")
directory = pd.DataFrame(players.get_players())

# Convert "Last, First" into "First Last"
df["full_name"] = df["player"].apply(
    lambda x: " ".join(x.split(", ", 1)[::-1])
    if isinstance(x, str) else None
)

df["name_key"] = df["full_name"].map(normalize)
directory["name_key"] = directory["full_name"].map(normalize)

# Prefer players represented in our 2024-25 game-log cache
cached_ids = {
    int(os.path.basename(path).split("_")[0])
    for path in glob.glob("nba_cache/*_2024-25_games.csv")
}

directory["cached"] = directory["id"].isin(cached_ids)

directory = directory.sort_values(
    "cached", ascending=False
)

# Keep only names that resolve unambiguously after prioritizing cached IDs
directory = directory.drop_duplicates("name_key", keep="first")

# Apply verified naming exceptions
exceptions = {
    "Brandon Boston Jr.": 1630527,
    "Trey Jemison": 1641998,
    "Carlton Carrington": 1642267,
    "Jimmy Butler": 202710,
    "David Jones": 1642357,
    "Alexandre Sarr": 1642259,
}

df = df.merge(
    directory[["name_key", "id"]],
    on="name_key",
    how="left",
    validate="many_to_one"
)

df["nba_id"] = df["full_name"].map(exceptions).fillna(df["id"])

df = df.drop(columns=["id", "name_key"])

df.to_csv("pregame_injuries_matched.csv", index=False)

players_only = df[df["player"].notna()]

print("Player-game records:", len(players_only))
print("Matched records:", players_only["nba_id"].notna().sum())
print("Unmatched records:", players_only["nba_id"].isna().sum())
print("\nUnmatched players:")
print(players_only.loc[
    players_only["nba_id"].isna(), "full_name"
].drop_duplicates().to_string(index=False))