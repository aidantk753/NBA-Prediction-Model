import requests
import pandas as pd
from datetime import datetime, timezone
from nba_api.stats.static import players
import unicodedata

url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

response = requests.get(url, timeout=15)
response.raise_for_status()

data = response.json()
records = []

for team in data["injuries"]:
    for injury in team.get("injuries", []):
        athlete = injury.get("athlete", {})
        details = injury.get("details", {})

        records.append({
            "player": athlete.get("displayName"),
            "team": team.get("displayName"),
            "position": athlete.get("position", {}).get("abbreviation"),
            "espn_id": next(
                (
                    link["href"].split("/id/")[1].split("/")[0]
                    for link in athlete.get("links", [])
                    if "/id/" in link.get("href", "")
                ),
                None
            ),
            "status": injury.get("status"),
            "report_date": injury.get("date"),
            "injury_type": details.get("type"),
            "injury_location": details.get("location"),
            "injury_detail": details.get("detail"),
            "injury_side": details.get("side"),
            "estimated_return": details.get("returnDate"),
            "short_comment": injury.get("shortComment"),
            "long_comment": injury.get("longComment"),
            "snapshot_time": datetime.now(timezone.utc).isoformat()
        })

df = pd.DataFrame(records)

nba_players = pd.DataFrame(players.get_players())

def normalize_name(name):
    if not isinstance(name, str):
        return ""

    name = unicodedata.normalize("NFKD", name)

    return "".join(
        char for char in name
        if not unicodedata.combining(char)
    ).lower().strip()


df["normalized_name"] = df["player"].apply(normalize_name)

nba_players["normalized_name"] = (
    nba_players["full_name"].apply(normalize_name)
)

matched = df.merge(
    nba_players[["id", "full_name", "normalized_name"]],
    on="normalized_name",
    how="left"
)

duplicates = matched[
    matched.duplicated(subset=["espn_id"], keep=False)
]

print("\nDUPLICATE PLAYER MATCHES")
print("Duplicate rows:", len(duplicates))

if not duplicates.empty:
    print(
        duplicates[
            ["player", "espn_id", "id", "full_name"]
        ].to_string(index=False)
    )

print("\nPLAYER MATCHING")
print("Matched:", matched["id"].notna().sum())
print("Unmatched:", matched["id"].isna().sum())

print("\nUNMATCHED PLAYERS")
print(matched.loc[matched["id"].isna(), "player"].to_string(index=False))

print("Total injury records:", len(df))
print(df[[
    "player", "team", "status",
    "injury_type", "injury_detail", "estimated_return"
]].head(15).to_string(index=False))

columns = [
    "status",
    "injury_type",
    "injury_detail",
    "report_date",
    "estimated_return",
    "espn_id"
]

print("\nDATA COMPLETENESS")
print(df[columns].notna().sum())

print("\nINJURY TYPES")
print(df["injury_type"].value_counts(dropna=False))

matched = matched.rename(columns={"id": "nba_id"})

matched.to_csv("espn_injuries.csv", index=False)

print("\nSaved: espn_injuries.csv")