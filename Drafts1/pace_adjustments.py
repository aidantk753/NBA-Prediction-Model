import os
import numpy as np
import pandas as pd
from nba_cache_utils import CACHE_DIR

logs = []

for filename in os.listdir(CACHE_DIR):
    if not filename.endswith("_2025-26_games.csv"):
        continue

    path = os.path.join(CACHE_DIR, filename)

    df = pd.read_csv(path, usecols=[
        "Player_ID",
        "Game_ID",
        "GAME_DATE",
        "MATCHUP",
        "FGA",
        "FG3A",
        "FTA",
        "OREB",
        "TOV"
    ])

    logs.append(df)

games = pd.concat(logs, ignore_index=True)

numeric_cols = ["FGA", "FG3A", "FTA", "OREB", "TOV"]

print("\nMissing player statistics:")
print(games[numeric_cols].isna().sum())

for col in numeric_cols:
    games[col] = pd.to_numeric(games[col], errors="coerce")

games["team"] = games["MATCHUP"].str.split().str[0]
games["opponent"] = games["MATCHUP"].str.split().str[-1]

games["2PA"] = games["FGA"] - games["FG3A"]

team_games = (
    games.groupby(["Game_ID", "GAME_DATE", "team", "opponent"], as_index=False)
    [["FGA", "FG3A", "2PA", "FTA", "OREB", "TOV"]]
    .sum()
)

team_games["possessions"] = (
    team_games["FGA"]
    + 0.44 * team_games["FTA"]
    - team_games["OREB"]
    + team_games["TOV"]
)

print("Team-game rows:", len(team_games))
print("Unique games:", team_games["Game_ID"].nunique())

print(
    "Games without exactly two teams:",
    (team_games.groupby("Game_ID")["team"].nunique() != 2).sum()
)

print(
    "Duplicate player-game records:",
    games.duplicated(["Player_ID", "Game_ID"]).sum()
)

opponent_stats = team_games[
    ["Game_ID", "team", "2PA", "FG3A", "FTA", "possessions"]
].copy()

opponent_stats = opponent_stats.rename(columns={
    "team": "opponent",
    "2PA": "opp_2PA",
    "FG3A": "opp_3PA",
    "FTA": "opp_FTA",
    "possessions": "opp_possessions"
})

team_games = team_games.merge(
    opponent_stats,
    on=["Game_ID", "opponent"],
    how="left",
    validate="one_to_one"
)

print("Missing opponent statistics:",
      team_games["opp_possessions"].isna().sum())

team_games["2PA_allowed_per_100"] = (
    100 * team_games["opp_2PA"] / team_games["opp_possessions"]
)

team_games["3PA_allowed_per_100"] = (
    100 * team_games["opp_3PA"] / team_games["opp_possessions"]
)

team_games["FTA_allowed_per_100"] = (
    100 * team_games["opp_FTA"] / team_games["opp_possessions"]
)

team_games["pace"] = (
    team_games["possessions"] + team_games["opp_possessions"]
) / 2

print(team_games[
    [
        "team",
        "opponent",
        "pace",
        "2PA_allowed_per_100",
        "3PA_allowed_per_100",
        "FTA_allowed_per_100"
    ]
].head(10).to_string(index=False))

print("\nPossession distribution:")
print(team_games["possessions"].describe())

print("\nLowest possession totals:")
print(
    team_games.nsmallest(5, "possessions")[
        ["GAME_DATE", "team", "opponent", "possessions"]
    ].to_string(index=False)
)

print("\nHighest possession totals:")
print(
    team_games.nlargest(5, "possessions")[
        ["GAME_DATE", "team", "opponent", "possessions"]
    ].to_string(index=False)
)
team_games["date"] = pd.to_datetime(
    team_games["GAME_DATE"],
    format="%b %d, %Y"
)

team_games = team_games.sort_values(
    ["team", "date", "Game_ID"]
).copy()

stats = {
    "2PA": "opp_2PA",
    "3PA": "opp_3PA",
    "FTA": "opp_FTA"
}

for stat, column in stats.items():
    team_games[f"prior_{stat}_allowed"] = (
        team_games.groupby("team")[column]
        .transform(lambda x: x.shift(1).cumsum())
    )

team_games["prior_possessions_faced"] = (
    team_games.groupby("team")["opp_possessions"]
    .transform(lambda x: x.shift(1).cumsum())
)

for stat in stats:
    team_games[f"pregame_{stat}_allowed_per_100"] = (
        100
        * team_games[f"prior_{stat}_allowed"]
        / team_games["prior_possessions_faced"]
    )

team_games["prior_possessions"] = (
    team_games.groupby("team")["possessions"]
    .transform(lambda x: x.shift(1).cumsum())
)

team_games["prior_games"] = (
    team_games.groupby("team").cumcount()
)

team_games["pregame_pace"] = (
    team_games["prior_possessions"]
    / team_games["prior_games"].replace(0, np.nan)
)

daily = (
    team_games.groupby("date", as_index=False)
    [["possessions", "2PA", "FG3A", "FTA"]]
    .sum()
    .sort_values("date")
)

for col in ["possessions", "2PA", "FG3A", "FTA"]:
    daily[f"prior_league_{col}"] = (
        daily[col].shift(1).cumsum()
    )

daily["prior_league_team_games"] = (
    daily.index.to_series().map(
        lambda i: team_games.groupby("date").size().iloc[:i].sum()
    ).values
)

daily["pregame_league_pace"] = (
    daily["prior_league_possessions"]
    / daily["prior_league_team_games"].replace(0, np.nan)
)

for stat, col in [
    ("2PA", "2PA"),
    ("3PA", "FG3A"),
    ("FTA", "FTA")
]:
    daily[f"pregame_league_{stat}_per_100"] = (
        100 * daily[f"prior_league_{col}"]
        / daily["prior_league_possessions"]
    )

team_games = team_games.merge(
    daily[
        ["date", "pregame_league_pace",
         "pregame_league_2PA_per_100",
         "pregame_league_3PA_per_100",
         "pregame_league_FTA_per_100"]
    ],
    on="date",
    how="left",
    validate="many_to_one"
)


print("\nPregame defensive ratings and pace:")
print(
    team_games[
        [
            "date",
            "team",
            "opponent",
            "pregame_pace",
            "pregame_2PA_allowed_per_100",
            "pregame_3PA_allowed_per_100",
            "pregame_FTA_allowed_per_100"
        ]
    ].dropna().head(10).to_string(index=False)
)

opponent_pace = team_games[
    ["Game_ID", "team", "pregame_pace"]
].rename(columns={
    "team": "opponent",
    "pregame_pace": "opponent_pregame_pace"
})

team_games = team_games.merge(
    opponent_pace,
    on=["Game_ID", "opponent"],
    how="left",
    validate="one_to_one"
)

team_games["expected_matchup_pace"] = (
    team_games["pregame_pace"]
    + team_games["opponent_pregame_pace"]
) / 2

print("\nExpected matchup pace:")
print(
    team_games[
        ["date", "team", "opponent",
         "pregame_pace", "opponent_pregame_pace",
         "expected_matchup_pace"]
    ].dropna().head(10).to_string(index=False)
)

defensive_cols = [
    "pregame_2PA_allowed_per_100",
    "pregame_3PA_allowed_per_100",
    "pregame_FTA_allowed_per_100"
]

opponent_defense = team_games[
    ["Game_ID", "team"] + defensive_cols
].copy()

opponent_defense = opponent_defense.rename(columns={
    "team": "opponent",
    **{col: f"opponent_{col}" for col in defensive_cols}
})

team_games = team_games.merge(
    opponent_defense,
    on=["Game_ID", "opponent"],
    how="left",
    validate="one_to_one"
)

print("\nOpponent defensive features:")
print(
    team_games[
        ["date", "team", "opponent", "expected_matchup_pace"]
        + [f"opponent_{col}" for col in defensive_cols]
    ].dropna().head(10).to_string(index=False)
)

feature_cols = [
    "Game_ID",
    "team",
    "opponent",
    "pregame_pace",
    "pregame_league_pace",
    "expected_matchup_pace",
    "pregame_league_2PA_per_100",
    "pregame_league_3PA_per_100",
    "pregame_league_FTA_per_100",
    "opponent_pregame_2PA_allowed_per_100",
    "opponent_pregame_3PA_allowed_per_100",
    "opponent_pregame_FTA_allowed_per_100"
]

team_games[feature_cols].to_csv(
    "matchup_features.csv",
    index=False
)

print("\nSaved matchup_features.csv")

