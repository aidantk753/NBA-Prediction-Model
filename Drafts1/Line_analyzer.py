import os
import requests
from dotenv import load_dotenv
from datetime import datetime
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import statistics
from Weighted_model import weighted_model
load_dotenv()
from Bet_finder import get_games, get_player_props, market_map
from nba_cache_utils import get_cached_game_logs
 
game_log_cache = {}

OPTIMAL_CONFIGS = {
    "PTS": {"recent": 0.5, "season": 0.5, "career": 0.0, "recent_games": 5},
    "REB": {"recent": 0.5, "season": 0.5, "career": 0.0, "recent_games": 5},
    "AST": {"recent": 0.5, "season": 0.5, "career": 0.0, "recent_games": 5},
    "STL": {"recent": 0.5, "season": 0.5, "career": 0.0, "recent_games": 10},
    "BLK": {"recent": 0.5, "season": 0.5, "career": 0.0, "recent_games": 10}
}

def get_season(date):
    year = int(date[:4])
    month = int(date[5:7])
    start_year = year if month >= 10 else year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"
 
def analyze_prop(player_name, stat_category, date):
    # Function to get player ID based on name
    matching_players = players.find_players_by_full_name(player_name)
 
    if not matching_players:
        print(f"Player not found: {player_name}")
        return None, None
 
    player_id = matching_players[0]["id"]
 
    # Fetch player game logs
    season = get_season(date)
    cache_key = (player_id, season)
 
    if cache_key not in game_log_cache:
        game_log_cache[cache_key] = get_cached_game_logs(player_id, season)
 
    game_logs = game_log_cache[cache_key]
 
    # Calculate weighted prediction using games before the selected date
    config = OPTIMAL_CONFIGS[stat_category]

    weights = {
        "recent": config["recent"],
        "season": config["season"],
        "career": config["career"]
    }

    season_average, std_dev = weighted_model(
        game_logs,
        stat_category,
        date,
        player_id,
        recent_games=config["recent_games"],
        weights=weights
    )
 
    if season_average is None or std_dev is None:
        print(f"Insufficient historical data for {player_name}.")
        return None, None
 
    # Display the result for mean and standard deviation
    print(f"The weighted prediction and standard deviation for {stat_category} is {season_average:.2f} ± {std_dev:.2f}")
    return season_average, std_dev
 
Date = input("What date (YYYY-MM-DD)? ")
 
games = get_games(Date)
 
if not games:
    print(f"No games found for {Date}.")
    exit()
 
analysis_cache = {}
 
for game in games:
    print(f"{game['home_team']} vs {game['away_team']}")
 
    odds = get_player_props(game["id"])
 
    for bookmaker in odds.get("bookmakers", []):
        print(f"Bookmaker: {bookmaker['title']}")
 
        for market in bookmaker.get("markets", []):
            if market["key"] not in market_map:
                continue
 
            for outcome in market.get("outcomes", []):
                if outcome.get("point") is None or not outcome.get("description"):
                    continue
 
                player_name = outcome["description"]
                betting_line = outcome["point"]
                stat_category = market_map[market["key"]]
                key = (player_name, stat_category)
 
                if key not in analysis_cache:
                    analysis_cache[key] = analyze_prop(
                        player_name, stat_category, Date
                    )
 
                mean, std_dev = analysis_cache[key]
 
                # Calculate z-score
                if mean is not None and std_dev is not None and std_dev > 0:
                    z_score = (betting_line - mean) / std_dev
                else:
                    z_score = None
                print(f"Name: {player_name}, Stat: {stat_category}, Line: {betting_line}, Z-Score: {z_score}")