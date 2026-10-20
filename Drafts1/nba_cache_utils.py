import os
import time
import pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import playergamelog
 
# Absolute cache folder so every script shares one cache regardless of where it is run from
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nba_cache")
 
# How long current-season files stay fresh before refetching from the API (in seconds)
CURRENT_SEASON_TTL = 6 * 3600
 
 
def _cache_is_fresh(cache_file, season, current_season):
    if not os.path.exists(cache_file):
        return False
 
    modified = os.path.getmtime(cache_file)
 
    if season == current_season:
        return time.time() - modified < CURRENT_SEASON_TTL
 
    # Completed seasons: only trust a file saved after the season ended (July 1 of the end year)
    season_end = datetime(int(season[:4]) + 1, 7, 1).timestamp()
    return modified > season_end
 
 
def get_cached_game_logs(player_id, season):
    os.makedirs(CACHE_DIR, exist_ok=True)
 
    cache_file = os.path.join(CACHE_DIR, f"{player_id}_{season}_games.csv")
 
    # Determine the current NBA season
    today = datetime.now()
    start_year = today.year if today.month >= 10 else today.year - 1
    current_season = f"{start_year}-{str(start_year + 1)[-2:]}"
 
    if _cache_is_fresh(cache_file, season, current_season):
        try:
            return pd.read_csv(cache_file)
        except pd.errors.EmptyDataError:
            pass
 
    # Refresh from the NBA API, falling back to the saved file if the API fails
    try:
        game_logs = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            timeout=30
        ).get_data_frames()[0]
    except Exception:
        if os.path.exists(cache_file):
            try:
                return pd.read_csv(cache_file)
            except pd.errors.EmptyDataError:
                pass
        raise
 
    # Avoid caching datasets without columns
    if not game_logs.empty or len(game_logs.columns) > 0:
        game_logs.to_csv(cache_file, index=False)
 
    return game_logs
 