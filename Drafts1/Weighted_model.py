import pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import playercareerstats
import os
import statistics
from nba_cache_utils import get_cached_game_logs, CACHE_DIR


career_stats_cache = {}
career_values_cache = {}
prepared_games_cache = {}

WEIGHTS = {
    "recent": 0.65,
    "season": 0.25,
    "career": 0.10
}



def weighted_model(game_logs, stat_category, cutoff_date, player_id, recent_games=10, weights=None, require_career=False):
    if weights is None:
        weights = WEIGHTS

    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("Weights must sum to 1")
    # Defines cutoff dates and filters previous games sorted by most to least recent
    cutoff_date = datetime.strptime(cutoff_date, "%Y-%m-%d").date()
    season_start_year = cutoff_date.year if cutoff_date.month >= 10 else cutoff_date.year - 1
    season_id = f"{season_start_year}-{str(season_start_year + 1)[-2:]}"
    cache_key = (player_id, season_id)

    if cache_key not in prepared_games_cache:
        prepared_games = game_logs.copy()
        prepared_games["GAME_DATE"] = pd.to_datetime(
            prepared_games["GAME_DATE"]
        ).dt.date
        prepared_games = prepared_games.sort_values("GAME_DATE")

        prepared_games_cache[cache_key] = prepared_games

    prepared_games = prepared_games_cache[cache_key]
    previous_games = prepared_games[
        prepared_games["GAME_DATE"] < cutoff_date
    ]
    
    recent = previous_games.tail(recent_games)[stat_category].dropna().tolist()
    from nba_api.stats.endpoints import playergamelog

    season = (
        previous_games.iloc[:-recent_games][stat_category].dropna().tolist()
        if len(previous_games) > recent_games
        else []
    )




    
    shooting_columns = {
        "2P_PCT": ("2PM", "2PA"),
        "3P_PCT": ("FG3M", "FG3A"),
        "FT_PCT": ("FTM", "FTA")
    }

    if stat_category in shooting_columns:
        makes_col, attempts_col = shooting_columns[stat_category]

        recent_games_data = previous_games.tail(recent_games)
        season_games_data = previous_games.iloc[:-recent_games]

        def shooting_percentage(games):
            attempts = games[attempts_col].sum()

            if attempts == 0:
                return None

            return games[makes_col].sum() / attempts

        average10 = shooting_percentage(recent_games_data)
        season_average = shooting_percentage(season_games_data)
        if weights["career"] == 0:
            if average10 is None and season_average is None:
                career_file = os.path.join(
                    CACHE_DIR, f"{player_id}_career_{season_id}.csv"
                )

                if os.path.exists(career_file):
                    historical = pd.read_csv(career_file)
                    historical = historical[
                        historical["SEASON_ID"] < season_id
                    ].copy()

                    if not historical.empty:
                        selected = []
                        for sid, group in historical.groupby("SEASON_ID"):
                            if "TOT" in group["TEAM_ABBREVIATION"].values:
                                selected.append(
                                    group[group["TEAM_ABBREVIATION"] == "TOT"]
                                )
                            else:
                                selected.append(group)

                        historical = pd.concat(selected)

                        if stat_category == "2P_PCT":
                            makes = historical["FGM"].sum() - historical["FG3M"].sum()
                            attempts = historical["FGA"].sum() - historical["FG3A"].sum()
                        elif stat_category == "3P_PCT":
                            makes = historical["FG3M"].sum()
                            attempts = historical["FG3A"].sum()
                        else:
                            makes = historical["FTM"].sum()
                            attempts = historical["FTA"].sum()

                        if attempts > 0:
                            return makes / attempts, None

                return None, None

            if season_average is None:
                prediction = average10
            else:
                prediction = (
                    average10 * weights["recent"]
                    + season_average * weights["season"]
                )

            std_dev = weighted_standard_deviation(
                recent, season, [], weights
            )

            return prediction, std_dev

    average10 = statistics.mean(recent) if recent else None
    season_average = statistics.mean(season) if season else None

    if weights["career"] == 0:
        available_weight = (
            (weights["recent"] if recent else 0)
            + (weights["season"] if season else 0)
        )

        if available_weight == 0:
            return None, None

        prediction = (
            (average10 * weights["recent"] if recent else 0)
            + (season_average * weights["season"] if season else 0)
        ) / available_weight

        std_dev = weighted_standard_deviation(
            recent, season, [], weights
        )

        return prediction, std_dev

    # Create cache directory
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Unique career statistics file for each player and season (refreshes when a new season starts)
    career_cache = os.path.join(CACHE_DIR, f"{player_id}_career_{season_id}.csv")

    if (player_id, season_id) in career_stats_cache:
        career_stats = career_stats_cache[(player_id, season_id)]
    else:
        if os.path.exists(career_cache):
            career_stats = pd.read_csv(career_cache)

            if career_stats.empty:
                os.remove(career_cache)
                career_stats = playercareerstats.PlayerCareerStats(
                    player_id,
                    timeout=10
                ).get_data_frames()[0]
                career_stats.to_csv(career_cache, index=False)
        else:
            career_stats = playercareerstats.PlayerCareerStats(
                player_id, 
                timeout=10
            ).get_data_frames()[0]
            career_stats.to_csv(career_cache, index=False)

        career_stats_cache[(player_id, season_id)] = career_stats

    previous_seasons = career_stats[
        career_stats["SEASON_ID"] < season_id
    ]
    if require_career and previous_seasons.empty:
        return None, None

    if previous_seasons.empty:
        if weights["career"] > 0:
            return None, None

        available_weight = (
            weights["recent"] if recent else 0
        ) + (
            weights["season"] if season else 0
        )

        if available_weight == 0:
            return None, None

        prediction = (
            (statistics.mean(recent) * weights["recent"] if recent else 0)
            + (statistics.mean(season) * weights["season"] if season else 0)
        ) / available_weight

        std_dev = weighted_standard_deviation(
            recent, season, [], weights
        )

        return prediction, std_dev

    else:
        selected_seasons=[]
        for sid, season_games in previous_seasons.groupby("SEASON_ID"):
            if "TOT" in season_games["TEAM_ABBREVIATION"].values:
                selected_seasons.append(
                season_games[season_games["TEAM_ABBREVIATION"] == "TOT"]
            )
            else:
                selected_seasons.append(season_games)

        # Combine the selected rows into one DataFrame
        filtered_career = pd.concat(selected_seasons)

        # Calculate career average
        total_games = filtered_career["GP"].sum()

        if total_games == 0:
            return None, None

        career_average = filtered_career[stat_category].sum() / total_games

    career_key = (
        player_id,
        stat_category,
        tuple(filtered_career["SEASON_ID"].unique())
    )

    if career_key in career_values_cache:
        career_values = career_values_cache[career_key]
    else:
        career_values = []

        for historical_season in filtered_career["SEASON_ID"].unique():
            historical_logs = get_cached_game_logs(
                player_id,
                historical_season
            )

            career_values.extend(
                historical_logs[stat_category].dropna().tolist()
            )

        career_values_cache[career_key] = career_values

    std_dev = weighted_standard_deviation(
        recent,
        season,
        career_values,
        weights
    )

    # Weighs all three
    if average10 is None:
        prediction = career_average

    elif season_average is None:
        total = weights["recent"] + weights["career"]
        if total == 0:
            return None, None

        prediction = (
            average10 * weights["recent"]
            + career_average * weights["career"]
        ) / total

    else:
        prediction = (
            average10 * weights["recent"]
            + season_average * weights["season"]
            + career_average * weights["career"]
        )

    return prediction, std_dev







def weighted_standard_deviation(recent, season, career, weights=None):
    if weights is None:
        weights = WEIGHTS
    groups = []

    if recent:
        groups.append((recent, weights["recent"]))

    if season:
        groups.append((season, weights["season"]))

    if career:
        groups.append((career, weights["career"]))

    # Match the fallback weights in Weighted_model.py
    if not recent:
        groups = [(career, 1.0)] if career else []

    elif not season:
        groups = [
            (recent, weights["recent"]),
            (career, weights["career"])
    ]

    groups = [(data, weight) for data, weight in groups if data]

    if not groups:
        return None

    total_weight = sum(weight for _, weight in groups)

    if total_weight == 0:
        return None

    weighted_mean = sum(
        weight * statistics.mean(data)
        for data, weight in groups
    ) / total_weight

    weighted_variance = sum(
        weight * (
            statistics.pvariance(data)
            + (statistics.mean(data) - weighted_mean) ** 2
        )
        for data, weight in groups
    ) / total_weight

    return weighted_variance ** 0.5

# INJURY MINUTES INTEGRATION
from injury_minutes_adjustment import adjust_for_injury_minutes

_original_weighted_model = weighted_model


def weighted_model(
    game_logs, stat_category, cutoff_date, player_id,
    recent_games=10, weights=None, require_career=False,
    injury_minutes=False
):
    prediction, std_dev = _original_weighted_model(
        game_logs,
        stat_category,
        cutoff_date,
        player_id,
        recent_games=recent_games,
        weights=weights,
        require_career=require_career
    )

    if injury_minutes and prediction is not None:
        prediction = adjust_for_injury_minutes(
            prediction,
            game_logs,
            stat_category,
            cutoff_date,
            player_id
        )

    return prediction, std_dev
