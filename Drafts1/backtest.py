from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from datetime import date, datetime, timedelta
import statistics
import os
import pandas as pd
from Weighted_model import weighted_model, WEIGHTS
from nba_cache_utils import get_cached_game_logs
from injury_probability import predict_appearance
 
def analyze_prop(player_name, stat_category, betting_line=None, cutoff_date=None, recent_games=10):
    # Function to get player ID based on name
    matching_players = players.find_players_by_full_name(player_name)
 
    if not matching_players:
        print(f"Player not found: {player_name}")
        return None
 
    player_id = matching_players[0]["id"]
    game_logs = get_cached_game_logs(player_id, "2024-25")
    stats_list = game_logs[stat_category].tolist()
 
    #Verify if a cutoff date is provided and filter the game logs accordingly
    game_dates = game_logs['GAME_DATE']
    if cutoff_date:
        cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d").date()
        filtered_stats_list = [stat for stat, date in zip(stats_list, game_dates) if datetime.strptime(date, "%b %d, %Y").date() < cutoff]
        stats_list = filtered_stats_list
 
    # Calculate weighted prediction
    if cutoff_date is None:
        cutoff_date = date.today().strftime("%Y-%m-%d")
    prediction, std_dev = weighted_model(
        game_logs,
        stat_category,
        cutoff_date,
        player_id,
        recent_games=recent_games,
        weights=WEIGHTS
    )
    if prediction is None or std_dev is None:
        return None
 
 
    # Display the result for mean and standard deviation
    print(f"{player_name} has a weighted prediction of {prediction:.2f} {stat_category}")
    print(f"Standard deviation: {std_dev:.2f}")
    if cutoff_date > "2025-04-13":
        appearance_probability = predict_appearance(player_id, cutoff_date)
    else:
        appearance_probability = None

    if appearance_probability is not None:
        print(f"Appearance probability: {appearance_probability:.2%}")
    else:
        print("Appearance probability: unavailable")
    # Calculate z-score for the player's betting line
    if betting_line is not None and std_dev != 0:
        z_score = (betting_line - prediction) / std_dev
    else:
        z_score = None
    return prediction, std_dev, z_score, game_logs
 
 
 
 
 
 
 
 
 
 
def backtest_prop(player_name, stat_category, cutoff_date, choice="generate"):
    result = analyze_prop(player_name, stat_category, betting_line=None, cutoff_date=cutoff_date)
 
    if result is None:
        print("Insufficient historical data. Skipping game.")
        return None
 
    mean, std_dev, z_score, game_logs = result
 
    betting_line = round(mean - 0.5) + 0.5
    print(f"Generated betting line for {player_name}'s {stat_category}: {betting_line:.2f}")
    if std_dev != 0:
        z_score = (betting_line - mean) / std_dev
    else:
        z_score = None
    if z_score is None:
        prediction = None
    elif z_score > 0:
        prediction = "under"
    elif z_score < 0:
        prediction = "over"
    else:
        prediction = None
    game_dates = game_logs["GAME_DATE"]
    stats_list = game_logs[stat_category].tolist()
    cutoff_date = datetime.strptime(cutoff_date, "%Y-%m-%d").date()
    game_found = False
    for stat, date in zip(stats_list, game_dates):
        if datetime.strptime(date, "%b %d, %Y").date() == cutoff_date:
            game_found = True
            actual_result = stat
            print(f"Actual result: {actual_result}")
            if actual_result > betting_line:
                print(f"{player_name} exceeded the betting line of {betting_line:.2f} with {actual_result} {stat_category}.")
                if prediction == "over":
                    print("The prediction was correct.")
                elif prediction == "under":
                    print("The prediction was incorrect.")
            elif actual_result < betting_line:
                print(f"{player_name} did not meet the betting line of {betting_line:.2f}, achieving only {actual_result} {stat_category}.")
                if prediction == "under":
                    print("The prediction was correct.")
                elif prediction == "over":
                    print("The prediction was incorrect.")
            else:
                print(f"{player_name} exactly met the betting line of {betting_line:.2f} with {actual_result} {stat_category}.")
    if not game_found:
        print(f"No game found for {player_name} on {cutoff_date}.")
 
 
 
 
 
 
 
 
 
 
 
def automated_backtest(player_name, stat_category, start_date, end_date, recent_games=10, weights=None, player_id=None, require_career=False):
 
    # Look the player up by name only when no ID was passed in
    if player_id is None:
        matching_players = players.find_players_by_full_name(player_name)
 
        if not matching_players:
            print(f"Player not found: {player_name}")
            return None
 
        player_id = matching_players[0]["id"]
    # Fetch game logs using the shared caching function
    game_logs = get_cached_game_logs(player_id, "2025-26").copy()
    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
 
    if start_date > end_date:
        print("Start date must be before end date.")
        return None
 
    if stat_category not in game_logs.columns:
        print(f"Invalid statistic: {stat_category}")
        return None
 
    # Convert dates and sort chronologically
    game_logs["GAME_DATE"] = game_logs["GAME_DATE"].apply(
        lambda date: datetime.strptime(date, "%b %d, %Y").date()
    )
 
    game_logs = game_logs.sort_values("GAME_DATE").reset_index(drop=True)
 
    correct_predictions = 0
    incorrect_predictions = 0
    skipped_games = 0
    results = []
    absolute_errors = []
    baseline_errors = []
 
    # Test every actual game in the requested date range
    for i in range(len(game_logs)):
 
        game_date = game_logs.loc[i, "GAME_DATE"]
 
        if not (start_date <= game_date <= end_date):
            continue
 
        print(f"\n========== {game_date} ==========")
 
        # Only use games played BEFORE the game being tested
        previous_games = game_logs.iloc[:i]
        stats_list = previous_games[stat_category].dropna().tolist()
 
        # Calculate weighted prediction
        mean, std_dev = weighted_model(
            game_logs,
            stat_category,
            game_date.strftime("%Y-%m-%d"),
            player_id,
            recent_games=recent_games,
            weights=weights,
            require_career=require_career,
            injury_minutes=True
        )
 
        # Skip games without sufficient historical data
        if mean is None or std_dev is None:
            print("Insufficient historical data. Skipping game.")
            skipped_games += 1
            continue
 
        if std_dev == 0:
            print("Standard deviation is zero. Skipping game.")
            skipped_games += 1
            continue
 
        # Generate betting line from historical season average
        if not stats_list:
            print("No previous games available. Skipping game.")
            skipped_games += 1
            continue
        historical_average = sum(stats_list) / len(stats_list)
        betting_line = round(historical_average - 0.5) + 0.5
 
        # Calculate z-score using weighted prediction
        z_score = (betting_line - mean) / std_dev
 
        # Make prediction using your existing logic
        if z_score > 0:
            prediction = "under"
        elif z_score < 0:
            prediction = "over"
        else:
            print("No prediction could be made. Skipping game.")
            skipped_games += 1
            continue
 
        # Get actual result
        actual_result = game_logs.loc[i, stat_category]
 
        if actual_result != actual_result:
            print("Missing actual result. Skipping game.")
            skipped_games += 1
            continue
 
        # Determine actual outcome
        if actual_result > betting_line:
            actual_outcome = "over"
        elif actual_result < betting_line:
            actual_outcome = "under"
        else:
            print("Push. Skipping game.")
            skipped_games += 1
            continue
 
        absolute_error = abs(mean - actual_result)
        absolute_errors.append(absolute_error)
 
        baseline_error = abs(historical_average - actual_result)
        baseline_errors.append(baseline_error)
        
        # Compare prediction against actual result
        correct = prediction == actual_outcome
 
        if correct:
            correct_predictions += 1
            result = "CORRECT"
        else:
            incorrect_predictions += 1
            result = "INCORRECT"
 
        print(f"Historical mean: {mean:.2f}")
        print(f"Standard deviation: {std_dev:.2f}")
        appearance_probability = predict_appearance(
            player_id,
            game_date.strftime("%Y-%m-%d")
        )

        if appearance_probability is not None:
            print(f"Appearance probability: {appearance_probability:.2%}")
        else:
            print("Appearance probability: unavailable")
        print(f"Generated betting line: {betting_line:.2f}")
        print(f"Z-score: {z_score:.3f}")
        print(f"Prediction: {prediction.upper()}")
        print(f"Actual {stat_category}: {actual_result}")
        print(f"Actual outcome: {actual_outcome.upper()}")
        print(f"Result: {result}")
 
        # Store results for later analysis
        results.append({
            "date": game_date,
            "player": player_name,
            "stat": stat_category,
            "mean": mean,
            "std_dev": std_dev,
            "betting_line": betting_line,
            "z_score": z_score,
            "prediction": prediction,
            "actual_result": actual_result,
            "actual_outcome": actual_outcome,
            "correct": correct
        })
 
    # Calculate final accuracy
    total_predictions = correct_predictions + incorrect_predictions
 
    print("\n========== FINAL BACKTEST RESULTS ==========")
    print(f"Player: {player_name}")
    print(f"Statistic: {stat_category}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {incorrect_predictions}")
    print(f"Skipped games: {skipped_games}")
    print(f"Total predictions: {total_predictions}")
 
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions * 100
        print(f"Prediction accuracy: {accuracy:.2f}%")
    else:
        accuracy = None
        print("No valid predictions were made.")
 
    if absolute_errors:
        mae = sum(absolute_errors) / len(absolute_errors)
        print(f"Mean Absolute Error: {mae:.2f}")
    else:
        mae = None
 
    if baseline_errors:
        baseline_mae = sum(baseline_errors) / len(baseline_errors)
        print(f"Baseline Mean Absolute Error: {baseline_mae:.2f}")
    else:
        baseline_mae = None
 
    return {
        "correct": correct_predictions,
        "incorrect": incorrect_predictions,
        "skipped": skipped_games,
        "total": total_predictions,
        "accuracy": accuracy,
        "results": results,
        "mae": mae,
        "mae_games": len(absolute_errors),
        "baseline_mae": baseline_mae
    }
 
if __name__ == "__main__":
    results = automated_backtest(
        "johnny juzang",
        "PTS",
        "2025-04-14",
        "2025-06-22"
    )