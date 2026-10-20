from backtest import automated_backtest
from nba_api.stats.endpoints import commonallplayers
import time
import contextlib
import io
from backtest import automated_backtest, WEIGHTS
from result_logs import save_results
import os
import pandas as pd
 
 
def league_backtest(stat_category, start_date, end_date, show_results=True, recent_games=10, weights=None, max_players=None, require_career=False):
 
 
    # Get players on NBA rosters during the selected season
    print("Fetching NBA player list...")
 
    cache_file = os.path.join(
        os.path.dirname(__file__),
        "nba_cache",
        "players_2025-26.csv"
    )
 
    if os.path.exists(cache_file):
        player_data = pd.read_csv(cache_file)
    else:
        player_data = commonallplayers.CommonAllPlayers(
            is_only_current_season=1,
            league_id="00",
            season="2025-26",
            timeout=10
        ).get_data_frames()[0]
 
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        player_data.to_csv(cache_file, index=False)
 
    # Restrict to players with a roster entry in the season
    player_data = player_data.drop_duplicates(subset="PERSON_ID")

    if max_players:
        player_data = player_data.sample(
            n=min(max_players, len(player_data)), random_state=42
        )

    total_correct = 0
    total_incorrect = 0
    total_skipped = 0
 
    player_results = []
    failed_players = []
 
    # Creates Weighted Error Statistics
    total_weighted_error = 0
    total_mae_games = 0
    total_baseline_error = 0
 
    for index, player in enumerate(player_data.itertuples(), start=1):
 
        player_name = player.DISPLAY_FIRST_LAST
 
        print(
            f"\n========== PLAYER {index}/{len(player_data)}: "
            f"{player_name} =========="
        )
        time.sleep(0.6)
 
        try:
            import contextlib
            import io
 
            # Retry transient failures (API timeouts, etc.) before giving up
            for attempt in range(3):
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = automated_backtest (
                            player_name,
                            stat_category,
                            start_date,
                            end_date,
                            recent_games=recent_games,
                            weights=weights,
                            player_id=int(player.PERSON_ID),
                            require_career=require_career
                        )
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
 
            if result is None:
                continue
 
            total_correct += result["correct"]
            total_incorrect += result["incorrect"]
            total_skipped += result["skipped"]
        
            if result["mae"] is not None:
                total_weighted_error += result["mae"] * result["mae_games"]
                total_mae_games += result["mae_games"]
                total_baseline_error += result["baseline_mae"] * result["mae_games"]
                
            if result["total"] > 0:
                player_results.append({
                    "player": player_name,
                    "correct": result["correct"],
                    "incorrect": result["incorrect"],
                    "total": result["total"],
                    "accuracy": result["accuracy"],
                    "mae": result["mae"],
                    "mae_games": result["mae_games"]
                })
 
        except Exception as error:
            print(f"Error processing {player_name}: {error}")
            failed_players.append(player_name)
 
    total_predictions = total_correct + total_incorrect
 
    league_accuracy = (
        total_correct / total_predictions * 100
        if total_predictions > 0
        else None
    )
 
    if total_mae_games > 0:
        league_mae = total_weighted_error / total_mae_games
        league_baseline_mae = total_baseline_error / total_mae_games 
    else:
        league_mae = None
        league_baseline_mae = None
 
 
 
    # Rank players by accuracy, then sample size
    player_results.sort(
        key=lambda player: (
            player["accuracy"],
            player["total"]
        ),
        reverse=True
    )
 
    if show_results:
        print("\n========== LEAGUE BACKTEST RESULTS ==========")
        print(f"Statistic: {stat_category}")
        print(f"Players processed: {len(player_data)}")
        print(f"Players with predictions: {len(player_results)}")
        print(f"Failed players: {len(failed_players)}")
        print(f"Correct predictions: {total_correct}")
        print(f"Incorrect predictions: {total_incorrect}")
        print(f"Skipped games: {total_skipped}")
        print(f"Total predictions: {total_predictions}")
 
        if league_accuracy is not None:
            print(f"League-wide accuracy: {league_accuracy:.2f}%")
        else:
            print("No valid predictions were made.")
 
        if league_mae is not None:
            print(f"League-wide Mean Absolute Error: {league_mae:.2f}")
        else:
            print("No valid predictions were made.")
 
        if league_baseline_mae is not None:
            print(f"League Baseline MAE: {league_baseline_mae:.2f}")
        else:
            print("No valid predictions were made.")
        
 
        if failed_players:
            print("\nFailed players:")
            for player_name in failed_players:
                print(player_name)
 
    return {
        "complete": len(failed_players) == 0,
        "accuracy": league_accuracy,
        "correct": total_correct,
        "incorrect": total_incorrect,
        "skipped": total_skipped,
        "total": total_predictions,
        "players": player_results,
        "failed_players": failed_players,
        "league_mae": league_mae,
        "league_baseline_mae": league_baseline_mae
    }
if __name__ == "__main__":
    stat_categories = ["PTS", "REB", "AST", "STL", "BLK"]
 
    all_results = {}
 
    for stat_category in stat_categories:
        all_results[stat_category] = league_backtest(
            stat_category,
            "2026-02-01",
            "2026-02-28",
            show_results=False
        )

        if not all_results[stat_category]["complete"]:
            raise SystemExit(f"{stat_category}: {len(all_results[stat_category]['failed_players'])} players failed, not saving")
    print("\n========== COMPLETE LEAGUE BACKTEST ==========")

 
    for stat_category, result in all_results.items():
        print(f"\n========== {stat_category} ==========")
        print(f"Total predictions: {result['total']}")
        print(f"Correct: {result['correct']}")
        print(f"Incorrect: {result['incorrect']}")
        print(f"Skipped: {result['skipped']}")
 
        if result["accuracy"] is not None:
            print(f"Accuracy: {result['accuracy']:.2f}%")
 
        if result["league_mae"] is not None:
            print(f"Model MAE: {result['league_mae']:.2f}")
 
        if result["league_baseline_mae"] is not None:
            print(f"Baseline MAE: {result['league_baseline_mae']:.2f}")
 
        print(f"Failed players: {len(result['failed_players'])}")
        print("\n========== TOP 10 PLAYERS ==========")
 
        eligible_players = [
            player for player in result["players"]
            if player["total"] >= 10
        ]
 
        if not eligible_players:
            print("No players met the minimum prediction requirement.")
 
        for rank, player in enumerate(eligible_players[:10], start=1):
            print(
                f"{rank}. {player['player']}: "
                f"{player['accuracy']:.2f}% "
                f"({player['correct']}/{player['total']})"
            )
 
    results_to_save = {
    "start_date": "2026-02-01",
    "end_date": "2026-02-28"
    }
 
    for stat, result in all_results.items():
        results_to_save[f"{stat}_accuracy"] = result["accuracy"]
        results_to_save[f"{stat}_mae"] = result["league_mae"]
        results_to_save[f"{stat}_baseline_mae"] = result["league_baseline_mae"]
        results_to_save[f"{stat}_correct"] = result["correct"]
        results_to_save[f"{stat}_incorrect"] = result["incorrect"]
        results_to_save[f"{stat}_skipped"] = result["skipped"]
        results_to_save[f"{stat}_total"] = result["total"]
 
    save_results(WEIGHTS, results_to_save, "league_backtest")
 