from itertools import product
from league_backtest import league_backtest
from result_logs import save_results
import pandas as pd
import os

weight_values = [0, 0.50, 1.0]

weight_combinations = [
    {"recent": r, "season": s, "career": c}
    for r, s, c in product(weight_values, repeat=3)
    if abs(r + s + c - 1.0) < 1e-9
]

recent_windows = [5, 10]

start_date = "2026-02-01"
end_date = "2026-02-28"

results_file = os.path.join(
    os.path.dirname(__file__),
    "backtest_results.csv"
)

completed_tests = set()

if os.path.exists(results_file):
    previous_results = pd.read_csv(results_file)

    required_columns = [
        "source", "recent", "season", "career",
        "recent_games", "start_date", "end_date", "stat_category"
    ]

    if all(column in previous_results.columns for column in required_columns):
        previous_results = previous_results[
            (previous_results["source"] == "weight_optimizer")
            & previous_results["stat_category"].notna()
        ]

        for _, row in previous_results.iterrows():
            completed_tests.add((
                round(float(row["recent"]), 8),
                round(float(row["season"]), 8),
                round(float(row["career"]), 8),
                int(row["recent_games"]),
                str(row["start_date"]),
                str(row["end_date"]),
                str(row["stat_category"])
            ))

print(f"Previously completed tests: {len(completed_tests)}")
print(f"Weight combinations: {len(weight_combinations)}")
print(f"Total configurations: {len(weight_combinations) * len(recent_windows)}")

stat_categories = ["PTS", "REB", "AST", "STL", "BLK"]

for recent_games in recent_windows:
    for weights in weight_combinations:

        print(f"\nTesting {recent_games} games with weights {weights}")

        for stat_category in stat_categories:

            test_key = (
                round(weights["recent"], 8),
                round(weights["season"], 8),
                round(weights["career"], 8),
                recent_games,
                start_date,
                end_date,
                stat_category
            )

            if test_key in completed_tests:
                print(f"Skipping completed test: {stat_category}")
                continue

            print(f"Running: {stat_category}")

            result = league_backtest(
                stat_category,
                start_date,
                end_date,
                show_results=False,
                recent_games=recent_games,
                weights=weights
            )

            if not result["complete"]:
                print(
                    f"INCOMPLETE: {stat_category}, "
                    f"{recent_games} games, weights {weights}. "
                    f"{len(result['failed_players'])} players failed. "
                    "Results will not be saved."
                )
                continue


            results_to_save = {
                "start_date": start_date,
                "end_date": end_date,
                "recent_games": recent_games,
                "stat_category": stat_category,
                "accuracy": result["accuracy"],
                "mae": result["league_mae"],
                "baseline_mae": result["league_baseline_mae"],
                "correct": result["correct"],
                "incorrect": result["incorrect"],
                "skipped": result["skipped"],
                "total": result["total"]
            }

            save_results(weights, results_to_save, "weight_optimizer")

            completed_tests.add(test_key)

            print(f"Saved: {stat_category}, {recent_games} games, weights {weights}")

        print(f"Completed configuration: {recent_games} games, weights {weights}")