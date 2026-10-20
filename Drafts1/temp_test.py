from Weighted_model import weighted_model
from nba_cache_utils import get_cached_game_logs
from nba_api.stats.static import players

player_id = players.find_players_by_full_name("Anthony Edwards")[0]["id"]

game_logs = get_cached_game_logs(player_id, "2025-26")

test_cases = [
    {
        "recent": 1.0,
        "season": 0.0,
        "career": 0.0
    },
    {
        "recent": 0.0,
        "season": 0.0,
        "career": 1.0
    }
]

expected_results = [
    (28.2, 7.12460525222275),
    (23.876640419947506, 9.329377034525375)
]

for weights, expected in zip(test_cases, expected_results):
    prediction, std_dev = weighted_model(
        game_logs,
        "PTS",
        "2026-02-15",
        player_id,
        recent_games=10,
        weights=weights
    )

    passed = (
        prediction is not None
        and std_dev is not None
        and abs(prediction - expected[0]) < 1e-9
        and abs(std_dev - expected[1]) < 1e-9
    )

    print(f"\nWeights: {weights}")
    print(f"Prediction: {prediction}")
    print(f"Standard deviation: {std_dev}")
    print(f"Result: {'PASS' if passed else 'FAIL'}")