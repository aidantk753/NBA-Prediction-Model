from pathlib import Path
import shutil

model_path = Path("Weighted_model.py")
backtest_path = Path("backtest.py")

if not model_path.exists() or not backtest_path.exists():
    raise FileNotFoundError("Run this from the directory containing Weighted_model.py and backtest.py")

for path in (model_path, backtest_path):
    backup = path.with_suffix(".before_injury_minutes.py")
    if not backup.exists():
        shutil.copy2(path, backup)

adjustment_code = '''
import pandas as pd
from pathlib import Path

STATUS_MINUTES_ADJUSTMENTS = {
    "Questionable": -1.41,
    "Probable": -0.53,
    "Available": -0.40,
}

COUNTING_STATS = {
    "PTS", "REB", "AST", "STL", "BLK",
    "OREB", "DREB", "FGM", "FGA",
    "FG3M", "FG3A", "FTM", "FTA",
    "TOV", "PF"
}

REPORT_PATH = Path(__file__).resolve().parent / "pregame_injuries_matched.csv"


def adjust_for_injury_minutes(
    prediction, game_logs, stat_category, cutoff_date, player_id
):
    if prediction is None or stat_category not in COUNTING_STATS:
        return prediction

    if not REPORT_PATH.exists() or "MIN" not in game_logs.columns:
        return prediction

    game_day = pd.Timestamp(cutoff_date).normalize()

    reports = pd.read_csv(REPORT_PATH)
    reports["game_day"] = pd.to_datetime(
        reports["game_date"], errors="coerce"
    ).dt.normalize()
    reports["nba_id"] = pd.to_numeric(
        reports["nba_id"], errors="coerce"
    )

    current = reports[
        (reports["nba_id"] == int(player_id))
        & (reports["game_day"] == game_day)
        & (reports["reason"] != "Trade Pending")
    ].copy()

    if current.empty:
        return prediction

    current = current.sort_values("snapshot_dt")
    status = str(current.iloc[-1]["status"])

    adjustment = STATUS_MINUTES_ADJUSTMENTS.get(status)

    if adjustment is None:
        return prediction

    historical = game_logs.copy()
    historical["GAME_DATE"] = pd.to_datetime(
        historical["GAME_DATE"], errors="coerce"
    ).dt.normalize()

    historical = historical[
        historical["GAME_DATE"] < game_day
    ].sort_values("GAME_DATE")

    minutes = pd.to_numeric(
        historical.tail(8)["MIN"], errors="coerce"
    ).dropna()

    if minutes.empty:
        return prediction

    baseline_minutes = float(minutes.mean())

    if baseline_minutes <= 0:
        return prediction

    projected_minutes = max(
        0.0, baseline_minutes + adjustment
    )

    return float(
        prediction * projected_minutes / baseline_minutes
    )
'''

Path("injury_minutes_adjustment.py").write_text(adjustment_code)

model = model_path.read_text()

marker = "# INJURY MINUTES INTEGRATION"

if marker not in model:
    model += '''

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
'''
    model_path.write_text(model)

backtest = backtest_path.read_text()

# Enable adjustment only in the automated backtest.
start = backtest.index("def automated_backtest(")
before = backtest[:start]
after = backtest[start:]

old = "require_career=require_career"
new = "require_career=require_career,\n            injury_minutes=True"

if new not in after:
    if old not in after:
        raise RuntimeError("Could not locate automated_backtest weighted_model call")
    after = after.replace(old, new, 1)

backtest_path.write_text(before + after)

print("Installation complete.")
print("Original files backed up.")
print("Injury minutes enabled for automated_backtest().")
print("Other weighted_model() calls remain unchanged.")
