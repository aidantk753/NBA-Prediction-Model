import os
import re
import numpy as np
import pandas as pd
from nba_cache_utils import CACHE_DIR

START = pd.Timestamp("2024-11-01")
END = pd.Timestamp("2025-04-13")

BEST_FILE = "all_stats_best_weights.csv"
OUTPUT = "validation_2024_25_results.csv"
DETAILS = "validation_2024_25_predictions.csv"

best = pd.read_csv(BEST_FILE)

CONFIGS = {}

for row in best.itertuples():
    CONFIGS[row.stat] = {
        "optimized": (
            int(row.window),
            row.recent_pct / 100,
            row.season_pct / 100,
            row.career_pct / 100
        ),
        "previous_50_50": (5, 0.50, 0.50, 0.0),
    }

CONFIGS["PTS"].update({
    "previous_70_30": (8, 0.70, 0.30, 0.0),
    "previous_80_20": (8, 0.80, 0.20, 0.0),
})

STATS = list(CONFIGS)
rows = []

def get_career(pid):
    for season in ("2024-25", "2025-26"):
        path = os.path.join(
            CACHE_DIR, f"{pid}_career_{season}.csv"
        )
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        if df.empty:
            continue

        df = df[
            df["SEASON_ID"].astype(str) < "2024-25"
        ].copy()

        if df.empty:
            continue

        selected = []
        for _, group in df.groupby("SEASON_ID"):
            if (group["TEAM_ABBREVIATION"] == "TOT").any():
                selected.append(
                    group[group["TEAM_ABBREVIATION"] == "TOT"]
                )
            else:
                selected.append(group)

        df = pd.concat(selected)
        gp = pd.to_numeric(
            df["GP"], errors="coerce"
        ).sum()

        if gp <= 0:
            continue

        return {
            stat: pd.to_numeric(
                df[stat], errors="coerce"
            ).sum() / gp
            for stat in STATS
        }

    return None

def predict(previous, career, stat, config):
    window, wr, ws, wc = config

    recent = previous[-window:]
    season = previous[:-window]

    recent = recent[np.isfinite(recent)]
    season = season[np.isfinite(season)]

    r = float(recent.mean()) if len(recent) else None
    s = float(season.mean()) if len(season) else None
    c = career.get(stat) if career else None

    if wc > 0 and c is None:
        return None

    if c is not None:
        if r is None:
            return c

        if s is None:
            denominator = wr + wc
            if denominator == 0:
                return None
            return (wr*r + wc*c) / denominator

        return wr*r + ws*s + wc*c

    available = (
        (wr if r is not None else 0)
        + (ws if s is not None else 0)
    )

    if available == 0:
        return None

    return (
        (wr*r if r is not None else 0)
        + (ws*s if s is not None else 0)
    ) / available

files = sorted(
    f for f in os.listdir(CACHE_DIR)
    if re.match(r"^\d+_2024-25_games\.csv$", f)
)

print(f"Validating {len(files)} cached player files...", flush=True)

for number, filename in enumerate(files, 1):
    pid = int(filename.split("_")[0])
    career = get_career(pid)

    # Match the career-eligible sample used in optimization.
    if career is None:
        continue

    games = pd.read_csv(os.path.join(CACHE_DIR, filename))
    games["GAME_DATE"] = pd.to_datetime(
        games["GAME_DATE"], errors="coerce"
    )
    games = games.dropna(subset=["GAME_DATE"])
    games = games.sort_values("GAME_DATE").reset_index(drop=True)

    for stat in STATS:
        values = pd.to_numeric(
            games[stat], errors="coerce"
        ).to_numpy(dtype=float)

        for i in range(1, len(games)):
            date = games.at[i, "GAME_DATE"]

            if not START <= date <= END:
                continue

            actual = values[i]
            if not np.isfinite(actual):
                continue

            previous = values[:i]
            if not np.isfinite(previous).any():
                continue

            predictions = {
                name: predict(previous, career, stat, config)
                for name, config in CONFIGS[stat].items()
            }

            # Fair comparison: all models must predict this game.
            if any(
                p is None or not np.isfinite(p)
                for p in predictions.values()
            ):
                continue

            for name, prediction in predictions.items():
                rows.append({
                    "player_id": pid,
                    "date": date.strftime("%Y-%m-%d"),
                    "stat": stat,
                    "model": name,
                    "actual": actual,
                    "prediction": prediction,
                    "absolute_error": abs(prediction - actual)
                })

    if number % 50 == 0:
        print(f"{number}/{len(files)} files checked", flush=True)

details = pd.DataFrame(rows)
if details.empty:
    raise RuntimeError("No eligible validation predictions found.")

details.to_csv(DETAILS, index=False)

summary = (
    details.groupby(["stat", "model"])
    .agg(
        games=("absolute_error", "size"),
        mae=("absolute_error", "mean")
    )
    .reset_index()
)

baseline = summary[
    summary["model"] == "optimized"
][["stat", "mae"]].rename(
    columns={"mae": "optimized_mae"}
)

summary = summary.merge(baseline, on="stat")
summary["improvement_vs_optimized_pct"] = (
    (summary["mae"] - summary["optimized_mae"])
    / summary["mae"] * 100
)

summary = summary.sort_values(["stat", "mae"])
summary.to_csv(OUTPUT, index=False)

print("\n2024-25 VALIDATION RESULTS")
print(summary.to_string(index=False))
print(f"\nSaved: {OUTPUT}")
print(f"Saved: {DETAILS}")
