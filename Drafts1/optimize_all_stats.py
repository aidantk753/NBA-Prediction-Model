import os
import re
import json
import time
import numpy as np
import pandas as pd
from nba_cache_utils import CACHE_DIR

STATS = ["PTS", "REB", "AST", "STL", "BLK"]
WINDOWS = range(1, 31)
START = "2025-11-01"
END = "2026-04-30"

CHECKPOINT = "all_stats_optimizer_checkpoint.npz"
RESULTS = "all_stats_optimization_results.csv"
BEST = "all_stats_best_weights.csv"

# A shared evaluation sample makes weight configurations comparable.
# Career data must exist, even for configurations with career weight 0.
REQUIRE_CAREER_FOR_COMPARISON = True

WEIGHT_INTS = np.array(
    [(r, s, 100-r-s)
     for r in range(101)
     for s in range(101-r)],
    dtype=np.int16
)
WEIGHTS = WEIGHT_INTS.astype(np.float64) / 100.0
N_WEIGHTS = len(WEIGHTS)
N_WINDOWS = len(WINDOWS)
N_STATS = len(STATS)
SHAPE = (N_STATS, N_WINDOWS, N_WEIGHTS)

GAME_PATTERN = re.compile(r"^(\d+)_2025-26_games\.csv$")
FILES = sorted(
    (int(m.group(1)), os.path.join(CACHE_DIR, filename))
    for filename in os.listdir(CACHE_DIR)
    if (m := GAME_PATTERN.match(filename))
)

def career_averages(pid):
    path = os.path.join(CACHE_DIR, f"{pid}_career_2025-26.csv")
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    if df.empty:
        return None

    df = df[df["SEASON_ID"].astype(str) < "2025-26"].copy()
    if df.empty:
        return None

    selected = []
    for _, group in df.groupby("SEASON_ID"):
        if (group["TEAM_ABBREVIATION"] == "TOT").any():
            selected.append(group[group["TEAM_ABBREVIATION"] == "TOT"])
        else:
            selected.append(group)

    df = pd.concat(selected)
    gp = pd.to_numeric(df["GP"], errors="coerce").sum()

    if not np.isfinite(gp) or gp <= 0:
        return None

    averages = {}
    for stat in STATS:
        values = pd.to_numeric(df[stat], errors="coerce")
        averages[stat] = float(values.sum() / gp)

    return averages

def load_checkpoint():
    if not os.path.exists(CHECKPOINT):
        return (
            np.zeros(SHAPE, dtype=np.float64),
            np.zeros(SHAPE, dtype=np.int64),
            set()
        )

    with np.load(CHECKPOINT) as data:
        errors = data["errors"].copy()
        counts = data["counts"].copy()
        completed = set(data["completed"].astype(int).tolist())

    if errors.shape != SHAPE or counts.shape != SHAPE:
        raise RuntimeError("Checkpoint shape does not match settings.")

    return errors, counts, completed

def save_checkpoint(errors, counts, completed):
    temporary = CHECKPOINT + ".tmp.npz"
    np.savez_compressed(
        temporary,
        errors=errors,
        counts=counts,
        completed=np.array(sorted(completed), dtype=np.int64)
    )
    os.replace(temporary, CHECKPOINT)

def process_player(pid, filename):
    career = career_averages(pid)

    if REQUIRE_CAREER_FOR_COMPARISON and career is None:
        return None

    games = pd.read_csv(filename)
    if games.empty:
        return None

    games["GAME_DATE"] = pd.to_datetime(
        games["GAME_DATE"], errors="coerce"
    )
    games = games.dropna(subset=["GAME_DATE"])
    games = games.sort_values("GAME_DATE").reset_index(drop=True)

    # Aggregate per-player errors, avoiding huge per-game CSV files.
    player_errors = np.zeros(SHAPE, dtype=np.float64)
    player_counts = np.zeros(SHAPE, dtype=np.int64)

    for stat_index, stat in enumerate(STATS):
        values = pd.to_numeric(games[stat], errors="coerce").to_numpy(
            dtype=np.float64
        )

        for i in range(1, len(games)):
            date = games.at[i, "GAME_DATE"]
            if not (pd.Timestamp(START) <= date <= pd.Timestamp(END)):
                continue

            actual = values[i]
            if not np.isfinite(actual):
                continue

            previous = values[:i]
            valid_previous = previous[np.isfinite(previous)]

            if len(valid_previous) == 0:
                continue

            for window_index, window in enumerate(WINDOWS):
                # Match Weighted_model.py: take last N rows, then drop NaNs.
                recent = previous[-window:]
                season = previous[:-window]

                recent = recent[np.isfinite(recent)]
                season = season[np.isfinite(season)]

                r = float(recent.mean()) if len(recent) else np.nan
                s = float(season.mean()) if len(season) else np.nan
                c = career[stat] if career is not None else np.nan

                wr = WEIGHTS[:, 0]
                ws = WEIGHTS[:, 1]
                wc = WEIGHTS[:, 2]

                if np.isfinite(c):
                    if not np.isfinite(r):
                        predictions = np.full(N_WEIGHTS, c)
                        valid = np.ones(N_WEIGHTS, dtype=bool)
                    elif not np.isfinite(s):
                        denominator = wr + wc
                        valid = denominator > 0
                        predictions = np.full(N_WEIGHTS, np.nan)
                        predictions[valid] = (
                            wr[valid] * r + wc[valid] * c
                        ) / denominator[valid]
                    else:
                        predictions = wr*r + ws*s + wc*c
                        valid = np.ones(N_WEIGHTS, dtype=bool)
                else:
                    # Matches the career=0 branch of Weighted_model.py.
                    valid = wc == 0
                    denominator = (
                        np.where(np.isfinite(r), wr, 0)
                        + np.where(np.isfinite(s), ws, 0)
                    )
                    valid &= denominator > 0
                    predictions = np.full(N_WEIGHTS, np.nan)
                    predictions[valid] = (
                        np.where(np.isfinite(r), wr[valid]*r, 0)
                        + np.where(np.isfinite(s), ws[valid]*s, 0)
                    ) / denominator[valid]

                good = valid & np.isfinite(predictions)
                player_errors[stat_index, window_index, good] += np.abs(
                    predictions[good] - actual
                )
                player_counts[stat_index, window_index, good] += 1

    return player_errors, player_counts

def export_results(errors, counts):
    rows = []
    for stat_index, stat in enumerate(STATS):
        for window_index, window in enumerate(WINDOWS):
            for weight_index, (r, s, c) in enumerate(WEIGHT_INTS):
                n = int(counts[stat_index, window_index, weight_index])
                if n == 0:
                    continue
                mae = errors[stat_index, window_index, weight_index] / n
                rows.append((
                    stat, window, int(r), int(s), int(c), n, mae
                ))

    df = pd.DataFrame(rows, columns=[
        "stat", "window", "recent_pct", "season_pct",
        "career_pct", "games", "mae"
    ])

    df = df.sort_values(["stat", "mae", "window"])
    df.to_csv(RESULTS, index=False)

    # Require equal sample counts within each statistic for ranking.
    best_rows = []
    for stat, group in df.groupby("stat"):
        maximum_games = group["games"].max()
        comparable = group[group["games"] == maximum_games]
        if not comparable.empty:
            best_rows.append(comparable.iloc[0])

    best = pd.DataFrame(best_rows)
    best.to_csv(BEST, index=False)

    print("\nBEST CONFIGURATIONS (MAXIMUM COVERAGE)")
    print(best.to_string(index=False))
    print(f"\nSaved: {RESULTS}")
    print(f"Saved: {BEST}")

def main():
    errors, counts, completed = load_checkpoint()

    print(f"Players found: {len(FILES)}")
    print(f"Players completed: {len(completed)}")
    print(f"Configurations: {N_STATS*N_WINDOWS*N_WEIGHTS:,}")
    print("Offline optimization started.", flush=True)

    for number, (pid, filename) in enumerate(FILES, 1):
        if pid in completed:
            continue

        try:
            result = process_player(pid, filename)
        except Exception as exc:
            print(f"FAILED player {pid}: {exc}", flush=True)
            continue

        if result is not None:
            player_errors, player_counts = result
            errors += player_errors
            counts += player_counts

        completed.add(pid)
        save_checkpoint(errors, counts, completed)

        if number % 10 == 0:
            print(
                f"{number}/{len(FILES)} players processed",
                time.strftime("%H:%M:%S"),
                flush=True
            )

    export_results(errors, counts)

if __name__ == "__main__":
    main()
