import os
import time
import pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import commonallplayers
from Weighted_model import weighted_model
from nba_cache_utils import get_cached_game_logs, CACHE_DIR

# ---------------- settings ----------------
START = "2025-11-01"
END = "2026-04-30"
STATS = ["Min"]
WINDOWS = [8, 10, 12, 15, 20]
WEIGHT_VALUES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
REQUIRE_CAREER = True
MAX_PLAYERS = None          # e.g. 20 for a quick test
OUT = "predictions_minutes.csv"
DONE = "done_minutes.txt"
# ------------------------------------------

CONFIGS = [
    (r, s, c)
    for r in WEIGHT_VALUES for s in WEIGHT_VALUES for c in WEIGHT_VALUES
    if abs(r + s + c - 1.0) < 1e-9 and c == 0 and r >= 0.5
]

start_date = datetime.strptime(START, "%Y-%m-%d").date()
end_date = datetime.strptime(END, "%Y-%m-%d").date()

players_file = os.path.join(CACHE_DIR, "players_2025-26.csv")
if os.path.exists(players_file):
    player_data = pd.read_csv(players_file)
else:
    player_data = commonallplayers.CommonAllPlayers(
        is_only_current_season=1, league_id="00",
        season="2025-26", timeout=10
    ).get_data_frames()[0]
    os.makedirs(CACHE_DIR, exist_ok=True)
    player_data.to_csv(players_file, index=False)

player_data = player_data.drop_duplicates(subset="PERSON_ID")
if MAX_PLAYERS:
    player_data = player_data.sample(n=min(MAX_PLAYERS, len(player_data)), random_state=42)

done = set()
if os.path.exists(DONE):
    done = set(open(DONE).read().split())

print(f"{len(CONFIGS)} weight configs x {len(WINDOWS)} windows x {len(STATS)} stats")
print(f"{len(done)} players already done")

failed = []


def process_player(pid):
    logs = get_cached_game_logs(pid, "2025-26").copy()
    if logs.empty or "GAME_DATE" not in logs.columns:
        return []
    logs["GAME_DATE"] = logs["GAME_DATE"].apply(
        lambda d: datetime.strptime(d, "%b %d, %Y").date()
    )
    logs = logs.sort_values("GAME_DATE").reset_index(drop=True)

    rows = []
    for i in range(len(logs)):
        game_date = logs.loc[i, "GAME_DATE"]
        if not (start_date <= game_date <= end_date):
            continue
        date_str = game_date.strftime("%Y-%m-%d")

        for stat in STATS:
            if stat not in logs.columns:
                continue
            prev = logs.iloc[:i][stat].dropna().tolist()
            actual = logs.loc[i, stat]
            if not prev or actual != actual:
                continue
            hist_avg = sum(prev) / len(prev)
            line = round(hist_avg - 0.5) + 0.5

            for window in WINDOWS:
                for r, s, c in CONFIGS:
                    mean, std = weighted_model(
                        logs, stat, date_str, pid,
                        recent_games=window,
                        weights={"recent": r, "season": s, "career": c},
                        require_career=REQUIRE_CAREER
                    )
                    rows.append({
                        "player_id": pid, "date": date_str, "stat": stat,
                        "window": window, "recent": r, "season": s, "career": c,
                        "mean": mean, "std_dev": std,
                        "line": line, "hist_avg": hist_avg, "actual": actual
                    })
    return rows


total = len(player_data)
for n, player in enumerate(player_data.itertuples(), start=1):
    pid = int(player.PERSON_ID)
    if str(pid) in done:
        continue

    cold = not (
        os.path.exists(os.path.join(CACHE_DIR, f"{pid}_2025-26_games.csv"))
        and os.path.exists(os.path.join(CACHE_DIR, f"{pid}_career_2025-26.csv"))
    )
    if cold:
        time.sleep(0.6)

    rows = None
    for attempt in range(3):
        try:
            rows = process_player(pid)
            break
        except Exception as error:
            if attempt == 2:
                print(f"FAILED {player.DISPLAY_FIRST_LAST}: {error}")
                failed.append(player.DISPLAY_FIRST_LAST)
            else:
                time.sleep(2 ** attempt)

    if rows is None:
        continue

    if rows:
        pd.DataFrame(rows).to_csv(
            OUT, mode="a", index=False,
            header=not os.path.exists(OUT)
        )
    with open(DONE, "a") as f:
        f.write(f"{pid}\n")

    if n % 25 == 0:
        print(f"{n}/{total} players", time.strftime("%H:%M:%S"))

print(f"\nFinished. Failed players: {len(failed)}")
for name in failed:
    print(" ", name)
