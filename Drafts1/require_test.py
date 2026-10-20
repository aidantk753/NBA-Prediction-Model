from league_backtest import league_backtest

w = {"recent": 0.5, "season": 0.5, "career": 0}

for flag in (False, True):
    r = league_backtest("PTS", "2026-02-01", "2026-02-28", show_results=False,
                        recent_games=5, weights=w, max_players=50, require_career=flag)
    print(f"require_career={flag}: total={r['total']} skipped={r['skipped']} complete={r['complete']}")
