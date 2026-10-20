from league_backtest import league_backtest

configs = [
    {"recent": 1.0, "season": 0.0, "career": 0.0},
    {"recent": 0.0, "season": 0.0, "career": 1.0},
    {"recent": 0.65, "season": 0.25, "career": 0.10},
]

for w in configs:
    r = league_backtest("PTS", "2026-02-01", "2026-02-28",
                        show_results=False, recent_games=10,
                        weights=w, max_players=50)
    print(w)
    print(f"  complete={r['complete']}  total={r['total']}  skipped={r['skipped']}")
    print(f"  accuracy={r['accuracy']:.2f}%  mae={r['league_mae']:.4f}  baseline={r['league_baseline_mae']:.4f}")