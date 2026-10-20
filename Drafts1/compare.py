from league_backtest import league_backtest

for w in [{"recent":.5,"season":.5,"career":0}, {"recent":.75,"season":.25,"career":0},
          {"recent":1,"season":0,"career":0}, {"recent":.5,"season":0,"career":.5}]:
    r = league_backtest("PTS", "2026-02-01", "2026-02-28", show_results=False,
                        recent_games=5, weights=w, require_career=True)
    print("RESULT", w, r["total"], f"{r['accuracy']:.2f}%",
          f"mae {r['league_mae']:.3f} vs base {r['league_baseline_mae']:.3f}", "complete", r["complete"])
