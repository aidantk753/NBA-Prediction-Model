import pandas as pd

d = pd.read_csv("predictions_jan.csv")
d = d[d["mean"].notna() & d["std_dev"].notna() & (d["std_dev"] > 0)]
d = d[(d["mean"] != d["line"]) & (d["actual"] != d["line"])].copy()

d["correct"] = ((d["mean"] < d["line"]) & (d["actual"] < d["line"])) | \
               ((d["mean"] > d["line"]) & (d["actual"] > d["line"]))
d["err"] = (d["mean"] - d["actual"]).abs()
d["base_err"] = (d["hist_avg"] - d["actual"]).abs()
d["cfg"] = d["recent"].astype(str) + "/" + d["season"].astype(str) + "/" + d["career"].astype(str)

summary = []
for (stat, window), g in d.groupby(["stat", "window"]):
    n_cfg = g["cfg"].nunique()
    counts = g.groupby(["player_id", "date"])["cfg"].transform("nunique")
    common = g[counts == n_cfg]
    for cfg, c in common.groupby("cfg"):
        n = len(c)
        acc = c["correct"].mean()
        mae = c["err"].mean()
        base = c["base_err"].mean()
        summary.append({
            "stat": stat, "window": window, "cfg": cfg, "n": n,
            "acc%": round(acc * 100, 2),
            "se%": round((acc * (1 - acc) / n) ** 0.5 * 100, 2),
            "mae": round(mae, 4), "base_mae": round(base, 4),
            "improve%": round((base - mae) / base * 100, 2),
        })

s = pd.DataFrame(summary)
pd.set_option("display.width", 200)
for (stat, window), g in s.groupby(["stat", "window"]):
    print(f"\n=== {stat}, recent_games={window} (same player-games for every config) ===")
    print(g.sort_values("improve%", ascending=False).drop(columns=["stat", "window"]).to_string(index=False))

avg = s.groupby(["cfg", "window"])["improve%"].mean().reset_index()
print("\n=== Average MAE improvement across all 5 stats ===")
print(avg.sort_values("improve%", ascending=False).to_string(index=False))
