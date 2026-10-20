import pandas as pd

FILE = "predictions_big.csv"
STATS = ["PTS", "REB", "AST", "STL", "BLK"]
COLS = ["player_id", "date", "stat", "window", "recent", "season",
        "mean", "std_dev", "line", "hist_avg", "actual"]
SPLIT = "2026-03-01"   # configs are picked before this date, checked after it

pd.set_option("display.width", 200)


def load(stat):
    parts = []
    for chunk in pd.read_csv(FILE, usecols=COLS, chunksize=500000):
        parts.append(chunk[chunk["stat"] == stat])
    d = pd.concat(parts)
    d = d[d["mean"].notna() & d["std_dev"].notna() & (d["std_dev"] > 0)]
    d = d[(d["mean"] != d["line"]) & (d["actual"] != d["line"])].copy()
    d["correct"] = ((d["mean"] < d["line"]) & (d["actual"] < d["line"])) | \
                   ((d["mean"] > d["line"]) & (d["actual"] > d["line"]))
    d["err"] = (d["mean"] - d["actual"]).abs()
    d["base_err"] = (d["hist_avg"] - d["actual"]).abs()
    d["gain"] = d["base_err"] - d["err"]
    d["cfg"] = d["recent"].astype(str) + "/" + d["season"].astype(str)

    # keep only player-games where every config produced a prediction
    parts = []
    for window, g in d.groupby("window"):
        n_cfg = g["cfg"].nunique()
        counts = g.groupby(["player_id", "date"])["cfg"].transform("nunique")
        parts.append(g[counts == n_cfg])
    return pd.concat(parts)


def summarize(d):
    out = d.groupby(["window", "cfg"]).agg(
        n=("gain", "size"), acc=("correct", "mean"), mae=("err", "mean"),
        base_mae=("base_err", "mean"), gain=("gain", "mean"), sd=("gain", "std")
    ).reset_index()
    out["improve%"] = (out["gain"] / out["base_mae"] * 100).round(3)
    out["se%"] = (out["sd"] / out["n"] ** 0.5 / out["base_mae"] * 100).round(3)
    out["acc%"] = (out["acc"] * 100).round(2)
    out["t"] = (out["improve%"] / out["se%"]).round(1)
    out["mae"] = out["mae"].round(4)
    out["base_mae"] = out["base_mae"].round(4)
    return out[["window", "cfg", "n", "acc%", "mae", "base_mae", "improve%", "se%", "t"]]


all_full, all_val = [], []
for stat in STATS:
    d = load(stat)
    if d.empty:
        print(f"\n##### {stat}: no data")
        continue
    full = summarize(d)
    sel = summarize(d[d["date"] < SPLIT])
    val = summarize(d[d["date"] >= SPLIT])
    m = sel.merge(val, on=["window", "cfg"], suffixes=("_sel", "_val"))

    print(f"\n##### {stat}: best 6 configs, all months (same player-games for every config)")
    print(full.sort_values("improve%", ascending=False).head(6).to_string(index=False))
    print(f"\n##### {stat}: top 5 picked on Nov-Feb, then checked on Mar-Apr")
    print(m.sort_values("improve%_sel", ascending=False).head(5)[
        ["window", "cfg", "n_sel", "improve%_sel", "n_val", "improve%_val", "se%_val", "acc%_val"]
    ].to_string(index=False))

    full["stat"] = stat
    val["stat"] = stat
    all_full.append(full)
    all_val.append(val)

if all_full:
    f = pd.concat(all_full)
    v = pd.concat(all_val)
    avg = f.groupby(["window", "cfg"])["improve%"].mean().round(3).reset_index()
    avgv = v.groupby(["window", "cfg"])["improve%"].mean().round(3).reset_index()
    avg = avg.merge(avgv, on=["window", "cfg"], suffixes=("_all", "_MarApr"))
    print("\n##### Average MAE improvement across the 5 stats (best 10)")
    print(avg.sort_values("improve%_all", ascending=False).head(10).to_string(index=False))
