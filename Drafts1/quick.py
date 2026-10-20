import pandas as pd
d = pd.read_csv("backtest_results.csv")
d = d[(d.source=="weight_optimizer") & (d.stat_category=="PTS")]
print(d[["recent","season","career","recent_games","total"]].sort_values("total"))