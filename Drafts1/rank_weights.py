import pandas as pd


stats = ["PTS", "REB", "AST", "STL", "BLK"]

df = pd.read_csv("backtest_results.csv")

df = df[df["source"] == "weight_optimizer"].copy()

df = df.sort_values("timestamp").drop_duplicates(
    subset=[
        "recent", "season", "career",
        "recent_games", "start_date", "end_date",
        "stat_category"
    ],
    keep="last"
)

df = df.pivot_table(
    index=[
        "recent", "season", "career",
        "recent_games", "start_date", "end_date"
    ],
    columns="stat_category",
    values=["accuracy", "mae", "baseline_mae", "correct", "total"],
    aggfunc="first"
)

df.columns = [
    f"{stat}_{metric}"
    for metric, stat in df.columns
]

df = df.reset_index()
df = df.dropna(
    subset=[f"{stat}_accuracy" for stat in stats]
)

df = df[
    (df["recent"] + df["season"] + df["career"] - 1.0).abs() < 1e-9
].copy()

accuracy_columns = [f"{stat}_accuracy" for stat in stats]
df["average_accuracy"] = df[accuracy_columns].mean(axis=1)
correct_columns = [f"{stat}_correct" for stat in stats]
total_columns = [f"{stat}_total" for stat in stats]

df["overall_accuracy"] = (
    df[correct_columns].sum(axis=1)
    / df[total_columns].sum(axis=1).replace(0, float("nan"))
) * 100
ranked = df.sort_values("average_accuracy", ascending=False)

# Rank Best Overall Average
print("Best Average Accuracy (mean of all stats): ")
print(ranked[[
    "recent",
    "season",
    "career",
    "recent_games",
    "average_accuracy"
]].head(10).to_string(index=False))

# Rank Best Overall Accuracy (All Predictions)
ranked = df.sort_values("overall_accuracy", ascending=False)

print("Best Overall Accuracy (All Predictions):")
print(ranked[[
    "recent",
    "season",
    "career",
    "recent_games",
    "overall_accuracy"
]].head(10).to_string(index=False))

# Rank Individual Stats
for stat in stats:
    column = f"{stat}_accuracy"
    ranked = df.sort_values(column, ascending=False)

    print(f"Best {stat} Accuracy: ")
    print(ranked[[
        "recent",
        "season",
        "career",
        "recent_games",
        "average_accuracy",
        column
    ]].head(10).to_string(index=False))

# Rank lowest MAE for each statistic
for stat in stats:
    column = f"{stat}_mae"
    ranked = df.sort_values(column, ascending=True)

    print(f"Lowest {stat} MAE: ")
    print(ranked[[
        "recent",
        "season",
        "career",
        "recent_games",
        column
    ]].head(10).to_string(index=False))

# Calculate MAE improvement over baseline
# Calculate MAE improvement over baseline
for stat in stats:
    df[f"{stat}_improvement"] = (
        (df[f"{stat}_baseline_mae"] - df[f"{stat}_mae"])
        / df[f"{stat}_baseline_mae"]
    ) * 100

    ranked = df.sort_values(f"{stat}_improvement", ascending=False)

    print(f"Best {stat} MAE performance: ")
    print(ranked[[
        "recent",
        "season",
        "career",
        "recent_games",
        f"{stat}_improvement"
    ]].head(10).to_string(index=False))
improvement_columns = [f"{stat}_improvement" for stat in stats]

df["average_improvement"] = df[improvement_columns].mean(axis=1)

ranked = df.sort_values("average_improvement", ascending=False)

print("Best MAE performance: ")
print(ranked[[
    "recent",
    "season",
    "career",
    "recent_games",
    "average_improvement"
]].head(10).to_string(index=False))