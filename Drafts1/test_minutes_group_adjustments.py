import pandas as pd
import numpy as np

TRAIN_END = pd.Timestamp("2025-02-28")
TEST_START = pd.Timestamp("2025-03-01")
TEST_END = pd.Timestamp("2025-04-13")

STATUS_ADJUSTMENTS = {
    "Questionable": -1.41,
    "Probable": -0.53,
    "Available": -0.40,
}

# Load existing results
minutes = pd.read_csv("actual_injury_minutes_errors.csv")
points = pd.read_csv("injury_points_comparison.csv")

for df in (minutes, points):
    df["game_day"] = pd.to_datetime(df["game_day"])
    df["nba_id"] = pd.to_numeric(df["nba_id"])

# Join actual minutes and points predictions
df = minutes.merge(
    points[
        ["nba_id", "game_day", "original", "actual_pts"]
    ],
    on=["nba_id", "game_day"],
    how="inner",
    validate="one_to_one"
)

df = df[
    df["game_day"].between("2025-01-01", "2025-04-13")
].copy()

df["minutes_group"] = pd.cut(
    df["projected_minutes"],
    bins=[0, 15, 25, 35, np.inf],
    labels=["0-15", "15-25", "25-35", "35+"],
    include_lowest=True
)

# Chronological split
train = df[df["game_day"] <= TRAIN_END].copy()
test = df[
    df["game_day"].between(TEST_START, TEST_END)
].copy()

# Learn adjustments using ONLY January-February
group_adjustments = (
    train.groupby(
        ["status", "minutes_group"],
        observed=True
    )["minutes_error"]
    .agg(["mean", "count"])
    .reset_index()
)

# Shrink small-group estimates toward the existing status adjustment
PRIOR_STRENGTH = 30

group_adjustments["prior"] = (
    group_adjustments["status"]
    .map(STATUS_ADJUSTMENTS)
)

group_adjustments["learned_adjustment"] = (
    group_adjustments["count"] * group_adjustments["mean"]
    + PRIOR_STRENGTH * group_adjustments["prior"]
) / (
    group_adjustments["count"] + PRIOR_STRENGTH
)

lookup = {
    (row["status"], str(row["minutes_group"])):
    row["learned_adjustment"]
    for _, row in group_adjustments.iterrows()
}

test["status_adjustment"] = (
    test["status"].map(STATUS_ADJUSTMENTS)
)

test["group_adjustment"] = test.apply(
    lambda row: lookup.get(
        (row["status"], str(row["minutes_group"])),
        STATUS_ADJUSTMENTS[row["status"]]
    ),
    axis=1
)

# Minutes predictions
test["minutes_baseline"] = test["projected_minutes"]

test["minutes_status"] = (
    test["projected_minutes"] + test["status_adjustment"]
).clip(lower=0)

test["minutes_group_model"] = (
    test["projected_minutes"] + test["group_adjustment"]
).clip(lower=0)

# Points predictions
test["pts_baseline"] = test["original"]

test["pts_status"] = (
    test["original"]
    * test["minutes_status"]
    / test["projected_minutes"]
)

test["pts_group_model"] = (
    test["original"]
    * test["minutes_group_model"]
    / test["projected_minutes"]
)

def evaluate(data, actual, columns):
    results = []

    for name, col in columns.items():
        error = (data[col] - data[actual]).abs()

        results.append({
            "Model": name,
            "Games": len(data),
            "MAE": round(error.mean(), 4)
        })

    return pd.DataFrame(results)

print("\nTRAINING DATA")
print(f"January-February games: {len(train)}")

print("\nLEARNED MINUTES ADJUSTMENTS")
print(
    group_adjustments[
        [
            "status", "minutes_group",
            "count", "mean", "learned_adjustment"
        ]
    ].round(3).to_string(index=False)
)

print("\nHOLDOUT: MARCH 1 - APRIL 13")
print(f"Games: {len(test)}")

print("\nMINUTES MAE")
print(
    evaluate(
        test,
        "actual_minutes",
        {
            "Baseline": "minutes_baseline",
            "Status only": "minutes_status",
            "Status + minutes group": "minutes_group_model"
        }
    ).to_string(index=False)
)

print("\nPOINTS MAE")
print(
    evaluate(
        test,
        "actual_pts",
        {
            "Baseline": "pts_baseline",
            "Status only": "pts_status",
            "Status + minutes group": "pts_group_model"
        }
    ).to_string(index=False)
)

print("\nPOINTS MAE BY INJURY STATUS")
for status, group in test.groupby("status"):
    print(f"\n{status} ({len(group)} games)")
    print(
        evaluate(
            group,
            "actual_pts",
            {
                "Baseline": "pts_baseline",
                "Status only": "pts_status",
                "Status + minutes group": "pts_group_model"
            }
        ).to_string(index=False)
    )

test.to_csv(
    "minutes_group_holdout_predictions.csv",
    index=False
)

group_adjustments.to_csv(
    "minutes_group_learned_adjustments.csv",
    index=False
)

print("\nSaved: minutes_group_holdout_predictions.csv")
print("Saved: minutes_group_learned_adjustments.csv")
