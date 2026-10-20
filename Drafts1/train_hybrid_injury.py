import pandas as pd
from evaluate_injuries import build_dataset
from compare_injury_models import reason_category

SMOOTHING = 30
TRAINING_END = "2025-04-13"

data = build_dataset()
data = data[data["game_day"] <= TRAINING_END].copy()
data["reason_category"] = data["reason"].apply(reason_category)

# Baseline: status probabilities
status_stats = data.groupby("status")["played"].agg(["sum", "count"])
status_stats["probability"] = (
    (status_stats["sum"] + 1) / (status_stats["count"] + 2)
)
status_stats.reset_index().to_csv(
    "hybrid_status_probabilities.csv", index=False
)

status_probs = status_stats["probability"].to_dict()

# Status + reason probabilities
reason_stats = data.groupby(
    ["status", "reason_category"]
)["played"].agg(["sum", "count"]).reset_index()

reason_stats["probability"] = reason_stats.apply(
    lambda row: (
        row["sum"] + SMOOTHING * status_probs[row["status"]]
    ) / (row["count"] + SMOOTHING),
    axis=1
)

reason_stats.to_csv(
    "hybrid_reason_probabilities.csv", index=False
)

reason_probs = {
    (row.status, row.reason_category): row.probability
    for row in reason_stats.itertuples()
}

# Questionable only: status + reason + previous status + gap
questionable = data[data["status"] == "Questionable"]

history_stats = questionable.groupby(
    ["status", "reason_category", "previous_status", "gap_category"]
)["played"].agg(["sum", "count"]).reset_index()

history_stats["probability"] = history_stats.apply(
    lambda row: (
        row["sum"] + SMOOTHING *
        reason_probs[(row["status"], row["reason_category"])]
    ) / (row["count"] + SMOOTHING),
    axis=1
)

history_stats.to_csv(
    "hybrid_history_probabilities.csv", index=False
)

print("\n========== HYBRID MODEL TRAINED ==========")
print("Training observations:", len(data))
print("Training through:", TRAINING_END)
print("Status groups:", len(status_stats))
print("Status + reason groups:", len(reason_stats))
print("Questionable history groups:", len(history_stats))
print("\nSaved three hybrid probability tables.")
