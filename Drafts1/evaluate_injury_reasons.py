import pandas as pd
import numpy as np

DATA = "injury_validation_predictions.csv"
SMOOTHING = 30

d = pd.read_csv(DATA)
d["game_day"] = pd.to_datetime(d["game_day"])

def reason_category(reason):
    reason = str(reason).strip()
    if reason.startswith("Injury/Illness"):
        return "Injury/Illness"
    if reason.startswith("G League"):
        return "G League"
    if reason in ("-", "nan", ""):
        return "Other/Unspecified"
    return "Other"

d["reason_category"] = d["reason"].apply(reason_category)

# The saved validation file contains test observations only.
# Reconstructing training data requires the original injury reports
# and cached game logs, so do not fit on this file.

print("Reason categories in held-out predictions:")
print(d.groupby(["status", "reason_category"]).agg(
    observations=("played", "size"),
    appearance_rate=("played", "mean")
).to_string())

print("\nThis is descriptive analysis, not a new model evaluation.")
