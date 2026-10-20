import csv
import os
from datetime import datetime

def save_results(weights, results, source):
    filename = os.path.join(
        os.path.dirname(__file__),
        "backtest_results.csv"
    )

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        **weights,
        **results
    }

    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0

    if file_exists:
        with open(filename, "r", newline="") as file:
            reader = csv.DictReader(file)
            existing_columns = reader.fieldnames or []
            existing_rows = list(reader)

        columns = list(dict.fromkeys(existing_columns + list(row.keys())))

        if columns != existing_columns:
            with open(filename, "w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
                writer.writerows(existing_rows)
    else:
        columns = list(row.keys())

    with open(filename, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)