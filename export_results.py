"""
Export ratings from all sessions (finished AND unfinished) to one CSV.

The app already appends finished sessions to results/<combo>/ratings.csv; this also picks up
people who stopped partway, which is useful for checking progress or dropout.

Usage:
    python export_results.py                     # every combination -> results/all_ratings.csv
    python export_results.py --finished-only
"""
import argparse
import csv
import json

from app import CSV_COLUMNS, trial_row
from common import HERE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finished-only", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results" / "all_ratings.csv"))
    args = ap.parse_args()

    files = sorted((HERE / "results").glob("*/sessions/*.json"))
    n_rows = n_sessions = n_finished = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS + ["session_finished"])
        for path in files:
            s = json.loads(path.read_text())
            done = bool(s.get("finished_at"))
            n_sessions += 1
            n_finished += done
            if args.finished_only and not done:
                continue
            for t in s["trials"]:
                if t["rating"] is None:
                    continue
                w.writerow(trial_row(s, t) + [done])
                n_rows += 1
    print(f"{n_sessions} sessions ({n_finished} finished), {n_rows} ratings -> {args.out}")


if __name__ == "__main__":
    main()
