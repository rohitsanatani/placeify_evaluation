"""
Bring the New York work done in Colab onto the server, from a downloaded `study` folder.

Each study/test_set/qXX/ folder from Colab holds:
    query_<filename>.jpg   -> which dataset image is test place qXX
    sketch.png             -> its sketch
    text_prompt.txt        -> its text prompt

This writes, in data/<city>/:
    fixed_queries.json     the 20 test images with their q-numbers (for build_test_set.py --fixed-queries)
    sketches/qXX.png
    text_prompts.csv       query_id, filename, prompt, word_count

Usage:
    python import_colab.py --city newyork --study ~/Documents/researchers/ashley/study
    python build_test_set.py --city newyork --fixed-queries data/newyork/fixed_queries.json
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

from common import CITIES, load_city


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=CITIES)
    ap.add_argument("--study", required=True, help="the study folder (the one containing test_set/)")
    args = ap.parse_args()

    root = Path(args.study).expanduser()
    ts = root / "test_set" if (root / "test_set").is_dir() else root
    places = sorted(p for p in ts.iterdir() if p.is_dir() and p.name.startswith("q"))
    if not places:
        raise SystemExit(f"no qXX folders found in {ts}")

    d = load_city(args.city, verbose=False)
    known = set(d["locs"].filename)
    out = d["out"]
    (out / "sketches").mkdir(parents=True, exist_ok=True)

    queries, prompts, problems = [], [], []
    for p in places:
        q = sorted(p.glob("query_*"))
        if len(q) != 1:
            problems.append(f"{p.name}: expected one query_*.jpg, found {len(q)}")
            continue
        fname = q[0].name[len("query_"):]
        if fname not in known:
            problems.append(f"{p.name}: {fname} is not a {args.city} image")
        queries.append({"query_id": p.name, "query": {"filename": fname}})

        sk = p / "sketch.png"
        if sk.exists():
            shutil.copy2(sk, out / "sketches" / f"{p.name}.png")
        else:
            problems.append(f"{p.name}: no sketch.png")
        tp = p / "text_prompt.txt"
        text = tp.read_text().strip() if tp.exists() else ""
        if not text:
            problems.append(f"{p.name}: no text_prompt.txt")
        prompts.append([p.name, fname, text, len(text.split())])

    with open(out / "fixed_queries.json", "w") as f:
        json.dump({"meta": {"source": str(ts)}, "queries": queries}, f, indent=2)
    with open(out / "text_prompts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "filename", "prompt", "word_count"])
        w.writerows(prompts)

    n_sk = len(list((out / "sketches").glob("q*.png")))
    print(f"{len(queries)} test places imported from {ts}")
    print(f"  {out / 'fixed_queries.json'}\n  {n_sk} sketches in {out / 'sketches'}\n  "
          f"{sum(1 for r in prompts if r[2])} prompts in {out / 'text_prompts.csv'}")
    for r in prompts:
        print(f"  {r[0]} ({r[3]:>2} words): {r[2]}")
    if problems:
        print("\nPROBLEMS:\n  " + "\n  ".join(problems))
    else:
        print("\nAll good. Next:\n  python build_test_set.py --city "
              f"{args.city} --fixed-queries {out / 'fixed_queries.json'}")


if __name__ == "__main__":
    main()
