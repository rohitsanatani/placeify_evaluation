"""
After rebuilding a city's test set, reuse sketches and text prompts from an earlier
version for any test image that is still in the new test set.

Places are matched by the test image's FILENAME (q-numbers can change between versions).
The earlier version is the one import_colab.py brought in:
    data/<city>/fixed_queries.json, data/<city>/sketches/qXX.png, data/<city>/text_prompts.csv

The first time it runs, it moves those into data/<city>/previous/ so nothing is lost, then writes
    data/<city>/sketches/qXX.png     only for places carried over, under their NEW q-number
    data/<city>/text_prompts.csv     carried-over prompts filled in, new places left empty
Step 3 then generates only the missing ones.

Usage:
    python build_test_set.py --city newyork
    python reuse_previous.py --city newyork
"""
import argparse
import csv
import json
import shutil

from common import CITIES, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=CITIES)
    args = ap.parse_args()
    out = paths(args.city)["out"]
    prev = out / "previous"

    if not prev.exists():
        if not (out / "fixed_queries.json").exists():
            raise SystemExit(f"nothing to reuse: {out / 'fixed_queries.json'} not found (run import_colab.py first)")
        prev.mkdir()
        for name in ["fixed_queries.json", "text_prompts.csv", "sketches"]:
            if (out / name).exists():
                shutil.move(str(out / name), str(prev / name))
        print(f"moved the earlier version to {prev}")

    old = {q["query"]["filename"]: q["query_id"] for q in json.load(open(prev / "fixed_queries.json"))["queries"]}
    old_prompts = {}
    if (prev / "text_prompts.csv").exists():
        old_prompts = {r["query_id"]: r.get("prompt", "") for r in csv.DictReader(open(prev / "text_prompts.csv"))}

    new = json.load(open(out / "test_set.json"))["queries"]
    sk = out / "sketches"
    sk.mkdir(exist_ok=True)
    for f in sk.glob("q*.png"):
        f.unlink()

    rows, reused, sketch_log = [], [], []
    for q in new:
        qid, fname = q["query_id"], q["query"]["filename"]
        text = ""
        if fname in old:
            o = old[fname]
            src = prev / "sketches" / f"{o}.png"
            if src.exists():
                shutil.copy2(src, sk / f"{qid}.png")
                sketch_log.append([qid, fname, "gpt-image-1 (Colab)", "1536x1024", "medium"])
            text = old_prompts.get(o, "")
            reused.append(f"{qid} <- old {o}")
        rows.append([qid, fname, text, len(text.split()), "gpt-5 (Colab)" if text else ""])
    with open(out / "text_prompts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "filename", "prompt", "word_count", "model"])
        w.writerows(rows)
    # tells generate_sketches.py which test image each reused sketch was drawn from
    with open(sk / "sketches.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "filename", "model", "size", "quality"])
        w.writerows(sketch_log)

    todo = [r[0] for r in rows if not r[2]]
    print(f"{len(reused)} of {len(new)} test places carried over (sketch + prompt reused):")
    for r in reused:
        print("  " + r)
    print(f"{len(todo)} need a new sketch and prompt in step 3: {' '.join(todo)}")


if __name__ == "__main__":
    main()
