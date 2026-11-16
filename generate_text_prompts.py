"""
Step 3a: a short text description (<= 15 words) of each test image, written by GPT-5.

Reads data/<city>/test_set.json and writes data/<city>/text_prompts.csv
(query_id, filename, prompt, word_count, model). Only places WITHOUT a valid prompt are sent to
the API: a prompt counts as valid if it's non-empty and was written for the same test image
(filename) as now. So re-running is free, and if a test set changes only the changed places are redone.

The OpenAI key comes from OPENAI_API_KEY in .env.

Usage:
    python generate_text_prompts.py --city boston
    python generate_text_prompts.py --city boston --only q03 q07     # redo these
"""
import argparse
import base64
import csv
import json
import os
import time

from common import CITIES, paths, all_places

INSTRUCTION = (
    "Describe this street scene in at most 15 words. Focus on what is visible: the type of "
    "street, buildings, vegetation, vehicles, and other notable physical features. Do not name "
    "the city, neighborhood, specific landmarks, businesses, or brands, and do not mention the "
    "camera, the photo, or any watermark. Reply with the description only."
)
MAX_WORDS = 15


def client():
    from openai import OpenAI
    key = (os.environ.get("OPENAI_API_KEY") or "").strip().strip('"').strip("'")
    if not key:
        raise SystemExit("OPENAI_API_KEY is empty - add it to .env")
    return OpenAI(api_key=key)


def describe(cl, model, path):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    resp = cl.responses.create(model=model, input=[{"role": "user", "content": [
        {"type": "input_text", "text": INSTRUCTION},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}]}])
    return resp.output_text.strip().strip('"').strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=CITIES)
    ap.add_argument("--only", nargs="*", help="write ONLY these places, e.g. --only q03 q07")
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--tries", type=int, default=3)
    args = ap.parse_args()

    p = paths(args.city)
    out, img_dir = p["out"], p["img_dir"]
    queries = all_places(json.load(open(out / "test_set.json")))   # 20 test places + practice p00
    csv_path = out / "text_prompts.csv"
    existing = {}
    if csv_path.exists():
        existing = {r["query_id"]: r for r in csv.DictReader(open(csv_path))}

    ids = [q["query_id"] for q in queries]
    if args.only:   # exactly these places, nothing else
        unknown = [x for x in args.only if x not in ids]
        if unknown:
            raise SystemExit(f"--only: no such places {unknown}")
        todo = [x for x in ids if x in args.only]
    else:           # every place without a valid prompt
        todo = []
        for q in queries:
            r = existing.get(q["query_id"])
            valid = r and r.get("prompt", "").strip() and r.get("filename") == q["query"]["filename"]
            if not valid:
                todo.append(q["query_id"])
    if args.only:
        print(f"{args.city}: --only, so writing just: {' '.join(todo)}")
    else:
        print(f"{args.city}: {len(queries) - len(todo)} prompts already done, {len(todo)} to write: {' '.join(todo)}")

    # one row per place, in test-set order; valid earlier prompts are kept, the rest start empty
    rows = {}
    for q in queries:
        qid, fname = q["query_id"], q["query"]["filename"]
        r = existing.get(qid, {})
        keep = qid not in todo
        text = r.get("prompt", "") if keep else ""
        rows[qid] = [qid, fname, text, len(text.split()), (r.get("model") or "gpt-5 (earlier run)") if keep else ""]

    def save():
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["query_id", "filename", "prompt", "word_count", "model"])
            w.writerows(rows.values())

    save()
    cl = client() if todo else None
    for qid in todo:
        fname = rows[qid][1]
        text = ""
        for attempt in range(args.tries):
            try:
                text = describe(cl, args.model, img_dir / fname)
            except Exception as e:
                print(f"  {qid}: API error ({e}); retrying")
                time.sleep(5)
                continue
            if 0 < len(text.split()) <= MAX_WORDS:
                break
            print(f"  {qid}: {len(text.split())} words, retrying")
        n = len(text.split())
        flag = "" if 0 < n <= MAX_WORDS else "   <-- CHECK: over the limit or empty"
        print(f"  {qid} ({n} words): {text}{flag}")
        rows[qid] = [qid, fname, text, n, args.model]
        save()   # after every place, so a crash never loses finished prompts
    rows = list(rows.values())

    (out / "text_prompts_instruction.txt").write_text(f"model: {args.model}\nmax words: {MAX_WORDS}\n\n{INSTRUCTION}\n")
    missing = [r[0] for r in rows if not r[2]]
    print(f"\nsaved {csv_path}: {len(rows) - len(missing)} of {len(rows)} prompts"
          + (f"; still missing {missing} (re-run to retry)" if missing else ""))


if __name__ == "__main__":
    main()
