"""
Step 3b: turn each test image into a line-drawing sketch with GPT-Image.

Reads data/<city>/test_set.json and writes data/<city>/sketches/qXX.png plus
data/<city>/sketches/sketches.csv (query_id, filename, model, size, quality). Only places WITHOUT a
valid sketch are sent to the API: a sketch counts as valid if the .png exists and sketches.csv says
it was drawn from the same test image (filename) as now. Re-running is free; if a test set changes,
only the changed places are redrawn.

The OpenAI key comes from OPENAI_API_KEY in .env.

Usage:
    python generate_sketches.py --city boston
    python generate_sketches.py --city boston --only q03 q07     # redraw these
"""
import argparse
import base64
import csv
import json
import time

from common import CITIES, paths, all_places
from generate_text_prompts import client

INSTRUCTION = (
    "Redraw this street photo as a simple black-and-white line drawing, like an architect's quick pen sketch. "
    "Use clean black outlines on a plain white background, with no shading, color, or texture fill. "
    "Keep the same viewpoint, composition, and main elements: the road, sidewalks, buildings, trees, vehicles, "
    "and other large objects in the same places. Leave out small details, people's faces, all text, signs, "
    "logos, and watermarks."
)


def draw(cl, model, path, size, quality):
    with open(path, "rb") as f:
        resp = cl.images.edit(model=model, image=f, prompt=INSTRUCTION, size=size, quality=quality)
    return base64.b64decode(resp.data[0].b64_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=CITIES)
    ap.add_argument("--only", nargs="*", help="draw ONLY these places, e.g. --only q03 q07")
    ap.add_argument("--model", default="gpt-image-1")
    ap.add_argument("--size", default="1536x1024")
    ap.add_argument("--quality", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--tries", type=int, default=3)
    args = ap.parse_args()

    p = paths(args.city)
    out, img_dir = p["out"], p["img_dir"]
    queries = all_places(json.load(open(out / "test_set.json")))   # 20 test places + practice p00
    sk = out / "sketches"
    sk.mkdir(exist_ok=True)
    log_path = sk / "sketches.csv"
    log = {}
    if log_path.exists():
        log = {r["query_id"]: r for r in csv.DictReader(open(log_path))}

    ids = [q["query_id"] for q in queries]
    if args.only:   # exactly these places, nothing else
        unknown = [x for x in args.only if x not in ids]
        if unknown:
            raise SystemExit(f"--only: no such places {unknown}")
        todo = [x for x in ids if x in args.only]
    else:           # every place without a valid sketch
        todo = []
        for q in queries:
            qid = q["query_id"]
            r = log.get(qid)
            valid = (sk / f"{qid}.png").exists() and r and r.get("filename") == q["query"]["filename"]
            if not valid:
                todo.append(qid)
    if args.only:
        print(f"{args.city}: --only, so drawing just: {' '.join(todo)}")
    else:
        print(f"{args.city}: {len(queries) - len(todo)} sketches already done, {len(todo)} to draw: {' '.join(todo)}")

    cl = client() if todo else None
    for q in queries:
        qid, fname = q["query_id"], q["query"]["filename"]
        if qid not in todo:
            continue
        (sk / f"{qid}.png").unlink(missing_ok=True)
        log.pop(qid, None)
        for attempt in range(args.tries):
            try:
                png = draw(cl, args.model, img_dir / fname, args.size, args.quality)
                (sk / f"{qid}.png").write_bytes(png)
                log[qid] = {"query_id": qid, "filename": fname, "model": args.model,
                            "size": args.size, "quality": args.quality}
                print(f"  {qid}: saved")
                break
            except Exception as e:
                print(f"  {qid}: error ({e})" + ("; retrying" if attempt < args.tries - 1 else "; giving up"))
                time.sleep(10)
        # save the log after every sketch so a crash never loses finished ones
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["query_id", "filename", "model", "size", "quality"])
            w.writeheader()
            w.writerows(log[x["query_id"]] for x in queries if x["query_id"] in log)

    (sk / "instruction.txt").write_text(f"model: {args.model}, size {args.size}, quality {args.quality}\n\n{INSTRUCTION}\n")
    missing = [q["query_id"] for q in queries if not (sk / f"{q['query_id']}.png").exists()]
    print(f"\n{len(queries) - len(missing)} of {len(queries)} sketches in {sk}"
          + (f"; still missing {missing} (re-run to retry)" if missing else ""))


if __name__ == "__main__":
    main()
