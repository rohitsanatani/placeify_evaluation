"""
Step 2c: one review picture per test place, to check quality (open them in VS Code).

Each sheet shows the test photo (plus its sketch and text prompt once they exist),
then 10 thumbnails for every search in test_set.json and 10 for each random set,
with similarity scores. Searches added later (sketch/text) appear automatically.

Writes data/<city>/sheets/qXX.jpg

Usage:
    python make_contact_sheets.py --city newyork
    python make_contact_sheets.py --city newyork --only q03 q08
"""
import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common import CITIES, paths, all_places

COLORS = {"image_m2f": "darkgreen", "image_siglip": "darkblue", "sketch_m2f": "darkorange",
          "sketch_siglip": "purple", "text_siglip": "brown"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=CITIES)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    p = paths(args.city)
    out, img_dir = p["out"], p["img_dir"]
    J = json.load(open(out / "test_set.json"))
    prompts = {}
    if (out / "text_prompts.csv").exists():
        prompts = {r["query_id"]: r.get("prompt", "") for r in csv.DictReader(open(out / "text_prompts.csv"))}
    sheets = out / "sheets"
    sheets.mkdir(exist_ok=True)

    def load(fname):
        f = img_dir / fname
        return Image.open(f).convert("RGB") if f.exists() else None

    def show(ax, img, title, color="black"):
        if img is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", color="red")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=7, color=color)
        ax.axis("off")

    for q in all_places(J):
        qid = q["query_id"]
        if args.only and qid not in args.only:
            continue
        sets = [(n, lst, COLORS.get(n, "black")) for n, lst in q["results"].items()]
        sets += [(f"random_{n}", lst, "gray") for n, lst in q["random"].items()]
        nrows = 1 + 2 * len(sets)
        fig = plt.figure(figsize=(15, 2.2 * nrows + 1.5))
        gs = fig.add_gridspec(nrows, 5, height_ratios=[1.6] + [1] * (nrows - 1))
        show(fig.add_subplot(gs[0, 0:2]), load(q["query"]["filename"]),
             f"{qid} test photo  {q['query']['filename']}\n{q['query_summary']} (cluster size {q['cluster_size']})")
        sk = out / "sketches" / f"{qid}.png"
        show(fig.add_subplot(gs[0, 2:4]), Image.open(sk).convert("RGB") if sk.exists() else None, "sketch")
        ax = fig.add_subplot(gs[0, 4])
        ax.axis("off")
        ax.text(0, 0.5, "text prompt:\n" + (prompts.get(qid) or "(not made yet)"), wrap=True, fontsize=9, va="center")
        for k, (name, lst, color) in enumerate(sets):
            for j, c in enumerate(lst):
                sims = " ".join(f"{key[4:]} {v:.2f}" for key, v in c.items() if key.startswith("sim_"))
                show(fig.add_subplot(gs[1 + 2 * k + j // 5, j % 5]), load(c["filename"]),
                     f"{name} {j + 1}\n{sims}", color)
        fig.tight_layout()
        fig.savefig(sheets / f"{qid}.jpg", dpi=70)
        plt.close(fig)
        print("wrote", sheets / f"{qid}.jpg")


if __name__ == "__main__":
    main()
