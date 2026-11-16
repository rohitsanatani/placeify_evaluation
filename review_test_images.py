"""
Show a city's test images (q00-q19 + practice p00) large, to check for blurred-out regions
(Street View privacy blur), glitches or anything else that makes a bad test image.

Writes data/<city>/review/test_images_1.jpg, _2.jpg ... (6 per page, near full resolution).
Open them in VS Code and zoom in.

If one is blurry, add its filename (printed in the title) to data/<city>/blurry.txt and re-run
build_test_set.py: that place then uses the next-closest image to its centroid.

Usage:
    python review_test_images.py --city boston
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common import CITIES, all_places, paths

PER_PAGE = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=CITIES)
    args = ap.parse_args()
    p = paths(args.city)
    places = all_places(json.load(open(p["out"] / "test_set.json")))
    review = p["out"] / "review"
    review.mkdir(exist_ok=True)
    for old in review.glob("test_images_*.jpg"):
        old.unlink()

    for page in range(0, len(places), PER_PAGE):
        chunk = places[page:page + PER_PAGE]
        rows = (len(chunk) + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(17, 5.9 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, q in zip(axes.flat, chunk):
            f = q["query"]["filename"]
            img = p["img_dir"] / f
            if img.exists():
                ax.imshow(Image.open(img).convert("RGB"))
            else:
                ax.text(0.5, 0.5, "missing", ha="center", color="red")
            tag = "PRACTICE " if q.get("practice") else ""
            ax.set_title(f"{tag}{q['query_id']}   {f}\n{q['query_summary']}", fontsize=11)
        fig.tight_layout()
        out = review / f"test_images_{page // PER_PAGE + 1}.jpg"
        fig.savefig(out, dpi=90)
        plt.close(fig)
        print("wrote", out)
    print(f"\nIf any are blurry: add the filename to {p['out'] / 'blurry.txt'}, then run "
          f"build_test_set.py --city {args.city}")


if __name__ == "__main__":
    main()
