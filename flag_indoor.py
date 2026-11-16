"""
Step 2a: flag indoor photos and tunnels so they're removed from the whole dataset.

Only views taken fully INSIDE are removed. Outdoor views stay, even under a bridge or
elevated tracks, beside a wall, or at a tunnel entrance. Using Mask2Former class shares,
with "street" = building, house, skyscraper, tree, grass, plant, sidewalk, car, sky,
truck, bus, van, fence, streetlight, pole, signboard, traffic light:

    tunnel : ceiling >= 15%, sky < 3%, street < 15%
    room   : (floor >= 10% or ceiling >= 10%), sky < 2%, street < 15%
    blank  : sky < 2% and street < 5%   (nothing outdoors visible: indoor shots, a wall filling the frame)

If 2+ headings of a panorama are inside, its other headings are removed too when they
show almost no sky (< 5%) - e.g. the far end of the same tunnel - but not views out of it.

Writes, in data/<city>/:
    excluded_indoor_tunnel.txt   one filename per line (build_test_set.py reads it automatically)
    excluded_indoor_tunnel.csv   the same images with the reason and class shares
    review/flagged_XX.jpg        pictures of every flagged panorama, one row each (open them in VS Code)

If a removed view is actually fine, put its filename in data/<city>/keep.txt (one per line)
and re-run; it won't be excluded. The review pictures show only panoramas with at least one
removed view: red = removed (with the reason), black = kept.

Usage:
    python flag_indoor.py --city newyork
    python flag_indoor.py --city london --no-review     # skip the pictures
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import ADE20K, CITIES, load_city

WALL, SKY, FLOOR, CEILING = 0, 2, 3, 5
STREET = ["building", "house", "skyscraper", "tree", "grass", "plant", "sidewalk", "car", "sky",
          "truck", "bus", "van", "fence", "streetlight", "pole", "signboard", "traffic light"]


def inside_rule(X, panoid):
    """Returns (excluded bool array, reason array) - see the module docstring."""
    street = X[:, [ADE20K.index(n) for n in STREET]].sum(1)
    sky, ceil, floor_ = X[:, SKY], X[:, CEILING], X[:, FLOOR]
    tunnel = (ceil >= 0.15) & (sky < 0.03) & (street < 0.15)
    room = ((floor_ >= 0.10) | (ceil >= 0.10)) & (sky < 0.02) & (street < 0.15)
    blank = (sky < 0.02) & (street < 0.05)
    hit = tunnel | room | blank
    n_inside = pd.Series(hit).groupby(np.asarray(panoid)).transform("sum").values
    pano = ~hit & (n_inside >= 2) & (sky < 0.05)
    reason = np.where(tunnel, "tunnel", np.where(room, "room", np.where(blank, "blank",
                      np.where(pano, "same panorama", ""))))
    return hit | pano, reason, street


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=CITIES)
    ap.add_argument("--no-review", action="store_true", help="don't draw the review pictures")
    args = ap.parse_args()

    d = load_city(args.city)
    X, L, out = d["m2f"], d["locs"], d["out"]
    out.mkdir(parents=True, exist_ok=True)

    flagged, reason, street = inside_rule(X, L.panoid.values)
    hit = flagged
    panos = list(dict.fromkeys(L.panoid[flagged]))      # panoramas to show in the review, CSV order
    excl = flagged.copy()

    keep_file = out / "keep.txt"
    if keep_file.exists():
        keep = {l.strip() for l in open(keep_file) if l.strip()}
        unknown = keep - set(L.filename)
        if unknown:
            raise SystemExit(f"keep.txt has filenames not in {args.city}: {sorted(unknown)[:3]}")
        excl = excl & ~np.array(L.filename.isin(keep).values, dtype=bool)
        print(f"keep.txt: {len(keep)} images kept despite being flagged")

    with open(out / "excluded_indoor_tunnel.txt", "w") as f:
        f.write("\n".join(L.filename[excl]) + "\n")
    t = L[excl][["filename", "panoid"]].copy()
    t["reason"] = reason[excl]
    for name, j in [("ceiling", CEILING), ("floor", FLOOR), ("wall", WALL), ("sky", SKY)]:
        t[name] = X[excl, j].round(3)
    t["street"] = street[excl].round(3)
    t.to_csv(out / "excluded_indoor_tunnel.csv", index=False)
    counts = pd.Series(reason[excl]).value_counts().to_dict()
    print(f"{args.city}: {excl.sum()} images excluded ({excl.mean():.1%} of {len(L)}) from {len(panos)} panoramas: "
          + ", ".join(f"{v} {k}" for k, v in counts.items()))
    print(f"wrote {out / 'excluded_indoor_tunnel.txt'} and .csv")

    if args.no_review:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    review = out / "review"
    review.mkdir(exist_ok=True)
    for old in review.glob("flagged_*.jpg"):
        old.unlink()
    shown = panos
    per_page = 8
    for page in range(0, len(shown), per_page):
        chunk = shown[page:page + per_page]
        ncol = max(int(L.panoid.eq(p).sum()) for p in chunk)
        fig, axes = plt.subplots(len(chunk), ncol, figsize=(4 * ncol, 2.9 * len(chunk)), squeeze=False)
        for r, p in enumerate(chunk):
            rows = L.index[L.panoid == p]
            for c in range(ncol):
                ax = axes[r, c]
                ax.axis("off")
                if c >= len(rows):
                    continue
                i = rows[c]
                path = d["img_dir"] / L.filename[i]
                if path.exists():
                    ax.imshow(Image.open(path))
                else:
                    ax.text(0.5, 0.5, "missing", ha="center", color="red")
                if excl[i]:
                    status, color = f"REMOVED ({reason[i]})", "red"
                elif hit[i]:
                    status, color = "KEPT (keep.txt)", "green"
                else:
                    status, color = "kept", "black"
                ax.set_title(f"{L.filename[i][:28]}  {status}\nceil {X[i, CEILING]:.0%} floor {X[i, FLOOR]:.0%} "
                             f"wall {X[i, WALL]:.0%} sky {X[i, SKY]:.0%} street {street[i]:.0%}",
                             fontsize=7, color=color)
        fig.tight_layout()
        path = review / f"flagged_{page // per_page:02d}.jpg"
        fig.savefig(path, dpi=70)
        plt.close(fig)
    print(f"wrote {len(range(0, len(shown), per_page))} review pictures in {review} "
          f"(red = removed, black = kept)")


if __name__ == "__main__":
    main()
