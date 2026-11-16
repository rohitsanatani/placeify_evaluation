"""
Step 1 check: confirm every city's data is found and lines up.

For each city it loads the data through common.load_city() and checks that
  - the locations CSV, Mask2Former and SigLIP embeddings have the same rows
  - every filename in the CSV exists in the image folder
  - the image folder has no images missing from the CSV
  - each panorama has 4 headings (0/90/180/270)

Usage:
    python check_data.py                 # all cities
    python check_data.py --city boston   # one city
"""
import argparse
import os

from common import CITIES, load_city


def check(city):
    d = load_city(city)
    locs = d["locs"]
    problems, notes = [], []
    if not d["img_dir"].is_dir():
        problems.append(f"image folder not found: {d['img_dir']}")
        files = set()
    else:
        files = {f for f in os.listdir(d["img_dir"]) if not f.startswith(".")}
    if files:
        missing = set(locs.filename) - files
        extra = files - set(locs.filename)
        if missing:
            problems.append(f"{len(missing)} CSV filenames not in the image folder, e.g. {sorted(missing)[:3]}")
        if extra:
            notes.append(f"{len(extra)} images in the folder are not in the CSV (ignored), e.g. {sorted(extra)[:3]}")
    per_pano = locs.groupby("panoid").heading.apply(lambda h: tuple(sorted(h)))
    odd = per_pano[per_pano != (0, 90, 180, 270)]
    if len(odd):
        notes.append(f"{len(odd)} panoramas don't have exactly headings 0/90/180/270")
    print(f"\n== {city}")
    print(f"   rows: {d['n_raw']} in files, {d['dropped']} duplicates dropped -> {len(locs)} images, "
          f"{locs.panoid.nunique()} panoramas")
    print(f"   embeddings: m2f {d['m2f'].shape}, siglip {d['siglip'].shape}")
    print(f"   image folder: {d['img_dir']} ({len(files)} files)")
    print(f"   dates: {locs.date.min()} to {locs.date.max()}")
    for n in notes:
        print(f"   note: {n}")
    for p in problems:
        print(f"   PROBLEM: {p}")
    print("   OK" if not problems else "   NOT OK")
    return not problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=CITIES)
    args = ap.parse_args()
    results = {c: check(c) for c in ([args.city] if args.city else CITIES)}
    print("\nSummary: " + ", ".join(f"{c} {'OK' if ok else 'NOT OK'}" for c, ok in results.items()))


if __name__ == "__main__":
    main()
