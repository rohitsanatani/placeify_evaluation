"""
Shared helpers: where each city's data lives, and one loader that returns a
clean, aligned dataset for any city.

Every script should load data through load_city() so all of them see exactly
the same rows in exactly the same order.

What load_city() fixes:
  - London's CSV has `year` + `month` instead of `date` -> builds `date` as YYYY-MM
  - Boston's CSV lists one panorama 4 times -> keeps the first copy of each
    filename and drops the repeats from the CSV *and* the embeddings
    (only after checking the repeated rows have identical embeddings)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

# ADE20K-150 class names in Mask2Former's label order (index = column of the m2f embeddings)
ADE20K = ["wall","building","sky","floor","tree","ceiling","road","bed","windowpane","grass","cabinet","sidewalk","person","earth","door","table","mountain","plant","curtain","chair","car","water","painting","sofa","shelf","house","sea","mirror","rug","field","armchair","seat","fence","desk","rock","wardrobe","lamp","bathtub","railing","cushion","base","box","column","signboard","chest of drawers","counter","sand","sink","skyscraper","fireplace","refrigerator","grandstand","path","stairs","runway","case","pool table","pillow","screen door","stairway","river","bridge","bookcase","blind","coffee table","toilet","flower","book","hill","bench","countertop","stove","palm","kitchen island","computer","swivel chair","boat","bar","arcade machine","hovel","bus","towel","light","truck","tower","chandelier","awning","streetlight","booth","television","airplane","dirt track","apparel","pole","land","bannister","escalator","ottoman","bottle","buffet","poster","stage","van","ship","fountain","conveyer belt","canopy","washer","plaything","swimming pool","stool","barrel","basket","waterfall","tent","bag","minibike","cradle","oven","ball","food","step","tank","trade name","microwave","pot","animal","bicycle","lake","dishwasher","screen","blanket","sculpture","hood","sconce","vase","traffic light","tray","ashcan","fan","pier","crt screen","plate","monitor","bulletin board","shower","radiator","glass","clock","flag"]

CITIES = ["newyork", "london", "boston"]
MODELS = ["m2f", "siglip"]
MODALITIES = ["image", "sketch", "text"]
# text can only be searched with SigLIP (Mask2Former has no text encoder)
VALID_COMBOS = [(m, mod) for mod in MODALITIES for m in MODELS if not (mod == "text" and m == "m2f")]


def _env(name):
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"{name} is not set. Copy .env.example to .env in {HERE} and fill it in.")
    return v


def check_city(city):
    if city not in CITIES:
        raise SystemExit(f"Unknown city '{city}'. Choose from: {', '.join(CITIES)}")


def paths(city):
    """All paths for one city."""
    check_city(city)
    emb = Path(_env("EMB_ROOT")) / city
    out = Path(os.environ.get("OUT_ROOT") or HERE / "data") / city
    return {
        "locs": emb / f"{city}_locs.csv",
        "m2f": emb / f"mask2former_{city}.npy",
        "siglip": emb / f"siglip_{city}.npy",
        "img_dir": Path(_env(f"IMG_DIR_{city.upper()}")),
        "out": out,
    }


def load_city(city, verbose=True):
    """
    Returns a dict with:
      locs   DataFrame (filename, panoid, lat, lon, date, heading), index 0..N-1
      m2f    (N, 150) Mask2Former class shares
      siglip (N, 768) SigLIP2 image embeddings
      img_dir, out   paths
      n_raw, dropped how many rows were in the files / removed as duplicates
    Row i of locs, m2f and siglip always describe the same image.
    """
    p = paths(city)
    locs = pd.read_csv(p["locs"])
    m2f = np.load(p["m2f"])
    siglip = np.load(p["siglip"])
    n_raw = len(locs)
    if not (len(m2f) == len(siglip) == n_raw):
        raise SystemExit(f"{city}: row counts differ - locs {n_raw}, m2f {len(m2f)}, siglip {len(siglip)}")

    # date column
    if "date" not in locs.columns:
        if {"year", "month"} <= set(locs.columns):
            locs["date"] = locs["year"].astype(int).astype(str) + "-" + locs["month"].astype(int).astype(str).str.zfill(2)
        else:
            locs["date"] = ""
    for col in ["filename", "panoid", "lat", "lon"]:
        if col not in locs.columns:
            raise SystemExit(f"{city}: {p['locs'].name} has no '{col}' column (columns: {list(locs.columns)})")

    # duplicate filenames: keep the first copy
    dup = locs.filename.duplicated(keep="first").values
    if dup.any():
        first = {f: i for i, f in reversed(list(enumerate(locs.filename)))}
        for i in np.where(dup)[0]:
            j = first[locs.filename.iloc[i]]
            if not (np.allclose(m2f[i], m2f[j]) and np.allclose(siglip[i], siglip[j])):
                raise SystemExit(f"{city}: row {i} repeats {locs.filename.iloc[i]} but its embeddings differ "
                                 f"from row {j} - ask Rohit before going on")
        n_panos = locs.panoid[dup].nunique()
        keep = ~dup
        locs, m2f, siglip = locs[keep].reset_index(drop=True), m2f[keep], siglip[keep]
        if verbose:
            print(f"{city}: dropped {dup.sum()} duplicate rows (identical repeats of {n_panos} panorama(s))")

    locs["heading"] = locs.filename.str.extract(r"_(\d+)\.\w+$")[0].astype(int)
    locs = locs[["filename", "panoid", "lat", "lon", "date", "heading"]]
    return {"locs": locs, "m2f": m2f, "siglip": siglip, "img_dir": p["img_dir"], "out": p["out"],
            "n_raw": n_raw, "dropped": int(dup.sum())}


def top_classes(vec, n=4):
    """'tree 27%, road 25%, ...' for a Mask2Former class-share vector."""
    idx = np.argsort(-vec)[:n]
    return ", ".join(f"{ADE20K[i]} {vec[i]:.0%}" for i in idx if vec[i] > 0.01)


def l2(a):
    a = np.asarray(a, dtype=np.float64)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def all_places(test_set):
    """The 20 test places plus the practice place p00 (if the test set has one)."""
    return test_set["queries"] + ([test_set["practice"]] if test_set.get("practice") else [])
