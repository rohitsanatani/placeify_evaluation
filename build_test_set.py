"""
Step 2b: build one city's test set.

 1. drop excluded images from the whole dataset:
      data/<city>/excluded_indoor_tunnel.txt   (from flag_indoor.py - required)
      data/<city>/excluded_manual.txt          (optional, your own list, one filename per line)
 2. k-means (k=20) on the L2-normalized Mask2Former embeddings of what is left
 3. the real image nearest each centroid -> 20 test images (q00..q19). Images listed in
      data/<city>/blurry.txt                   (optional, one filename per line)
    are skipped (the next-closest image to that centroid is used instead) and are never shown
    to participants, but they stay in the clustering, so the clusters themselves don't change.
    (or, with --fixed-queries, reuse the test images of an earlier test_set.json)
 4. remove the test images from the search pool
 5. for each test image: top-N (N=10) by cosine similarity with Mask2Former and with SigLIP,
    plus - if data/<city>/query_embeddings.npz exists (step 4) - top-N for its sketch
    (Mask2Former, SigLIP) and its text prompt (SigLIP). Other headings of the test image's
    own panorama are never returned.
 6. a separate random set of N per search. Within a query type (image/sketch/text) random sets
    never overlap that type's top-N or each other. Each type uses its own random generator,
    so adding sketch/text searches later never changes the image sets.

Writes data/<city>/test_set.json, test_set.csv, cluster_labels.npy

Usage:
    python build_test_set.py --city newyork
    python build_test_set.py --city newyork --fixed-queries data/newyork/colab_test_set.json
"""
import argparse
import csv
import json

import numpy as np
from sklearn.cluster import KMeans

from common import CITIES, l2, load_city, top_classes


def read_list(path):
    return {l.strip() for l in open(path) if l.strip()} if path.exists() else set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=CITIES)
    ap.add_argument("--fixed-queries", help="earlier test_set.json whose 20 test images (and q-numbers) to reuse")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = load_city(args.city)
    locs, M, out = d["locs"], d["m2f"], d["out"]
    N = len(locs)
    pano = locs.panoid.values
    emb = {"m2f": l2(M), "siglip": l2(d["siglip"])}
    fname_to_row = {f: i for i, f in enumerate(locs.filename)}

    # 1. exclusions
    auto = out / "excluded_indoor_tunnel.txt"
    if not auto.exists():
        raise SystemExit(f"{auto} not found - run: python flag_indoor.py --city {args.city}")
    excluded = np.zeros(N, bool)
    for path in [auto, out / "excluded_manual.txt"]:
        names = read_list(path)
        if not names:
            continue
        unknown = names - set(fname_to_row)
        if unknown:
            raise SystemExit(f"{path.name}: not in {args.city}: {sorted(unknown)[:3]}")
        excluded |= locs.filename.isin(names).values
        print(f"{path.name}: {len(names)} images")
    valid = np.where(~excluded)[0]
    print(f"{args.city}: {N} images, {excluded.sum()} excluded, {len(valid)} used")

    # 2. cluster (Mask2Former)
    km = KMeans(n_clusters=args.k, n_init=20, random_state=args.seed).fit(emb["m2f"][valid])
    labels = np.full(N, -1)
    labels[valid] = km.labels_
    sizes = np.bincount(km.labels_, minlength=args.k)

    # 3. test images (skipping blurry ones - see blurry.txt in the docstring)
    blurry = read_list(out / "blurry.txt")
    unknown = blurry - set(fname_to_row)
    if unknown:
        raise SystemExit(f"blurry.txt: not in {args.city}: {sorted(unknown)[:3]}")
    is_blurry = locs.filename.isin(blurry).values
    if blurry:
        print(f"blurry.txt: {len(blurry)} images (never test images, never shown)")
    replaced = []
    if args.fixed_queries:
        prev = json.load(open(args.fixed_queries))["queries"]
        files = [q["query"]["filename"] for q in prev]
        ids = [q["query_id"] for q in prev]
        missing = [f for f in files if f not in fname_to_row]
        if missing:
            raise SystemExit(f"--fixed-queries: not in {args.city}: {missing}")
        queries = np.array([fname_to_row[f] for f in files])
        bad = [f for f, q in zip(files, queries) if excluded[q] or is_blurry[q]]
        if bad:
            raise SystemExit(f"--fixed-queries: these test images are now excluded or blurry: {bad}")
        # how does this compare with what k-means picks here?
        fresh = set()
        for c in range(args.k):
            m = valid[km.labels_ == c]
            fresh.add(int(m[np.argmin(np.linalg.norm(emb['m2f'][m] - km.cluster_centers_[c], axis=1))]))
        print(f"--fixed-queries: reusing {len(queries)} test images from {args.fixed_queries} "
              f"({len(fresh & set(queries.tolist()))} of them are also what k-means picks on this machine)")
    else:
        qs = []
        for c in range(args.k):
            members = valid[km.labels_ == c]
            dist = np.linalg.norm(emb["m2f"][members] - km.cluster_centers_[c], axis=1)
            order = members[np.argsort(dist)]
            pick = next((int(i) for i in order if not is_blurry[i]), None)
            if pick is None:
                raise SystemExit(f"cluster {c}: every image is listed in blurry.txt")
            skipped = int(np.searchsorted(np.sort(dist), dist[members == pick][0]))  # 0 = closest
            if skipped:
                replaced.append((c, len(qs), skipped))
            qs.append(pick)
        queries = np.array(qs)
        ids = [f"q{i:02d}" for i in range(len(queries))]
        for c, qi, k in replaced:
            print(f"  q{qi:02d}: closest {k} image(s) to the centroid are in blurry.txt -> using "
                  f"{locs.filename.iloc[queries[qi]]} ({k + 1}{'nd' if k == 1 else 'rd' if k == 2 else 'th'} closest)")

    # 4. search pool (blurry images are never shown)
    pool = np.setdiff1d(valid[~is_blurry[valid]], queries)

    # optional sketch/text query embeddings (step 4), looked up by the query image's filename
    qvec = {}          # filename -> {"sketch_m2f": vec, "sketch_siglip": vec, "text_siglip": vec}
    QKEYS = ["sketch_m2f", "sketch_siglip", "text_siglip"]
    qe = out / "query_embeddings.npz"
    if qe.exists():
        Z = np.load(qe)
        for j, f in enumerate(str(x) for x in Z["query_files"]):
            qvec[f] = {k: l2(Z[k][j:j + 1])[0] for k in QKEYS}
        print(f"query_embeddings.npz found: adding searches {QKEYS}")
        missing = [qid for qid, q in zip(ids, queries) if locs.filename.iloc[q] not in qvec]
        if missing:
            print(f"note: {' '.join(missing)} have no sketch/text embedding yet (new test images), so they get "
                  f"image searches only for now. Run generate_text_prompts.py, generate_sketches.py and "
                  f"embed_queries.py --city {args.city}, then this script again.")

    def info(i):
        r = locs.iloc[i]
        return {"index": int(i), "filename": r.filename, "panoID": r.panoid, "heading": int(r.heading),
                "lat": float(r.lat), "lon": float(r.lon), "date": str(r.date)}

    def entry(i, q):
        e = info(i)
        for m, E in emb.items():
            e[f"sim_{m}"] = round(float(E[i] @ E[q]), 4)
        return e

    # 5 + 6
    # Random sets are drawn per condition (image / sketch / text) with their own random
    # generator, so adding sketch/text searches never changes the image sets.
    def make_record(qid, q, cand, rngs, n):
        fname = locs.filename.iloc[q]
        results = {}
        for m, E in emb.items():
            s_ = E[cand] @ E[q]
            results[f"image_{m}"] = [entry(i, q) for i in cand[np.argsort(-s_)[:n]]]
        for key, vec in qvec.get(fname, {}).items():
            m = key.split("_")[1]
            s_ = emb[m][cand] @ vec
            order = np.argsort(-s_)[:n]
            results[key] = [{**entry(i, q), "sim_query": round(float(v), 4)} for i, v in zip(cand[order], s_[order])]
        randoms = {}
        for cond, rng in rngs.items():
            names = [k for k in results if k.startswith(cond + "_")]
            used = {e["index"] for k in names for e in results[k]}
            for name in names:
                rand = rng.choice(np.setdiff1d(cand, list(used)), size=n, replace=False)
                randoms[name] = [entry(i, q) for i in rand]
                used |= set(rand.tolist())
                if name in QKEYS:
                    m = name.split("_")[1]
                    for e in randoms[name]:
                        e["sim_query"] = round(float(emb[m][e["index"]] @ qvec[fname][name]), 4)
        return {"query_id": qid, "cluster": int(labels[q]), "cluster_size": int(sizes[labels[q]]),
                "query_summary": top_classes(M[q]), "query": info(q), "results": results, "random": randoms}

    def csv_rows(rec):
        for setname, lst in list(rec["results"].items()) + [(f"random_{k}", v) for k, v in rec["random"].items()]:
            for rank, e in enumerate(lst, 1):
                yield [rec["query_id"], rec["query"]["filename"], setname, rank, e["filename"],
                       e["sim_m2f"], e["sim_siglip"], e.get("sim_query", "")]

    rngs = {"image": np.random.default_rng(args.seed),
            "sketch": np.random.default_rng(args.seed + 1),
            "text": np.random.default_rng(args.seed + 2)}
    records, rows = [], []
    for qid, q in zip(ids, queries):
        rec = make_record(qid, q, pool[pano[pool] != pano[q]], rngs, args.n)
        records.append(rec)
        rows.extend(csv_rows(rec))

    # 7. practice place (p00), shown before the real study. Chosen and searched AFTER the 20 test
    #    places so their results are unaffected. It comes from the largest cluster (a typical scene),
    #    is never a test image or from a test image's panorama, and preferably isn't shown anywhere in
    #    the real study. Its own results avoid the test panoramas, and also avoid images shown in the
    #    real study when enough others are left (small cities like Boston may need some overlap;
    #    each participant only sees 2-3 places, so overlap with other places hardly matters).
    shown = {e["index"] for r in records for d in (r["results"], r["random"]) for lst in d.values() for e in lst}
    test_panos = set(pano[queries])
    shown_panos = {pano[i] for i in shown}
    free = np.array([pano[i] not in test_panos for i in pool])
    unshown = np.array([pano[i] not in shown_panos for i in pool])
    big = int(np.argmax(sizes))
    in_big = labels[pool] == big
    members = pool[free & in_big & unshown] if (free & in_big & unshown).any() else pool[free & in_big]
    p = int(members[np.argmin(np.linalg.norm(emb["m2f"][members] - km.cluster_centers_[big], axis=1))])
    base = free & (pano[pool] != pano[p])
    p_cand = pool[base & unshown] if (base & unshown).sum() >= 300 else pool[base]
    p_rngs = {"image": np.random.default_rng(args.seed + 10),
              "sketch": np.random.default_rng(args.seed + 11),
              "text": np.random.default_rng(args.seed + 12)}
    practice = make_record("p00", p, p_cand, p_rngs, args.n)
    practice["practice"] = True
    p_shown = {e["index"] for d in (practice["results"], practice["random"]) for lst in d.values() for e in lst}
    overlap = len({pano[i] for i in p_shown} & shown_panos) + (pano[p] in shown_panos)
    rows.extend(csv_rows(practice))
    if qvec and locs.filename.iloc[p] not in qvec:
        print("note: the practice place p00 has no sketch/text embedding yet - run generate_sketches.py, "
              "generate_text_prompts.py and embed_queries.py, then this script again")

    meta = {
        "city": args.city, "n_images": N, "n_duplicates_dropped": d["dropped"], "n_excluded": int(excluded.sum()),
        "k": args.k, "n": args.n, "seed": args.seed,
        "test_images": f"reused from {args.fixed_queries}" if args.fixed_queries else "nearest image to each k-means centroid",
        "clustering": "k-means on L2-normalized Mask2Former embeddings of non-excluded images",
        "similarity": "cosine",
        "random": ("a separate set per search; within a query type (image/sketch/text) never overlaps that "
                   "type's top-N or other random sets; never the test image's own panorama"),
        "searches": list(records[0]["results"]),
        "practice": "p00: an image near the centre of the largest cluster, never a test image or from a test "
                    "image's panorama; searched after the 20 test places so their results are unaffected",
        "sim_fields": "sim_m2f / sim_siglip = similarity to the test PHOTO; sim_query = similarity to the "
                      "sketch or text query that was searched with (sketch_/text_ searches only)",
        "pool_size": int(len(pool)),
        "blurry_skipped": sorted(blurry),
    }
    with open(out / "test_set.json", "w") as f:
        json.dump({"meta": meta, "queries": records, "practice": practice}, f, indent=2)
    with open(out / "test_set.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "query_filename", "set", "rank", "candidate_filename", "sim_m2f", "sim_siglip", "sim_query"])
        w.writerows(rows)
    np.save(out / "cluster_labels.npy", labels)

    print(f"pool after removing test images: {len(pool)}; searches: {meta['searches']}")
    print(f"{'q':5}{'size':>6}{'top m2f':>10}{'top siglip':>12}{'random':>9}  test image content")
    for r in records:
        tm = np.mean([e["sim_m2f"] for e in r["results"]["image_m2f"]])
        ts = np.mean([e["sim_siglip"] for e in r["results"]["image_siglip"]])
        rm = np.mean([e["sim_m2f"] for e in r["random"]["image_m2f"]])
        print(f"{r['query_id']:5}{r['cluster_size']:>6}{tm:>10.3f}{ts:>12.3f}{rm:>9.3f}  {r['query_summary']}")
    print(f"practice place p00: {practice['query']['filename']} ({practice['query_summary']}); "
          f"searches: {list(practice['results'])}; "
          + ("shares no panorama with the real study" if not overlap else
             f"{overlap} of its panoramas also appear somewhere in the real study (small city)"))
    print(f"\nwrote {out}/test_set.json, test_set.csv, cluster_labels.npy")


if __name__ == "__main__":
    main()
