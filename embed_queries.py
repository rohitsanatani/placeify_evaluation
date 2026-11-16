"""
Step 4a: turn one city's sketches and text prompts into embeddings, so they can be searched.

  sketch -> Mask2Former (same steps as Rohit's notebook: facebook/mask2former-swin-large-ade-semantic,
            semantic segmentation, share of pixels per ADE20K class = 150 numbers)
  sketch -> SigLIP2 image encoder (google/siglip2-base-patch16-224, 768 numbers)
  text   -> SigLIP2 text encoder  (same model, 768 numbers)

First it runs a CHECK: it re-embeds a few of the city's photos and compares them with the
saved mask2former_<city>.npy / siglip_<city>.npy rows. If the model or preprocessing doesn't
reproduce how the dataset was embedded, it stops and saves nothing.

Reads:   data/<city>/test_set.json, data/<city>/sketches/qXX.png, data/<city>/text_prompts.csv
Writes:  data/<city>/query_embeddings.npz   (then re-run build_test_set.py to add the searches)

Usage:
    python embed_queries.py --city boston
    python build_test_set.py --city boston
"""
import argparse
import csv
import hashlib
import json
import sys

import numpy as np
from PIL import Image

from common import ADE20K, CITIES, l2, load_city, all_places

M2F_ID = "facebook/mask2former-swin-large-ade-semantic"
SIGLIP_ID = "google/siglip2-base-patch16-224"


def as_tensor(x):
    import torch
    # get_*_features returns a tensor in most transformers versions, an output object in some
    return x if torch.is_tensor(x) else getattr(x, "pooler_output", x[0])


class M2F:
    def __init__(self, device):
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(M2F_ID)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(M2F_ID).to(device).eval()

    def embed(self, images, batch=8):
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(images), batch):
                imgs = images[i:i + batch]
                px = self.proc(images=imgs, return_tensors="pt")["pixel_values"].to(self.device)
                res = self.model(px)
                segs = self.proc.post_process_semantic_segmentation(res, target_sizes=[im.size[::-1] for im in imgs])
                for seg in segs:
                    seg = seg.cpu().numpy()
                    out.append(np.bincount(seg.ravel(), minlength=len(ADE20K)) / seg.size)
        return np.vstack(out)


class SigLIP:
    def __init__(self, device):
        from transformers import AutoModel, AutoProcessor
        self.device = device
        self.proc = AutoProcessor.from_pretrained(SIGLIP_ID)
        self.model = AutoModel.from_pretrained(SIGLIP_ID).to(device).eval()

    def embed_images(self, images, batch=16):
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(images), batch):
                inp = self.proc(images=images[i:i + batch], return_tensors="pt").to(self.device)
                out.append(as_tensor(self.model.get_image_features(**inp)).float().cpu().numpy())
        return np.vstack(out)

    def embed_texts(self, texts):
        import torch
        # SigLIP2 was trained with text padded to 64 tokens; the model card says to do the same
        with torch.no_grad():
            inp = self.proc(text=texts, padding="max_length", max_length=64, truncation=True,
                            return_tensors="pt").to(self.device)
            return as_tensor(self.model.get_text_features(**inp)).float().cpu().numpy()


def rgb(path):
    return Image.open(path).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=CITIES)
    ap.add_argument("--n-check", type=int, default=12)
    ap.add_argument("--skip-check", action="store_true", help="only if you know why the check fails")
    ap.add_argument("--recompute-all", action="store_true", help="re-embed every place instead of reusing stored ones")
    args = ap.parse_args()

    import torch
    d = load_city(args.city, verbose=False)
    out, img_dir, locs = d["out"], d["img_dir"], d["locs"]
    queries = all_places(json.load(open(out / "test_set.json")))   # 20 test places + practice p00
    qids = [q["query_id"] for q in queries]
    files = [q["query"]["filename"] for q in queries]

    # inputs present and made for these test images?
    sk_log = {r["query_id"]: r["filename"] for r in csv.DictReader(open(out / "sketches" / "sketches.csv"))} \
        if (out / "sketches" / "sketches.csv").exists() else {}
    prompts = {r["query_id"]: r for r in csv.DictReader(open(out / "text_prompts.csv"))} \
        if (out / "text_prompts.csv").exists() else {}
    bad_sk = [q for q, f in zip(qids, files) if not (out / "sketches" / f"{q}.png").exists() or sk_log.get(q) != f]
    bad_tx = [q for q, f in zip(qids, files)
              if not prompts.get(q, {}).get("prompt", "").strip() or prompts[q].get("filename") != f]
    if bad_sk or bad_tx:
        sys.exit(f"Not ready: sketches missing/out of date for {bad_sk}, prompts for {bad_tx}.\n"
                 f"Run generate_sketches.py / generate_text_prompts.py --city {args.city} first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{args.city}: device {device}; loading models (first time downloads ~1 GB)...")
    m2f, sig = M2F(device), SigLIP(device)

    # ---- check: re-embed dataset photos and compare with the saved embeddings ----
    if not args.skip_check:
        rows = np.random.default_rng(0).choice(len(locs), size=args.n_check, replace=False)
        photos = [rgb(img_dir / f) for f in locs.filename.iloc[rows]]
        cos_m = (l2(m2f.embed(photos)) * l2(d["m2f"][rows])).sum(1)
        cos_s = (l2(sig.embed_images(photos)) * l2(d["siglip"][rows])).sum(1)
        print(f"\nCHECK on {args.n_check} {args.city} photos (1.000 = identical to the saved embeddings)")
        print(f"  Mask2Former: mean {cos_m.mean():.4f}, worst {cos_m.min():.4f}")
        print(f"  SigLIP2    : mean {cos_s.mean():.4f}, worst {cos_s.min():.4f}")
        # small differences are normal (GPU vs CPU, library versions); a different model scores far lower
        if cos_m.mean() <= 0.97 or cos_s.mean() <= 0.97:
            sys.exit("\nCHECK FAILED: " + ("Mask2Former " if cos_m.mean() <= 0.97 else "")
                     + ("SigLIP " if cos_s.mean() <= 0.97 else "")
                     + "does not reproduce the saved embeddings, so sketch/text embeddings would not be "
                       "comparable. Ask Rohit how the dataset embeddings were made. Nothing saved.")
        print("  -> both match; sketch/text embeddings will be in the same space as the dataset.\n")

    texts = [prompts[q]["prompt"].strip() for q in qids]
    sk_md5 = [hashlib.md5((out / "sketches" / f"{q}.png").read_bytes()).hexdigest() for q in qids]

    # Reuse stored embeddings for places whose test image, prompt and sketch haven't changed, so adding
    # the practice place (or redoing one sketch) never shifts the other places' results.
    old = {}
    if (out / "query_embeddings.npz").exists() and not args.recompute_all:
        Z = np.load(out / "query_embeddings.npz")
        has_md5 = "sketch_md5" in Z.files
        for j, f in enumerate(str(x) for x in Z["query_files"]):
            old[f] = {"text": str(Z["texts"][j]), "md5": str(Z["sketch_md5"][j]) if has_md5 else None,
                      "sketch_m2f": Z["sketch_m2f"][j], "sketch_siglip": Z["sketch_siglip"][j],
                      "text_siglip": Z["text_siglip"][j]}
    reuse = [f in old and old[f]["text"] == t and old[f]["md5"] in (None, m)   # None = file from before hashes were stored
             for f, t, m in zip(files, texts, sk_md5)]
    todo = [i for i, r in enumerate(reuse) if not r]
    print(f"reusing stored embeddings for {sum(reuse)} places; computing {len(todo)}: {' '.join(qids[i] for i in todo)}")

    sketch_m2f = np.zeros((len(qids), len(ADE20K)))
    sketch_siglip = np.zeros((len(qids), 768), dtype=np.float32)
    text_siglip = np.zeros((len(qids), 768), dtype=np.float32)
    for i, r in enumerate(reuse):
        if r:
            o = old[files[i]]
            sketch_m2f[i], sketch_siglip[i], text_siglip[i] = o["sketch_m2f"], o["sketch_siglip"], o["text_siglip"]
    if todo:
        sketches = [rgb(out / "sketches" / f"{qids[i]}.png") for i in todo]
        sketch_m2f[todo] = m2f.embed(sketches)
        sketch_siglip[todo] = sig.embed_images(sketches)
        text_siglip[todo] = sig.embed_texts([texts[i] for i in todo])

    np.savez(out / "query_embeddings.npz", query_ids=np.array(qids), query_files=np.array(files),
             sketch_m2f=sketch_m2f, sketch_siglip=sketch_siglip, text_siglip=text_siglip, texts=np.array(texts),
             sketch_md5=np.array(sk_md5), models=np.array([M2F_ID, SIGLIP_ID]))
    print(f"saved {out / 'query_embeddings.npz'}: sketch_m2f {sketch_m2f.shape}, "
          f"sketch_siglip {sketch_siglip.shape}, text_siglip {text_siglip.shape}")

    print("\nWhat Mask2Former sees in each sketch vs its photo. A sketch that's mostly 'wall' or 'sky'")
    print("means segmentation struggled on the line drawing:")
    for q, v, rec in zip(qids, sketch_m2f, queries):
        top = np.argsort(-v)[:4]
        print(f"  {q} sketch: " + ", ".join(f"{ADE20K[i]} {v[i]:.0%}" for i in top))
        print(f"  {q} photo : {rec['query_summary']}")
    print(f"\nNext: python build_test_set.py --city {args.city}")


if __name__ == "__main__":
    main()
