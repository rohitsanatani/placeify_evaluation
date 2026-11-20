"""
The rating app.

One run of the app serves ONE combination, set in config.json:
    city             newyork | london | boston | random
                     random = each new participant is given one of the 3 cities: whichever has the fewest
                     sessions so far (ties broken at random), so the cities fill up evenly
    model            m2f | siglip
    modality         image | sketch | text          (text only works with siglip)
    comparison_view  panorama | single   panorama = all 4 headings of the comparison place in
                                         0/90/180/270 order, the matched heading enlarged
    n_places         how many reference images (of the 20 test places) each participant rates, e.g. 3;
                     handed out least-used first, so all 20 get rated about equally often
    n_per_set        comparisons per reference: this many top results (ranks 1..n) AND this many random
                     images, 1-10 (default 10, i.e. 10 + 10). Lower it to rate more references per session.
    n_repeats        consistency check: how many comparisons per place are shown a second time
                     (at a random later position, never within 3 of the original)
    practice         true = start with a practice block (place p00: 2 top + 2 random comparisons)

Participants tick the consent box, enter their Prolific ID, do the practice block, then rate
n_places references x (2 x n_per_set comparisons + n_repeats repeats) on a 1-7 scale, with an optional comment.

Saving (in results/<city>_<modality>_<model>_<view>/, using the participant's assigned city when city is random):
    sessions/<session_id>.json   written after EVERY rating, so nothing is lost if a tab closes
    ratings.csv                  one row per comparison (practice rows included, flagged), appended
                                 when a participant finishes
Use export_results.py to also get rows from sessions that weren't finished.

Run:
    python app.py                       # uses config.json
    python app.py --config other.json   # e.g. a second combination on another port
"""
import argparse
import csv
import json
import os
import random
import threading
import uuid
from datetime import datetime, timezone

from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from common import CITIES, HERE, MODELS, MODALITIES, load_city, paths

DEFAULTS = {"comparison_view": "panorama", "n_places": 3, "n_per_set": 10, "n_repeats": 2, "practice": True,
            "port": 5050, "completion_url": "", "completion_code": ""}
MIN_REPEAT_GAP = 3          # a repeat is shown at least this many comparisons after its original
PRACTICE_PER_SET = 2        # practice: this many top + this many random comparisons

CSV_COLUMNS = ["prolific_id", "study_id", "prolific_session_id", "session_id", "city", "model", "modality",
               "comparison_view", "block", "place_id", "place_position", "query_file", "query_panoid", "query_text",
               "comparison_file", "comparison_panoid", "comparison_heading", "comparison_position", "set", "rank",
               "is_repeat", "repeat_of_position", "similarity", "sim_to_photo_m2f", "sim_to_photo_siglip",
               "rating", "comment", "time_on_screen_ms", "answered_at", "session_started_at", "session_finished_at"]
# block               = practice | main
# set, rank           = top (rank 1-10 in this search) or random (1-10)
# is_repeat           = True for the consistency-check copy; repeat_of_position = position of the original
# similarity          = the score this session's search ranked by: cosine similarity between the comparison
#                       image and the reference (photo for image sessions, sketch / text prompt otherwise),
#                       using this session's model
# sim_to_photo_m2f/_siglip = similarity between the comparison image and the reference PHOTO, under each model


def trial_row(sess, t):
    """One CSV row for one comparison (used by the app and by export_results.py)."""
    return [sess["prolific_id"], sess["study_id"], sess["prolific_session_id"], sess["session_id"],
            sess["city"], sess["model"], sess["modality"], sess.get("comparison_view", "single"),
            t.get("block", "main"), t["place_id"], t["place_position"], t["query_file"], t.get("query_panoid", ""),
            t.get("query_text", ""), t["comparison_file"], t.get("comparison_panoid", ""),
            t.get("comparison_heading", ""), t["comparison_position"], t["set"], t["rank"],
            t.get("is_repeat", False), t.get("repeat_of_position", ""), t.get("similarity", ""),
            t.get("sim_to_photo_m2f", ""), t.get("sim_to_photo_siglip", ""), t["rating"], t.get("comment", ""),
            t["time_ms"], t["answered_at"], sess["started_at"], sess.get("finished_at", "")]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def study_cities(cfg):
    """The cities this run serves: one, or all of them when city is "random"."""
    if cfg["city"] == "random":
        return list(CITIES)
    if cfg["city"] not in CITIES:
        raise SystemExit(f'config: city must be one of {CITIES} or "random"')
    return [cfg["city"]]


def load_study(cfg, city):
    """Validate the config and load everything a session in this city needs. Fails loudly at startup."""
    model, modality, view = cfg["model"], cfg["modality"], cfg["comparison_view"]
    if city not in CITIES or model not in MODELS or modality not in MODALITIES:
        raise SystemExit(f"config: city must be {CITIES} or random, model {MODELS}, modality {MODALITIES}")
    if modality == "text" and model != "siglip":
        raise SystemExit("config: text can only be searched with siglip")
    if view not in ("panorama", "single"):
        raise SystemExit('config: comparison_view must be "panorama" or "single"')
    n_per, n_rep = int(cfg["n_per_set"]), int(cfg["n_repeats"])
    if not 1 <= int(cfg["n_places"]) <= 20:
        raise SystemExit("config: n_places must be 1-20")
    if not 1 <= n_per <= 10:
        raise SystemExit("config: n_per_set must be 1-10 (that many top results AND that many random images)")
    max_rep = max_repeats(2 * n_per)
    if not 0 <= n_rep <= min(5, max_rep):
        raise SystemExit(f"config: with n_per_set = {n_per} ({2 * n_per} comparisons per reference) n_repeats can be "
                         f"at most {min(5, max_rep)}, since each repeat must come at least {MIN_REPEAT_GAP + 1} "
                         f"comparisons after its original")

    d = load_city(city, verbose=False)
    p = paths(city)
    ts = p["out"] / "test_set.json"
    if not ts.exists():
        raise SystemExit(f"{ts} not found - build the test set first (step 2)")
    J = json.load(open(ts))
    search = f"{modality}_{model}"
    if any(search not in q["results"] for q in J["queries"]):
        raise SystemExit(f"test set has no '{search}' search yet - run step 4 (embed_queries.py + "
                         f"build_test_set.py --city {city})")

    prompts = {}
    if modality == "text":
        with open(p["out"] / "text_prompts.csv") as f:
            prompts = {r["query_id"]: r["prompt"].strip() for r in csv.DictReader(f)}

    def item(e, set_name, rank):
        # image searches are ranked by similarity to the photo; sketch/text searches by similarity to the query
        sim = e["sim_query"] if modality != "image" else e[f"sim_{model}"]
        return {"file": e["filename"], "panoid": e["panoID"], "heading": e["heading"], "set": set_name,
                "rank": rank, "similarity": sim, "sim_to_photo_m2f": e["sim_m2f"], "sim_to_photo_siglip": e["sim_siglip"]}

    def place(q, n_top, n_rand):
        qid = q["query_id"]
        if search not in q["results"]:
            raise SystemExit(f"{qid} has no '{search}' search - for the practice place run generate_text_prompts.py, "
                             f"generate_sketches.py, embed_queries.py and build_test_set.py --city {city}")
        pl = {"place_id": qid, "query_file": q["query"]["filename"], "query_panoid": q["query"]["panoID"],
              "items": [item(e, "top", r) for r, e in enumerate(q["results"][search][:n_top], 1)]
                       + [item(e, "random", r) for r, e in enumerate(q["random"][search][:n_rand], 1)]}
        if modality == "text":
            if not prompts.get(qid):
                raise SystemExit(f"no text prompt for {qid} - run generate_text_prompts.py --city {city}")
            pl["query_text"] = prompts[qid]
        if modality == "sketch" and not (p["out"] / "sketches" / f"{qid}.png").exists():
            raise SystemExit(f"no sketch for {qid} - run generate_sketches.py --city {city}")
        return pl

    places = [place(q, n_per, n_per) for q in J["queries"]]
    practice = None
    if cfg["practice"]:
        if not J.get("practice"):
            raise SystemExit(f"test set has no practice place - re-run build_test_set.py --city {city}")
        k = min(PRACTICE_PER_SET, n_per)
        practice = place(J["practice"], k, k)

    # every heading of every panorama, in 0/90/180/270 order (only files that exist)
    by_pano = {}
    for f, pid, h in zip(d["locs"].filename, d["locs"].panoid, d["locs"].heading):
        by_pano.setdefault(pid, []).append((int(h), f))
    all_places = places + ([practice] if practice else [])
    panoramas = {}
    for pl in all_places:
        for it in pl["items"]:
            heads = sorted(by_pano.get(it["panoid"], [(it["heading"], it["file"])]))
            panoramas[it["panoid"]] = [(h, f) for h, f in heads if (d["img_dir"] / f).exists() or f == it["file"]]

    allowed = {it["file"] for pl in all_places for it in pl["items"]}
    if view == "panorama":
        allowed |= {f for heads in panoramas.values() for _, f in heads}
    if modality == "image":
        allowed |= {pl["query_file"] for pl in all_places}
    missing_files = [f for f in allowed if not (d["img_dir"] / f).exists()]
    if missing_files:
        raise SystemExit(f"{len(missing_files)} images not found in {d['img_dir']}, e.g. {missing_files[:3]}")
    return {"city": city, "model": model, "modality": modality, "view": view, "places": places,
            "practice": practice, "panoramas": panoramas, "allowed": allowed, "img_dir": d["img_dir"],
            "sketch_dir": p["out"] / "sketches", "n_places": int(cfg["n_places"]), "n_per_set": n_per, "n_repeats": n_rep,
            "results": HERE / "results" / f"{city}_{modality}_{model}_{view}"}


def max_repeats(n_items):
    """How many repeats fit in a sequence of n_items with the minimum gap rule."""
    return max(0, n_items - MIN_REPEAT_GAP - 1)


def order_with_repeats(items, n_repeats, rng):
    """Shuffle a place's comparisons and insert n_repeats copies, each >= MIN_REPEAT_GAP after its original."""
    seq = [dict(it, is_repeat=False) for it in items]
    rng.shuffle(seq)
    # only items early enough that a copy fits at least MIN_REPEAT_GAP later
    eligible = list(range(len(seq) - MIN_REPEAT_GAP - 1))
    originals = [seq[k] for k in sorted(rng.sample(eligible, min(n_repeats, len(eligible))))]  # before any insert
    for orig in originals:
        cur = next(i for i, s in enumerate(seq) if s is orig)
        pos = rng.randint(cur + MIN_REPEAT_GAP + 1, len(seq))      # insert before index pos
        seq.insert(pos, dict(orig, is_repeat=True, _orig=id(orig)))
    # number the positions and link each repeat to its original's position
    pos_of = {}
    for i, s in enumerate(seq, 1):
        if not s["is_repeat"]:
            pos_of[id(s)] = i
    out = []
    for i, s in enumerate(seq, 1):
        t = {k: v for k, v in s.items() if k != "_orig"}
        t["comparison_position"] = i
        t["repeat_of_position"] = pos_of[s["_orig"]] if s["is_repeat"] else ""
        out.append(t)
    return out


def create_app(cfg):
    cfg = {**DEFAULTS, **cfg}
    studies = {c: load_study(cfg, c) for c in study_cities(cfg)}     # every city must be fully built
    for St in studies.values():
        (St["results"] / "sessions").mkdir(parents=True, exist_ok=True)
    S0 = next(iter(studies.values()))      # settings that are the same for every city (modality, counts, ...)
    lock = threading.Lock()
    app = Flask(__name__, static_folder=None)

    def find_session(sid):
        """(study of the session's city, path of its JSON file)."""
        try:
            uuid.UUID(sid)
        except (ValueError, TypeError):
            abort(400)
        for St in studies.values():
            f = St["results"] / "sessions" / f"{sid}.json"
            if f.exists():
                return St, f
        abort(404)

    def session_path(sid):
        return find_session(sid)[1]

    def n_sessions(St):
        return sum(1 for _ in (St["results"] / "sessions").glob("*.json"))

    def write_json(path, data):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        os.replace(tmp, path)          # atomic: a crash never leaves a half-written file

    def place_usage(S):
        """How many sessions in this city have been given each place so far (finished or not)."""
        used = {pl["place_id"]: 0 for pl in S["places"]}
        for f in (S["results"] / "sessions").glob("*.json"):
            try:
                for pid in json.loads(f.read_text()).get("assigned_places", []):
                    if pid in used:
                        used[pid] += 1
            except (OSError, ValueError):
                pass
        return used

    def trial(pl, block, place_position, n_places, t, n_in_place):
        tr = {"block": block, "place_id": pl["place_id"], "place_position": place_position, "n_places": n_places,
              "query_file": pl["query_file"], "query_panoid": pl["query_panoid"],
              "comparison_file": t["file"], "comparison_panoid": t["panoid"], "comparison_heading": t["heading"],
              "comparison_position": t["comparison_position"], "n_in_place": n_in_place, "set": t["set"],
              "rank": t["rank"], "is_repeat": t["is_repeat"], "repeat_of_position": t["repeat_of_position"],
              "similarity": t["similarity"], "sim_to_photo_m2f": t["sim_to_photo_m2f"],
              "sim_to_photo_siglip": t["sim_to_photo_siglip"],
              "rating": None, "comment": "", "time_ms": 0, "answered_at": ""}
        if "query_text" in pl:
            tr["query_text"] = pl["query_text"]
        return tr

    @app.get("/")
    def index():
        return send_from_directory(HERE / "templates", "instructions.html")

    @app.get("/study")
    def study():
        return send_from_directory(HERE / "templates", "study.html")

    @app.get("/api/info")
    def info():
        S = S0
        per = 2 * S["n_per_set"] + S["n_repeats"]
        n_main = S["n_places"] * per
        n_prac = len(S["practice"]["items"]) if S["practice"] else 0
        return jsonify(modality=S["modality"], view=S["view"], n_places=S["n_places"],
                       per_place=per, n_ratings=n_main, n_practice=n_prac)

    @app.post("/api/start")
    def start():
        body = request.get_json(force=True)
        pid = (body.get("prolific_id") or "").strip()
        if not pid or len(pid) > 64:
            return jsonify(error="Please enter your Prolific ID."), 400
        if body.get("consent") is not True:
            return jsonify(error="Please tick the consent box to continue."), 400
        rng = random.Random()
        trials = []
        with lock:   # so two people starting at once don't both get the same least-used city / places
            # city: the only one, or (random) the one with the fewest sessions so far, ties broken at random
            counts = {c: n_sessions(St) for c, St in studies.items()}
            fewest = min(counts.values())
            S = studies[rng.choice([c for c, n in counts.items() if n == fewest])]
            if S["practice"]:
                pr = S["practice"]
                items = [dict(it, is_repeat=False, repeat_of_position="") for it in pr["items"]]
                rng.shuffle(items)
                for i, it in enumerate(items, 1):
                    it["comparison_position"] = i
                    trials.append(trial(pr, "practice", "practice", 0, it, len(items)))
            usage = place_usage(S)
            ids = list(usage)
            rng.shuffle(ids)                       # random tie-break
            ids.sort(key=lambda x: usage[x])       # least-used first
            chosen = ids[:S["n_places"]]
            rng.shuffle(chosen)
            by_id = {pl["place_id"]: pl for pl in S["places"]}
            for pi, place_id in enumerate(chosen, 1):
                pl = by_id[place_id]
                seq = order_with_repeats(pl["items"], S["n_repeats"], rng)
                for t in seq:
                    trials.append(trial(pl, "main", pi, len(chosen), t, len(seq)))
            sid = str(uuid.uuid4())
            sess = {"session_id": sid, "prolific_id": pid, "study_id": (body.get("study_id") or "")[:64],
                    "prolific_session_id": (body.get("session_id") or "")[:64], "city": S["city"],
                    "model": S["model"], "modality": S["modality"], "comparison_view": S["view"],
                    "city_setting": cfg["city"],
                    "n_per_set": S["n_per_set"], "n_repeats": S["n_repeats"], "assigned_places": chosen,
                    "started_at": now(),
                    "consented_at": now(), "user_agent": request.headers.get("User-Agent", "")[:300],
                    "trials": trials}
            write_json(S["results"] / "sessions" / f"{sid}.json", sess)
        return jsonify(session_id=sid)

    @app.get("/api/session/<sid>")
    def get_session(sid):
        S, path = find_session(sid)
        sess = json.loads(path.read_text())
        c = S["city"]
        # only what the page needs to show: set, rank, similarity and repeat flags stay on the server
        out = []
        for t in sess["trials"]:
            if S["view"] == "panorama":
                comp = [{"src": f"/img/{c}/{f}", "heading": h, "main": f == t["comparison_file"]}
                        for h, f in S["panoramas"][t["comparison_panoid"]]]
            else:
                comp = [{"src": f"/img/{c}/{t['comparison_file']}", "heading": t["comparison_heading"], "main": True}]
            out.append({"block": t["block"], "place_position": t["place_position"], "n_places": t["n_places"],
                        "comparison_position": t["comparison_position"], "n_in_place": t["n_in_place"],
                        "query": (f"/sketch/{c}/{t['place_id']}" if S["modality"] == "sketch" else
                                  f"/img/{c}/{t['query_file']}" if S["modality"] == "image" else None),
                        "query_text": t.get("query_text"), "comparison": comp,
                        "rating": t["rating"], "comment": t.get("comment", "")})
        first_open = next((i for i, t in enumerate(sess["trials"]) if t["rating"] is None), len(out))
        return jsonify(modality=S["modality"], view=S["view"], trials=out, resume_at=first_open,
                       finished=bool(sess.get("finished_at")))

    @app.post("/api/rate/<sid>")
    def rate(sid):
        body = request.get_json(force=True)
        try:
            i, rating, ms = int(body["index"]), int(body["rating"]), int(body.get("time_ms", 0))
        except (KeyError, TypeError, ValueError):
            abort(400)
        comment = str(body.get("comment") or "")[:1000].strip()
        if not 1 <= rating <= 7:
            abort(400)
        with lock:
            path = session_path(sid)
            sess = json.loads(path.read_text())
            if sess.get("finished_at") or not 0 <= i < len(sess["trials"]):
                abort(400)
            t = sess["trials"][i]
            t["rating"], t["comment"] = rating, comment
            t["time_ms"], t["answered_at"] = t["time_ms"] + max(0, ms), now()
            write_json(path, sess)
        return jsonify(ok=True)

    @app.post("/api/finish/<sid>")
    def finish(sid):
        with lock:
            S, path = find_session(sid)
            sess = json.loads(path.read_text())
            left = sum(t["rating"] is None for t in sess["trials"])
            if left:
                return jsonify(error=f"{left} comparisons not rated yet"), 400
            if not sess.get("finished_at"):
                sess["finished_at"] = now()
                write_json(path, sess)
                out = S["results"] / "ratings.csv"
                new = not out.exists()
                with open(out, "a", newline="") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(CSV_COLUMNS)
                    w.writerows(trial_row(sess, t) for t in sess["trials"])
        return jsonify(completion_url=cfg.get("completion_url", ""), completion_code=cfg.get("completion_code", ""))

    @app.get("/img/<city>/<path:fname>")
    def img(city, fname):
        S = studies.get(city)
        if not S or fname not in S["allowed"]:
            abort(404)
        return send_file(S["img_dir"] / fname, max_age=86400)

    @app.get("/sketch/<city>/<qid>")
    def sketch(city, qid):
        S = studies.get(city)
        if not S:
            abort(404)
        ok = {pl["place_id"] for pl in S["places"]} | ({S["practice"]["place_id"]} if S["practice"] else set())
        if S["modality"] != "sketch" or qid not in ok:
            abort(404)
        return send_file(S["sketch_dir"] / f"{qid}.png", max_age=86400)

    app.config["STUDIES"] = studies
    app.config["STUDY"] = S0
    app.config["CFG"] = cfg
    return app


# Ports that Chrome/Firefox refuse to open (they'd show ERR_UNSAFE_PORT to participants)
BROWSER_BLOCKED_PORTS = {1719, 1720, 1723, 2049, 3659, 4045, 4190, 5060, 5061, 6000, 6566, 6665, 6666, 6667,
                         6668, 6669, 6679, 6697, 10080}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    cfg = {**DEFAULTS, **json.load(open(args.config))}
    port = int(cfg["port"])
    if port in BROWSER_BLOCKED_PORTS or port < 1024:
        raise SystemExit(f"port {port} can't be used: browsers block it (or it needs admin rights). "
                         f"Pick another, e.g. 5050-5059.")
    app = create_app(cfg)
    S = app.config["STUDY"]
    studies = app.config["STUDIES"]
    per = 2 * S["n_per_set"] + S["n_repeats"]
    city_txt = S["city"] if len(studies) == 1 else "random city (" + ", ".join(studies) + ")"
    print(f"Study: {city_txt} / {S['modality']} / {S['model']} / {S['view']} view - "
          f"{S['n_places']} of {len(S['places'])} references per participant x {per} comparisons "
          f"({S['n_per_set']} top + {S['n_per_set']} random + {S['n_repeats']} repeats) = {S['n_places'] * per} ratings"
          + (f" + {len(S['practice']['items'])} practice" if S["practice"] else ""))
    for St in studies.values():
        print(f"Results: {St['results']}")
    print(f"Open: http://localhost:{port}/  (or the server's address)")
    try:
        from waitress import serve
        serve(app, host=args.host, port=port, threads=8)
    except ImportError:
        app.run(host=args.host, port=port)


if __name__ == "__main__":
    main()
