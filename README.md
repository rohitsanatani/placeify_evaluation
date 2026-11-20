# PLACEIFY user study

Builds the test sets for the user study (20 test places per city, top-10 results per
model and query type, random baselines) and runs the rating app.

Cities: New York, London, Boston. Models: Mask2Former (`m2f`), SigLIP2 (`siglip`).
Query types: image, sketch, text (text is SigLIP only).

## Folder layout

```
placeify/
  .env            your paths and API key (copy from .env.example; never share or commit)
  common.py       where each city's data lives + the shared loader every script uses
  check_data.py   step 1 check
  flag_indoor.py, build_test_set.py, make_contact_sheets.py   step 2
  data/<city>/    pipeline outputs: test_set.json, review/, sheets/, sketches/, ...
  app.py, config.json, templates/   step 5: the rating app
  results/        ratings (created by the app)
```

The input data is not copied here. `.env` points to it:
- embeddings + locations CSV: `~/Documents/researchers/ashley/<city>/`
- images: `/data/Datasets/<city folder>/img`

## Step 1: setup (once)

On the server, over VS Code SSH:

```bash
conda create -n ashley-placeify python=3.11 -y
conda activate ashley-placeify
cd ~/Documents/researchers/ashley/placeify
pip install -r requirements.txt
cp .env.example .env        # then open .env and check the paths
python check_data.py
```

Every city should end with `OK`. Boston prints that it dropped 12 duplicate rows: one
panorama is listed 4 times in `boston_locs.csv`; the loader keeps the first copy.

Each new terminal: `conda activate ashley-placeify` before running anything.

## Step 2: build the test sets (per city)

```bash
python flag_indoor.py --city boston            # 2a: flag indoor/tunnels -> data/boston/review/*.jpg
python build_test_set.py --city boston         # 2b: 20 test images + image searches
python make_contact_sheets.py --city boston    # 2c: one review picture per place -> data/boston/sheets/
```

Only views fully inside a tunnel or building are removed (see the top of `flag_indoor.py`);
outdoor views under bridges, beside walls or at tunnel entrances stay.

Review in VS Code (click the .jpg files):
- `data/<city>/review/flagged_XX.jpg`: every panorama with a removed view (red = removed,
  with the reason; black = kept). If a removed view is actually fine, add its filename to
  `data/<city>/keep.txt` and re-run 2a and 2b.
- `data/<city>/sheets/qXX.jpg`: each test place with its results. To drop a bad image
  (glitched, blocked view), add its filename to `data/<city>/excluded_manual.txt` and re-run 2b.

**New York**: built fresh like the other cities. Sketches and prompts from the earlier
Colab run are reused for any test image that's still in the new set (matched by filename):

```bash
python import_colab.py --city newyork --study ~/Documents/researchers/ashley/study   # once
python flag_indoor.py --city newyork
python build_test_set.py --city newyork
python reuse_previous.py --city newyork      # lists which places still need a sketch + prompt
python make_contact_sheets.py --city newyork
```

The Colab version is kept in `data/newyork/previous/`.

### Blurry test images

Street View blurs out some regions (faces, houses on request). To check each city's test images:

```bash
python review_test_images.py --city boston     # -> data/boston/review/test_images_1.jpg ... (open + zoom in)
```

If a test image is blurry, add its filename to `data/<city>/blurry.txt` (one per line) and re-run
`build_test_set.py`. That place then uses the next-closest image to its centroid. Blurry images are
never test images and never shown to participants, but they stay in the clustering, so the clusters
(and the other test places) don't change. Random sets are redrawn. Then give the new test images a
prompt, sketch and embedding (only the changed places are generated):

```bash
python generate_text_prompts.py --city boston
python generate_sketches.py --city boston --model gpt-image-2.5-flare-2026-09-08
python embed_queries.py --city boston
python build_test_set.py --city boston
```

## Step 3: sketches and text prompts (per city)

Put the OpenAI key in `.env` (`OPENAI_API_KEY=sk-...`), then:

```bash
python generate_text_prompts.py --city boston    # -> data/boston/text_prompts.csv
python generate_sketches.py --city boston        # -> data/boston/sketches/qXX.png
python make_contact_sheets.py --city boston      # sheets now show sketch + prompt
```

Both only call the API for places that don't have a valid prompt/sketch yet (matched by the
test image's filename), so re-running is free and a changed test set only redoes changed
places. Redo specific places with `--only q03 q07`. The exact instructions used are saved in
`data/<city>/text_prompts_instruction.txt` and `data/<city>/sketches/instruction.txt`.

## Step 4: sketch and text searches (per city)

Install PyTorch with GPU support and transformers once (check the GPU with `nvidia-smi`):

```bash
pip install torch transformers
python -c "import torch; print(torch.cuda.is_available())"    # should print True
```

Then for each city:

```bash
python embed_queries.py --city boston        # CHECK must pass; -> data/boston/query_embeddings.npz
python build_test_set.py --city boston       # adds sketch_m2f, sketch_siglip, text_siglip searches
python make_contact_sheets.py --city boston
```

For New York, `build_test_set.py` picks the same test images as before (same data, same machine),
and stops with a clear message if the embeddings were made for different ones.

## Step 5: the rating app

`config.json` sets the ONE combination a run of the app serves (see its `_options` block):

| setting | values |
|---|---|
| `city` / `model` / `modality` | newyork, london, boston / m2f, siglip / image, sketch, text (text needs siglip) |
| `city: "random"` | each new participant gets one of the 3 cities: the one with the fewest sessions so far for this modality/model/view (ties at random), so the cities fill up evenly. All 3 cities must be built for the modality, or the app won't start. Results still go to each city's own folder |
| `comparison_view` | `panorama` = all 4 headings of the comparison place (0/90/180/270 order, the matched one marked with an arrow); `single` = just the matched image |
| `n_places` | reference images (test places) per participant, 1-20; least-used ones are handed out first |
| `n_per_set` | comparisons per reference: this many top results (ranks 1..n) AND this many random images, 1-10 (default 10) |
| `n_repeats` | consistency check: comparisons per reference shown a second time, >= 4 positions later (0 = off; at most 2 x n_per_set - 4) |
| `practice` | `true` = start with 4 practice comparisons (place p00: 2 top + 2 random) |
| `port`, `completion_url`, `completion_code` | server port; Prolific completion link / code |

```bash
pip install flask waitress
python app.py                          # -> http://<server>:5050/
python app.py --config other.json      # a second combination at the same time, on another port
```

Hovering over any image (reference, panorama views or the single comparison image) enlarges it in place (2x, after 0.2 s).

Restart after changing `config.json` or any `.py` file; template (.html) edits only need a browser refresh.
The app refuses to start if the combination is invalid or anything it needs isn't built yet.

Participant flow: instructions + consent + Prolific ID -> practice round -> "practice complete" ->
n_places references x (2 x n_per_set + n_repeats) comparisons, shuffled, 1-7 slider + optional comment -> completion code.

Results, in `results/<city>_<modality>_<model>_<view>/` (with `random`, the participant's assigned city; the session JSON also has `city_setting: "random"`):
- `sessions/<id>.json`: saved after every rating (a closed tab can be reopened and resumes)
- `ratings.csv`: one row per comparison, added when a participant finishes. Columns include block
  (practice/main), reference + comparison filenames, panorama IDs and heading, top/random + rank,
  is_repeat + repeat_of_position, `similarity` (the score this search ranked by), `sim_to_photo_m2f` /
  `sim_to_photo_siglip`, rating and comment
- `python export_results.py` -> `results/all_ratings.csv` with every combination, including unfinished sessions

**Practice place**: `build_test_set.py` adds p00 to every city's test set (the 20 real places are unchanged).
Before running the app for a city, give it a sketch, prompt and embedding (only p00 is new; ~2 API calls):

```bash
python build_test_set.py --city boston
python generate_text_prompts.py --city boston
python generate_sketches.py --city boston --model gpt-image-2.5-flare-2026-09-08
python embed_queries.py --city boston        # reuses the 20 places' stored embeddings, computes only p00
python build_test_set.py --city boston
```

## Data quirks handled by `common.load_city()`

| City | Quirk | Handling |
|---|---|---|
| London | `year` + `month` columns instead of `date` | builds `date` as YYYY-MM |
| Boston | panorama `5BpxIOXm1cPBvGKYu1wepw` listed 4 times (16 rows) | keeps first copy of each filename, drops the 12 repeats from CSV and embeddings, after checking they're identical |
