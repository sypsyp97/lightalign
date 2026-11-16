"""CheXpert Plus subset from Redivis (Stanford AIMI, research-use agreement) for the chest X-ray payoff: the report
table, the CheXbert label files, and the frontal PNGs of N random train studies plus every valid study, resized to a
longest side of 1,024 px (8-bit PNG). A larger --n grows an existing subset.json: its studies stay in, new ones are drawn
from the rest. Token: file named by $REDIVIS_TOKEN_FILE. Stdlib only (urllib); PIL from the venv.

    python -m align.chexpert meta   --out DIR
    python -m align.chexpert images --out DIR --n 6000 [--workers 16]

Scale-up (every train-split study beyond the 30,000 of subset.json, cut into chunks that download as array tasks).
--out stays the read-only source directory; every write goes to --dest.

    python -m align.chexpert scale-plan   --out DIR --dest DEST --chunks 4
    python -m align.chexpert scale-images --out DIR --dest DEST --emb EMB --chunk K [--workers 16]
                                          -> DEST/png/ (resumable), DEST/undecodable/, EMB/excluded_pngs.txt
"""
import argparse
import csv
import http.client
import io
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://redivis.com/api/v1"
DS = "aimi.chexpert_plus:5yyj:v1_0"
TABLES = {"reports": "df_chexpert_plus_240401:bavj", "png_train": "png_train:s6cj", "png_valid": "png_valid:41v9",
          "labels": "chexpert_labels:pmec"}
TOKEN = open(os.environ["REDIVIS_TOKEN_FILE"]).read().strip()


def _transient(e):
    if isinstance(e, urllib.error.HTTPError):
        return e.code == 429 or e.code >= 500
    return isinstance(e, (http.client.RemoteDisconnected, http.client.IncompleteRead, ConnectionError, TimeoutError,
                          urllib.error.URLError))


def fetch(url, path=None, attempts=8):
    """GET with the token; the token is not forwarded on the redirect to the signed storage URL. Transient network
    errors and HTTP 429/5xx are retried with exponential backoff (2 s doubling, capped at 120 s, plus jitter)."""
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(url)
        req.add_unredirected_header("Authorization", f"Bearer {TOKEN}")
        tmp = path + ".tmp" if path else None
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                if path is None:
                    return r.read()
                with open(tmp, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                os.replace(tmp, path)
                return
        except Exception as e:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
            if not _transient(e) or attempt == attempts:
                raise
            delay = min(2 ** attempt, 120) + random.uniform(0, 1)
            print(f"fetch retry {attempt}/{attempts - 1} in {delay:.0f}s: ...{url[-60:]}: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            time.sleep(delay)


def table_csv(name, path):
    if not os.path.exists(path):
        fetch(f"{API}/tables/{DS}.{TABLES[name]}/rows?format=csv", path)
    return list(csv.DictReader(open(path, newline="")))


def meta(a):
    os.makedirs(a.out, exist_ok=True)
    for name in TABLES:
        rows = table_csv(name, f"{a.out}/{name}.csv")
        print(f"{name}: {len(rows)} rows", flush=True)
    for r in table_csv("labels", f"{a.out}/labels.csv"):         # CheXbert labels of the report text
        p = f"{a.out}/{r['file_name']}"
        if not os.path.exists(p):
            fetch(f"{API}/rawFiles/{r['file_id']}", p)
        print(r["file_name"], os.path.getsize(p), flush=True)


def key_of(p):
    """train/patient1/study1/view1_frontal.jpg -> index name patient1/study1/view1_frontal.png"""
    return p.split("/", 1)[1].rsplit(".", 1)[0] + ".png"


def frontal_studies(out):
    """The first frontal view of every study with a PNG in the Redivis index: (train, valid) sorted lists of
    path_to_image, and the index {file_name: row}."""
    rep = table_csv("reports", f"{out}/reports.csv")
    index = {r["file_name"]: r for n in ("png_train", "png_valid") for r in table_csv(n, f"{out}/{n}.csv")}
    frontal = [r for r in rep if r["frontal_lateral"] == "Frontal" and key_of(r["path_to_image"]) in index]
    first = {}
    for r in frontal:                                               # one frontal image per study, the first view
        first.setdefault(r["path_to_image"].rsplit("/", 1)[0], r)
    train = sorted(v["path_to_image"] for k, v in first.items() if v["split"] == "train")
    valid = sorted(v["path_to_image"] for k, v in first.items() if v["split"] == "valid")
    return train, valid, index


def png_ok(path):
    """The file decodes completely as an 8-bit grayscale image with a longest side of at most 1,024 px."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            im.load()
            return im.mode == "L" and max(im.size) <= 1024
    except (OSError, ValueError, SyntaxError):
        return False


def save_png(index, p, out, verify=False, attempts=4, undecodable=None):
    """Download one study's image, resize to a longest side of 1,024 px (8-bit), write out atomically; 0 if present
    (verify: only if it passes png_ok, else it is downloaded again). A body that does not decode (a corrupt transfer
    with HTTP 200) is downloaded again, up to `attempts` times; after that the study key is raised, or, with an
    `undecodable` directory, recorded there as a file named like `out` (atomic; skipped on later calls), 0 returned."""
    from PIL import Image

    if os.path.exists(out) and (not verify or png_ok(out)):
        return 0
    mark = undecodable and f"{undecodable}/{os.path.basename(out)}"
    if mark and os.path.exists(out):                                # cached PNG failed png_ok: drop it
        os.remove(out)
    if mark and os.path.exists(mark):
        return 0
    for attempt in range(1, attempts + 1):
        data = fetch(f"{API}/rawFiles/{index[key_of(p)]['file_id']}")
        try:
            im = Image.open(io.BytesIO(data))
            im.load()                                               # decode here: thumbnail() may skip it
            im = im.convert("L") if im.mode not in ("L", "I;16") else im
            if im.mode == "I;16":                                   # 16-bit: rescale to 8 bit over the full range
                im = im.point(lambda v: v / 256).convert("L")
            im.thumbnail((1024, 1024), Image.LANCZOS)
            break
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as e:
            print(f"[save_png] {key_of(p)}: attempt {attempt}/{attempts} does not decode ({len(data)} bytes, "
                  f"starts {data[:16]!r}): {e}", file=sys.stderr, flush=True)
            if attempt == attempts:
                if not undecodable:
                    raise OSError(f"{key_of(p)}: undecodable after {attempts} downloads") from e
                open(mark + ".tmp", "w").write(f"{key_of(p)}: {len(data)} bytes, {e}\n")
                os.replace(mark + ".tmp", mark)
                return 0
    im.save(out + ".tmp.png")
    os.replace(out + ".tmp.png", out)
    return 1


def images(a):
    train, valid, index = frontal_studies(a.out)
    old = f"{a.out}/subset.json"
    keep = json.load(open(old))["train"] if os.path.exists(old) else []
    if len(keep) > a.n:
        raise SystemExit(f"{old} already holds {len(keep)} train studies > --n {a.n}")
    rest = sorted(set(train) - set(keep))                          # grow an earlier subset: its studies stay in
    pick = sorted(keep + random.Random(len(keep)).sample(rest, a.n - len(keep))) + valid
    json.dump({"train": pick[:a.n], "valid": valid}, open(old, "w"), indent=0)
    os.makedirs(f"{a.out}/png", exist_ok=True)

    with ThreadPoolExecutor(a.workers) as pool:
        done = sum(pool.map(lambda p: save_png(index, p, f"{a.out}/png/" + key_of(p).replace("/", "_")), pick))
    print(f"downloaded {done}, total {len(pick)} (train {a.n}, valid {len(valid)})", flush=True)


def scale_plan(a):
    """Every train-split study (align.coco.split_of of the patient: 4 of 6 patients) not in subset.json, in a seeded
    random order cut into --chunks equal parts. subset.json, the PNGs and the features of the 30,000 stay untouched."""
    from align.coco import split_of

    train, valid, index = frontal_studies(a.out)
    keep = json.load(open(f"{a.out}/subset.json"))["train"]
    have = set(keep)
    new = [p for p in train if p not in have and split_of(p.split("/")[1]) == "train"]
    random.Random(len(keep)).shuffle(new)
    size = -(-len(new) // a.chunks)
    chunks = [new[i * size:(i + 1) * size] for i in range(a.chunks)]
    os.makedirs(a.dest, exist_ok=True)
    plan = {"base": keep, "valid": json.load(open(f"{a.out}/subset.json"))["valid"], "chunks": chunks}
    json.dump(plan, open(f"{a.dest}/scale_plan.json", "w"), indent=0)
    n_train = sum(split_of(p.split("/")[1]) == "train" for p in train)
    print(f"frontal studies: train-file {len(train)}, of them train-split {n_train}; subset.json train {len(keep)} "
          f"(train-split {sum(split_of(p.split('/')[1]) == 'train' for p in keep)}); new {len(new)} in {a.chunks} "
          f"chunks of {[len(c) for c in chunks]}", flush=True)


def accept_all(dest, emb, pool_workers, max_excluded=0.001):
    """Write EMB/excluded_pngs.txt and then DEST/IMAGES_DONE (both atomic; IMAGES_DONE holds the PNG count) when every
    study of every chunk has a PNG that decodes or a record in DEST/undecodable/ (those studies are excluded). Quiet
    return while another chunk is still missing studies; a present but undecodable PNG is reported and no marker
    written; more than `max_excluded` of the studies excluded is a systemic fault: exit 1, no marker."""
    plan = json.load(open(f"{dest}/scale_plan.json"))["chunks"]
    names = [key_of(p).replace("/", "_") for c in plan for p in c]
    files = [f"{dest}/png/{n}" for n in names]
    excluded = [n for n, f in zip(names, files) if not os.path.exists(f) and os.path.exists(f"{dest}/undecodable/{n}")]
    have = [f for f in files if os.path.exists(f)]
    if len(have) + len(excluded) != len(files):
        print("[accept] other chunks not finished, no IMAGES_DONE yet", flush=True)
        return
    with ThreadPoolExecutor(pool_workers) as pool:
        bad = [f for f, ok in zip(have, pool.map(png_ok, have)) if not ok]
    if bad:
        print(f"[accept] {len(bad)} PNGs do not decode, e.g. {bad[:3]}; no IMAGES_DONE", flush=True)
        return
    print(f"[accept] {len(excluded)} of {len(files)} studies excluded (undecodable source): {sorted(excluded)}", flush=True)
    if len(excluded) > max_excluded * len(files):
        raise SystemExit(f"[accept] more than {max_excluded:.1%} of the studies undecodable; no IMAGES_DONE")
    for path, text in ((f"{emb}/excluded_pngs.txt", "".join(f"{n}\n" for n in sorted(excluded))),
                       (f"{dest}/IMAGES_DONE", f"{len(have)}\n")):
        tmp = f"{path}.tmp{os.getpid()}"
        open(tmp, "w").write(text)
        os.replace(tmp, path)
    print(f"[accept] IMAGES_DONE written, {len(have)} PNGs decode, {len(excluded)} excluded", flush=True)


def scale_images(a):
    _, _, index = frontal_studies(a.out)
    chunk = json.load(open(f"{a.dest}/scale_plan.json"))["chunks"][a.chunk]
    png, undec = f"{a.dest}/png", f"{a.dest}/undecodable"
    os.makedirs(png, exist_ok=True)
    os.makedirs(undec, exist_ok=True)
    name = lambda p: key_of(p).replace("/", "_")
    with ThreadPoolExecutor(a.workers) as pool:
        done = sum(pool.map(lambda p: save_png(index, p, f"{png}/{name(p)}", True, undecodable=undec), chunk))
        ok = list(pool.map(lambda p: png_ok(f"{png}/{name(p)}"), chunk))
    skipped = [name(p) for p, o in zip(chunk, ok) if not o and os.path.exists(f"{undec}/{name(p)}")]
    if sum(ok) + len(skipped) != len(chunk):
        raise SystemExit(f"chunk {a.chunk}: {sum(ok)} decodable PNGs + {len(skipped)} undecodable of {len(chunk)}")
    print(f"chunk {a.chunk}: downloaded {done}, decodable {sum(ok)}, undecodable {len(skipped)} {skipped}, "
          f"total {len(chunk)}", flush=True)
    accept_all(a.dest, a.emb, pool_workers=a.workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["meta", "images", "scale-plan", "scale-images"])
    ap.add_argument("--out")
    ap.add_argument("--dest", help="scale-*: directory all writes go to (never the --out source directory)")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--emb", help="scale-images: cases directory that receives excluded_pngs.txt")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.step in ("meta", "images", "scale-plan", "scale-images") and not a.out:
        ap.error("--out is required")
    if a.step.startswith("scale-") and (not a.dest or (a.out and os.path.realpath(a.out) == os.path.realpath(a.dest))):
        ap.error("scale-*: --dest is required and must differ from --out")
    if a.step == "scale-images" and not a.emb:
        ap.error("scale-images: --emb is required")
    {"meta": meta, "images": images, "scale-plan": scale_plan, "scale-images": scale_images}[a.step](a)


if __name__ == "__main__":
    main()
