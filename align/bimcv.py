"""BIMCV-R as an external cohort: BIMCV-COVID19 chest CT (Valencia) with Spanish reports, their English translation
and the PadChest-style finding labels extracted from the Spanish text (HF dataset cyd0806/BIMCV-R).

    python -m align.bimcv index  --out DIR                                  (internet)
    python -m align.bimcv cases  --out DIR --ctrate-emb DIR
    python -m align.bimcv wan-context --out DIR --datasets DIR                         (GPU)
    python -m align.bimcv parity --out DIR --ctrate-emb DIR --models DIR --datasets DIR    (GPU, internet)
    python -m align.bimcv images --out DIR --models DIR --datasets DIR --rank R --world N (GPU, internet)
    python -m align.bimcv merge  --out DIR

Volumes are read straight out of the remote zips with HTTP range requests, one member per case: no zip is downloaded
whole and no raw volume reaches the shared file system (each goes to $TMPDIR, is encoded and segmented, then
deleted). Image side per case: the encoders of align.ctrate (same preprocessing) and TotalSegmentator 2.7.0 (the
version of the CT-RATE masks) for the anchor measurements of align.measures. Report side: Report_en.

index: central directory of every CT zip (member offset, sizes, CRC) and the metadata table, to DIR/index.
cases: one volume per patient for a hashed sample of patients; reports that describe a radiograph and not a CT are
removed; split by a second hash digit into "train" (unpaired halves, known pairs) and "test". The CT-RATE labels that
have a PadChest counterpart are mapped (LABEL_MAP); the other CT-RATE labels are absent from cases.jsonl.
parity: before any BIMCV feature, PARITY_N CT-RATE test cases of --ctrate-emb are re-encoded and re-segmented here and
compared with the CT-RATE features and the measurements from the released masks; any miss raises.
"""
import argparse
import ast
import glob
import hashlib
import io
import json
import os
import re
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib

import numpy as np
import pandas as pd
import torch

from align.anchor_zero import LUNG
from align.ctrate import build_encoders, encode_volume, to_1mm
from align.extract import feature_ok
from align.measures import TASKS, measure_masks

REPO = "cyd0806/BIMCV-R"
CTRATE_REPO = "ibrahimhamamci/CT-RATE"
CT_ZIPS = [f"CT_{g}_{p}.zip" for g in range(1, 11) for p in range(1, 5)]
META_ZIP, META_CSV = "meta_1.zip", "curated_ct_report_path_En.csv"
SAMPLE_PER_MILLE = 240           # hashed patient sample: 1,090 patients after the case filters
TEST_DIGIT = 0                   # second hash digit (mod 5) of the test patients
CT_TERMS = re.compile(r"\btc\b|\btac\b|tcar|tacar|\btcmd|angio ?tc|angiotc|tomograf|ventana pulmonar|\bhrct")
UNTRANSLATED = "Something wrong with the data"    # Report_en of rows whose translation failed
RX_TERMS = re.compile(r"\brx\b|radiograf|proyecci|\bpa y lat|portatil")
LABEL_MAP = {
    "Medical material": ("NSG tube", "central venous catheter", "central venous catheter via jugular vein",
                         "central venous catheter via subclavian vein", "reservoir central venous catheter", "catheter",
                         "chest drain tube", "endotracheal tube", "tracheostomy tube", "pacemaker",
                         "artificial aortic heart valve", "artificial mitral heart valve", "artificial heart valve",
                         "aortic endoprosthesis", "endoprosthesis", "suture material", "osteosynthesis material"),
    "Arterial wall calcification": ("aortic atheromatosis",),
    "Cardiomegaly": ("cardiomegaly",),
    "Pericardial effusion": ("pericardial effusion",),
    "Hiatal hernia": ("hiatal hernia",),
    "Lymphadenopathy": ("adenopathy",),
    "Emphysema": ("emphysema",),
    "Atelectasis": ("atelectasis", "laminar atelectasis", "lobar atelectasis", "segmental atelectasis",
                    "atelectasis basal", "round atelectasis", "total atelectasis"),
    "Lung nodule": ("nodule", "multiple nodules"),
    "Lung opacity": ("ground glass pattern", "infiltrates", "alveolar pattern", "increased density"),
    "Pulmonary fibrotic sequela": ("fibrotic band", "pulmonary fibrosis", "tuberculosis sequelae"),
    "Pleural effusion": ("pleural effusion", "loculated pleural effusion", "loculated fissural effusion"),
    "Consolidation": ("consolidation",),
    "Bronchiectasis": ("bronchiectasis",),
}
PARITY_N = 10
PARITY_COS = 0.99                # every encoder, every parity case
PARITY_REL = 0.05                # heart and lung ml: relative difference
PARITY_EFFUSION_ML = 10.0        # pleural and pericardial effusion ml: absolute difference
HU_P1_MAX, HU_P99_MIN = -900.0, 100.0
SPACING_MM = (0.2, 10.0)        # header voxel spacing outside this range is corrupt (e.g. 268.9 mm slices)
EXTENT_MM = (2000.0, 900.0, 900.0)  # sorted extents: whole-body scan length, then twice the widest gantry bore


def token():
    return os.environ["HF_TOKEN"]


def url_of(repo, name):
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{name}"


class Remote:
    """Byte ranges of one file on the Hub; the signed CDN URL is resolved once and again when it expires."""

    def __init__(self, repo, name):
        self.src = url_of(repo, name)
        self.resolve()

    def resolve(self):
        req = urllib.request.Request(self.src, method="HEAD", headers={"Authorization": f"Bearer {token()}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            self.url, self.size = r.url, int(r.headers["Content-Length"])

    def read(self, start, n):
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + n - 1}"})
                with urllib.request.urlopen(req, timeout=600) as r:
                    if r.status != 206:
                        raise IOError(f"range request answered with status {r.status}")
                    b = r.read(n)
                if len(b) != n:
                    raise IOError(f"short read {len(b)} of {n} bytes")
                return b
            except (urllib.error.URLError, IOError, TimeoutError) as e:
                if attempt == 4:
                    raise
                print(f"[bimcv] range read retry {attempt + 1}: {e}", flush=True)
                time.sleep(10 * (attempt + 1))
                self.resolve()


class RemoteFile(io.RawIOBase):
    """Seekable file view of a Remote, enough for zipfile to read a central directory."""

    def __init__(self, remote):
        self.r, self.pos = remote, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = (off, self.pos + off, self.r.size + off)[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.r.size - self.pos)
        if n <= 0:
            return 0
        b[:n] = self.r.read(self.pos, n)
        self.pos += n
        return n


def build_index(out):
    os.makedirs(f"{out}/index", exist_ok=True)
    rows = []
    for z in CT_ZIPS:
        for i in zipfile.ZipFile(RemoteFile(Remote(REPO, z))).infolist():
            if i.filename.endswith(".nii.gz"):
                rows.append({"zip": z, "member": i.filename, "name": os.path.basename(i.filename), "offset": i.header_offset,
                             "csize": i.compress_size, "size": i.file_size, "crc": i.CRC, "method": i.compress_type})
        print(f"[index] {z} members so far {len(rows)}", flush=True)
    idx = pd.DataFrame(rows)
    if idx.name.duplicated().any():
        raise ValueError(f"{int(idx.name.duplicated().sum())} volume names occur in more than one zip")
    idx.to_csv(f"{out}/index/members.csv", index=False)
    meta = zipfile.ZipFile(RemoteFile(Remote(REPO, META_ZIP)))
    with meta.open(META_CSV) as f:
        open(f"{out}/index/{META_CSV}", "wb").write(f.read())
    print(f"[index] volumes={len(idx)} compressed={idx.csize.sum() / 1e9:.0f} GB", flush=True)


def fetch(row, dst):
    """One zip member by range request: local header, data, inflate if needed, CRC check."""
    r = Remote(REPO, row["zip"])
    head = r.read(int(row["offset"]), 30)
    sig, *_, name_len, extra_len = struct.unpack("<IHHHHHIIIHH", head)
    if sig != 0x04034B50:
        raise ValueError(f"no local file header at {row['offset']} in {row['zip']}")
    data = r.read(int(row["offset"]) + 30 + name_len + extra_len, int(row["csize"]))
    if row["method"] == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif row["method"] != zipfile.ZIP_STORED:
        raise ValueError(f"compression method {row['method']} of {row['name']}")
    if len(data) != row["size"] or zlib.crc32(data) != row["crc"]:
        raise ValueError(f"size or CRC mismatch for {row['name']}")
    with open(dst + ".tmp", "wb") as f:
        f.write(data)
    os.replace(dst + ".tmp", dst)


def hash_int(s):
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


def build_cases(out, ctrate_emb):
    names = json.load(open(f"{ctrate_emb}/labels.json"))["labels"]
    missing = set(LABEL_MAP) - set(names)
    if missing:
        raise KeyError(f"LABEL_MAP names not among the CT-RATE labels: {missing}")
    meta = pd.read_csv(f"{out}/index/{META_CSV}")
    members = pd.read_csv(f"{out}/index/members.csv").set_index("name")
    found = {l.strip() for x in meta.Labels.dropna() for l in ast.literal_eval(x)}
    unused = {p for ps in LABEL_MAP.values() for p in ps} - found
    if unused:
        raise KeyError(f"PadChest labels of LABEL_MAP never occur: {unused}")
    es = meta.Report.fillna("").str.lower()
    ids = meta.path.str.extract(r"^(sub-[^_]+)_(ses-[^_]+)_")
    en = meta.Report_en.fillna("").str.strip()
    meta = meta.assign(radiograph=es.str.contains(RX_TERMS) & ~es.str.contains(CT_TERMS),
                       # the last 1,333 rows (the COVID-negative part) carry another row's patient, session and report
                       mislinked=(ids[0] != meta.PatientID) | (ids[1] != meta.ReportID),
                       untranslated=en.eq("") | en.str.contains(UNTRANSLATED, regex=False),
                       shared_report=meta.duplicated("Report", keep=False))
    keep = meta[~(meta.radiograph | meta.mislinked | meta.untranslated | meta.shared_report)]
    keep = keep[keep.PatientID.map(lambda p: hash_int(p) % 1000 < SAMPLE_PER_MILLE)]
    keep = keep.sort_values("path").groupby("PatientID", sort=True).head(1)
    rows = []
    for r in keep.itertuples():
        found_labels = {l.strip() for l in ast.literal_eval(r.Labels)} if isinstance(r.Labels, str) else set()
        m = members.loc[r.path]
        rows.append({"id": r.path.replace(".nii.gz", ""), "key": r.path.replace(".nii.gz", ""), "patient": r.PatientID,
                     "split": "test" if (hash_int(r.PatientID) // 1000) % 5 == TEST_DIGIT else "train",
                     "report": str(r.Report_en).strip(), "zip": m["zip"], "offset": int(m["offset"]),
                     "csize": int(m["csize"]), "size": int(m["size"]), "crc": int(m["crc"]), "method": int(m["method"]),
                     **{c: int(bool(found_labels & set(ps))) for c, ps in LABEL_MAP.items()}})
    cases = pd.DataFrame(rows)
    cases.to_json(f"{out}/cases.jsonl.tmp", orient="records", lines=True)
    os.replace(f"{out}/cases.jsonl.tmp", f"{out}/cases.jsonl")
    labels = json.load(open(f"{ctrate_emb}/labels.json"))
    json.dump(labels, open(f"{out}/labels.json.tmp", "w"), indent=1)   # the CT-RATE prompts, for the text side
    os.replace(f"{out}/labels.json.tmp", f"{out}/labels.json")
    print(f"[cases] meta={len(meta)} " + " ".join(f"{c}={int(meta[c].sum())}" for c in ("radiograph", "mislinked", "untranslated", "shared_report"))
          + f" cases={len(cases)} "
          f"{cases.split.value_counts().to_dict()} compressed={cases.csize.sum() / 1e9:.0f} GB "
          f"prevalence={ {c: round(float(cases[c].mean()), 3) for c in LABEL_MAP} }", flush=True)


def load_hu(path):
    """NIfTI -> LPS zyx float HU tensor and spacing zyx, as align.ctrate.Nifti."""
    import nibabel as nib

    img = nib.as_closest_canonical(nib.load(path))
    a = np.asarray(img.dataobj, dtype=np.float32)[::-1, ::-1, :]
    sx, sy, sz = (float(v) for v in img.header.get_zooms()[:3])
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 1, 0))), (sz, sy, sx)


def segment(path, key):
    """TotalSegmentator 2.7.0 caps torch below 2.6 (nnU-Net checkpoints do not load under the weights_only default of
    torch 2.6), so it runs from its own venv (TOTALSEG_BIN) through its command line, one multi-label file per task."""
    import subprocess

    import nibabel as nib

    masks = {}
    for t in TASKS:
        dst = f"{path[:-len('.nii.gz')]}_{t}.nii.gz"
        subprocess.run([os.environ["TOTALSEG_BIN"], "-i", path, "-o", dst, "--task", t.removeprefix("ts_"), "--ml",
                        "--device", "gpu", "--quiet"], check=True)
        masks[t] = nib.load(dst)
    try:
        return measure_masks(key, masks)
    finally:
        for t in TASKS:
            os.remove(f"{path[:-len('.nii.gz')]}_{t}.nii.gz")


def hu_check(hu):
    x = hu.flatten()
    q = torch.quantile(x[::max(97, x.numel() // 2 ** 24 + 1)], torch.tensor([.01, .5, .99], device=hu.device)).tolist()  # quantile() caps input at 2^24
    return q, q[0] <= HU_P1_MAX and q[2] >= HU_P99_MIN


class Fetched(torch.utils.data.Dataset):
    """Worker side: range-read one case into tmp, load it. The main process encodes, segments and deletes it."""

    def __init__(self, rows, tmp, fetcher):
        self.rows, self.tmp, self.fetcher = rows, tmp, fetcher

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        """Header check first: a corrupt header can describe a volume far too large to load (one took 56 GB)."""
        import nibabel as nib

        row = self.rows[i]
        dst = f"{self.tmp}/{row['key']}.nii.gz"
        t = time.perf_counter()
        self.fetcher(row, dst)
        hdr = nib.load(dst).header
        shape, sp = [int(v) for v in hdr.get_data_shape()[:3]], [float(v) for v in hdr.get_zooms()[:3]]
        ext = sorted((n * s for n, s in zip(shape, sp)), reverse=True)
        if not all(SPACING_MM[0] <= s <= SPACING_MM[1] for s in sp):
            reject = {"reason": "implausible voxel spacing", "spacing_mm": sp, "shape": shape}
        elif any(e > m for e, m in zip(ext, EXTENT_MM)):
            reject = {"reason": "implausible field of view", "extent_mm": ext, "shape": shape}
        else:
            vol, spacing = load_hu(dst)
            return row["key"], dst, vol, spacing, None, time.perf_counter() - t
        return row["key"], dst, None, None, reject, time.perf_counter() - t


def run_images(out, names, rank, world):
    cases = pd.read_json(f"{out}/cases.jsonl", lines=True)
    path = lambda n, k: f"{out}/{n}/{k}.npy"
    mpath = lambda k: f"{out}/measures/{k}.json"
    for n in names:
        os.makedirs(f"{out}/{n}", exist_ok=True)
    os.makedirs(f"{out}/measures", exist_ok=True)
    os.makedirs(f"{out}/rejected", exist_ok=True)
    rows = [r for r in cases.to_dict("records")[rank::world]
            if not os.path.exists(f"{out}/rejected/{r['key']}.json")
            and not (all(feature_ok(path(n, r["key"])) for n in names) and os.path.exists(mpath(r["key"])))]
    print(f"[bimcv] rank={rank} todo={len(rows)} encoders={names}", flush=True)
    encoders = build_encoders(names)
    tmp = os.environ["TMPDIR"]
    # segmentation takes ~95% of the time, so 3 workers keep the GPU fed; each prefetched volume costs host memory
    loader = torch.utils.data.DataLoader(Fetched(rows, tmp, fetch), batch_size=None, num_workers=3, prefetch_factor=1,
                                         collate_fn=lambda b: b)
    t0, fetch_s, seg_s, enc_s = time.perf_counter(), 0.0, 0.0, 0.0
    for i, (key, local, vol, spacing, reject, fs) in enumerate(loader):
        fetch_s += fs
        if reject is not None:
            json.dump({"key": key, **reject}, open(f"{out}/rejected/{key}.json", "w"))
            print(f"[bimcv] rejected {key}: {reject}", flush=True)
            os.remove(local)
            continue
        sp = [float(s) for s in spacing]
        hu = to_1mm(vol.cuda(non_blocking=True), sp)
        q, ok = hu_check(hu)
        if i == 0:
            print(f"[bimcv] first volume {key} spacing={spacing} shape1mm={tuple(hu.shape)} HU p1/p50/p99={[round(v) for v in q]}",
                  flush=True)
        if not ok:
            json.dump({"key": key, "reason": "not HU-like", "hu_p1_p50_p99": q}, open(f"{out}/rejected/{key}.json", "w"))
            print(f"[bimcv] rejected {key}: HU p1/p50/p99={q}", flush=True)
            os.remove(local)
            continue
        try:
            t = time.perf_counter()
            if not os.path.exists(mpath(key)):
                torch.cuda.empty_cache()     # TotalSegmentator is a subprocess on this GPU: free the last encode's cache
                json.dump(segment(local, key), open(mpath(key) + ".tmp", "w"))
                os.replace(mpath(key) + ".tmp", mpath(key))
            seg_s += time.perf_counter() - t
            t = time.perf_counter()
            feats = encode_volume({n: e for n, e in encoders.items() if not feature_ok(path(n, key))}, hu)
            for n, feat in feats.items():
                np.save(path(n, key) + ".tmp.npy", feat.cpu().numpy())
                os.replace(path(n, key) + ".tmp.npy", path(n, key))
            enc_s += time.perf_counter() - t
        finally:
            os.remove(local)
        if i % 20 == 0 or i == len(rows) - 1:
            el = time.perf_counter() - t0
            print(f"[bimcv] rank={rank} {i + 1}/{len(rows)} {(i + 1) / el * 3600:.0f} vol/h segment={seg_s / el:.0%} "
                  f"encode={enc_s / el:.0%} fetch(workers)={fetch_s / (i + 1):.0f} s/vol", flush=True)
    print(f"[bimcv] rank={rank} done", flush=True)


def merge(out):
    """measures.csv of the kept volumes. A volume whose TotalSegmentator mask has no lung is not a chest CT (BIMCV
    file names say bp-chest_ct for some pelvis scans) and is rejected here, like the header and HU rejections."""
    cases = pd.read_json(f"{out}/cases.jsonl", lines=True)
    rejected = {os.path.basename(p)[:-5] for p in glob.glob(f"{out}/rejected/*.json")}
    m = pd.DataFrame([json.load(open(f"{out}/measures/{k}.json")) for k in cases.key if k not in rejected])
    m = m.fillna(0.0).set_index("key")
    lung = m.reindex(columns=LUNG, fill_value=0.0).sum(1)
    for key in m.index[lung <= 0]:
        json.dump({"key": key, "reason": "no lung in the TotalSegmentator mask"}, open(f"{out}/rejected/{key}.json", "w"))
        print(f"[merge] rejected {key}: no lung in the TotalSegmentator mask", flush=True)
    m[lung > 0].to_csv(f"{out}/measures.csv")
    print(f"[merge] measured={int((lung > 0).sum())} rejected={len(rejected) + int((lung <= 0).sum())}", flush=True)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not (na > 0 and nb > 0):
        raise ValueError(f"zero or non-finite feature norm ({na}, {nb})")
    return float(np.dot(a, b) / na / nb)


def absent_is_zero(row, column):
    """measure_masks writes a label only when it has voxels; merge fills the gaps with 0 ml the same way."""
    return row[column] if column in row else 0.0


def parity(out, ctrate_emb, names):
    """CT-RATE test cases re-encoded and re-segmented with this environment against the CT-RATE features and the
    measurements from the released masks."""
    from huggingface_hub import hf_hub_download

    cases = pd.read_json(f"{ctrate_emb}/cases.jsonl", lines=True)
    ref = pd.read_csv(f"{ctrate_emb}/measures.csv").set_index("key")
    pick = cases[cases.split == "test"].sort_values("key").head(PARITY_N)
    encoders = build_encoders(names)
    tmp = os.environ["TMPDIR"]
    lung = [f"ts_total_{i}" for i in range(10, 15)]
    report, fails = [], []
    for r in pick.itertuples():
        local = hf_hub_download(CTRATE_REPO, f"dataset/{r.key}.nii.gz", repo_type="dataset", local_dir=tmp, token=token())
        vol, spacing = load_hu(local)
        hu = to_1mm(vol.cuda(), list(spacing))
        feats = encode_volume(encoders, hu)
        cos = {n: cosine(f.cpu().numpy(), np.load(f"{ctrate_emb}/{n}/{r.id}.npy")) for n, f in feats.items()}
        m, o = segment(local, r.key), ref.loc[r.key]
        mine = {"heart": m["ts_total_51"], "lung": sum(m[c] for c in lung),
                "pleural": absent_is_zero(m, "ts_pleural_pericard_effusion_2"),
                "pericardial": absent_is_zero(m, "ts_pleural_pericard_effusion_3"),
                "nodules": m["ts_lung_nodules_components"]}
        theirs = {"heart": o["ts_total_51"], "lung": o[lung].sum(), "pleural": o["ts_pleural_pericard_effusion_2"],
                  "pericardial": o["ts_pleural_pericard_effusion_3"], "nodules": o["ts_lung_nodules_components"]}
        report.append({"key": r.key, "cos": cos, "mine": mine, "released": theirs})
        fails += [f"{r.key} {n} cos={c:.4f}" for n, c in cos.items() if not c >= PARITY_COS]
        fails += [f"{r.key} {k} {mine[k]:.0f} vs {theirs[k]:.0f} ml" for k in ("heart", "lung")
                  if abs(mine[k] - theirs[k]) > PARITY_REL * theirs[k]]
        fails += [f"{r.key} {k} {mine[k]:.1f} vs {theirs[k]:.1f} ml" for k in ("pleural", "pericardial")
                  if abs(mine[k] - theirs[k]) > PARITY_EFFUSION_ML]
        print(f"[parity] {r.key} min cos={min(cos.values()):.5f} ({min(cos, key=cos.get)}) "
              + " ".join(f"{k}={mine[k]:.0f}/{theirs[k]:.0f}" for k in mine), flush=True)
        os.remove(local)
    os.makedirs(out, exist_ok=True)
    json.dump({"cases": report, "fails": fails}, open(f"{out}/parity.json", "w"), indent=1)
    if fails:
        raise RuntimeError(f"parity failed ({len(fails)}): " + "; ".join(fails))
    print("[parity] pass", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["index", "cases", "wan-context", "parity", "images", "merge"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctrate-emb")
    ap.add_argument("--models", help="directory of the zoo weights (align.zoo.MODELS)")
    ap.add_argument("--datasets", help="directory of the Wan / LTX / DINOv3 weights (align.zoo.DATASETS)")
    ap.add_argument("--encoders")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    a = ap.parse_args()
    if a.step in ("parity", "images"):
        from align import zoo

        zoo.MODELS, zoo.DATASETS = a.models, a.datasets
    if a.step == "index":
        build_index(a.out)
    elif a.step == "cases":
        build_cases(a.out, a.ctrate_emb)
    elif a.step == "parity":
        parity(a.out, a.ctrate_emb, a.encoders.split(","))
    elif a.step == "wan-context":
        from align.zoo import write_empty_context

        write_empty_context(f"{a.datasets}/wan21-t2v-1.3b")
    elif a.step == "images":
        run_images(a.out, a.encoders.split(","), a.rank, a.world)
    else:
        merge(a.out)


if __name__ == "__main__":
    main()
