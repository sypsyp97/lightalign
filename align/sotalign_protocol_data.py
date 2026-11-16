"""Prepare the CC3M protocol without retaining CC3M image archives."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tarfile
import time
import zipfile

import numpy as np
import requests

VISION_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
VISION_REV = "ea8dc2863c51be0a264bab82070e3e8836b02d51"
TEXT_ID = "nvidia/NV-Embed-v2"
TEXT_REV = "3fa59658547db50a1e8e3346cf057fd0c77ed6ef"
CC_ID = "pixparse/cc3m-wds"
CAPTION_ID = "qidouxiong619/dreamlip_long_captions"
CAPTION_FILE = "cc3m_3long_3short_1raw_captions_url.csv"
SOURCE_REV = "64c89eedd2f57f437e9d52575f47320e11db2fe7"
PAIR_RESERVE = 110000
UNPAIRED_RESERVE = 1100000


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def download(url, path):
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with temp.open("wb") as out:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        out.write(chunk)
            temp.replace(path)
            return path
        except (requests.RequestException, OSError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(url)


def select_sources(root):
    root = Path(root)
    info = json.loads((root / "cc3m_info.json").read_text())
    n = info["splits"]["train"]["num_samples"]
    permutation = np.random.RandomState(42).permutation(n)
    roles = np.zeros(n, dtype=np.uint8)
    roles[permutation[:PAIR_RESERVE]] = 1
    roles[permutation[PAIR_RESERVE:PAIR_RESERVE + UNPAIRED_RESERVE]] = 2
    roles[permutation[PAIR_RESERVE + UNPAIRED_RESERVE:
                      PAIR_RESERVE + 2 * UNPAIRED_RESERVE]] = 3
    temp = root / "source_roles.tmp.npy"
    np.save(temp, roles)
    digest = hashlib.sha256(temp.read_bytes()).hexdigest()
    if list((root / "cc_features/train").glob("*.json")):
        previous = hashlib.sha256((root / "source_roles.npy").read_bytes()).hexdigest()
        if previous != digest:
            raise ValueError("Cannot change source roles after feature shards have been published")
    temp.replace(root / "source_roles.npy")
    selected = {"cc3m_source_count": n,
                "role_counts": {str(k): int((roles == k).sum()) for k in range(4)}, "roles_sha256": digest}
    for name in ("access_and_sources", "setup"):
        path = root / "results" / f"{name}.json"
        if path.exists():
            provenance = json.loads(path.read_text())
            provenance.update(selected)
            write_json(path, provenance)
    write_json(root / "results/source_selection.json", selected)
    print(json.dumps(selected), flush=True)
    return selected


def setup(root):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    root = Path(root)
    api = HfApi()
    provenance = {"sotalign_revision": SOURCE_REV, "models": {}, "datasets": {}}
    for name, revision in ((VISION_ID, VISION_REV), (TEXT_ID, TEXT_REV)):
        info = api.model_info(name, revision=revision)
        provenance["models"][name] = {"revision": info.sha, "gated": info.gated}
    for name in (CC_ID, CAPTION_ID):
        info = api.dataset_info(name)
        provenance["datasets"][name] = info.sha
    meta = Path(hf_hub_download(CC_ID, "_info.json", repo_type="dataset",
                              revision=provenance["datasets"][CC_ID]))
    info = json.loads(meta.read_text())
    write_json(root / "cc3m_info.json", info)
    provenance.update(select_sources(root))
    write_json(root / "results" / "access_and_sources.json", provenance)
    caption_path = hf_hub_download(CAPTION_ID, CAPTION_FILE, repo_type="dataset",
                                  revision=provenance["datasets"][CAPTION_ID], local_dir=root / "metadata")
    db_path = root / "captions.sqlite"
    if not db_path.exists():
        temp_db = Path(os.environ["TMPDIR"]) / "sotalign-captions.sqlite"
        temp_db.unlink(missing_ok=True)
        db = sqlite3.connect(temp_db)
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA cache_size=-262144")
        db.execute("CREATE TABLE captions (url TEXT PRIMARY KEY, raw_caption TEXT NOT NULL)")
        with open(caption_path, newline="") as f:
            reader = csv.DictReader(f)
            if not {"Image Path", "raw_caption"}.issubset(reader.fieldnames):
                raise ValueError(f"Unexpected DreamLIP columns: {reader.fieldnames}")
            batch, processed = [], 0
            for row in reader:
                batch.append((row["Image Path"], row["raw_caption"]))
                if len(batch) == 50000:
                    db.executemany("INSERT OR REPLACE INTO captions VALUES (?, ?)", batch)
                    db.commit()
                    processed += len(batch)
                    print(f"[captions] indexed={processed}", flush=True)
                    batch.clear()
            db.executemany("INSERT OR REPLACE INTO captions VALUES (?, ?)", batch)
            db.commit()
        db.close()
        shared_temp = Path(str(db_path) + ".tmp")
        shutil.copyfile(temp_db, shared_temp)
        shared_temp.replace(db_path)
        Path(str(shared_temp) + "-journal").unlink(missing_ok=True)
        temp_db.unlink()
    with sqlite3.connect(db_path) as db:
        provenance["dreamlip_urls"] = db.execute("SELECT COUNT(*) FROM captions").fetchone()[0]
    snapshot_download(TEXT_ID, revision=TEXT_REV, local_dir=root / "models" / "NV-Embed-v2",
                      ignore_patterns=["*.md", "*.png", "*.jpg", "1_Pooling/*"])
    write_json(root / "results" / "setup.json", provenance)
    print(json.dumps(provenance), flush=True)


def benchmarks(root, data_root, image_root):
    root, data_root, image_root = map(Path, (root, data_root, image_root))
    archive = download("https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip",
                       root / "metadata" / "caption_datasets.zip")
    datasets = {}
    with zipfile.ZipFile(archive) as z:
        for domain in ("coco", "flickr30k"):
            name = f"dataset_{domain}.json"
            members = [n for n in z.namelist() if n.endswith(name)]
            if len(members) != 1:
                raise ValueError((name, members))
            source = json.loads(z.read(members[0]))
            rows = []
            for entry in source["images"]:
                if entry["split"] != "test":
                    continue
                if domain == "flickr30k":
                    path = image_root / "domains/flickr30k/flickr30k-images" / entry["filename"]
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    row = {"id": entry["filename"], "image": str(path)}
                else:
                    image_id = int(entry["cocoid"])
                    existing = data_root / "coco/train2017" / f"{image_id:012d}.jpg"
                    path = existing if existing.is_file() else root / "benchmark_images" / entry["filename"]
                    row = {"id": str(image_id), "image": str(path),
                           "url": f"http://images.cocodataset.org/{entry['filepath']}/{entry['filename']}"}
                row["captions"] = [s["raw"] for s in entry["sentences"][:5]]
                if len(row["captions"]) != 5:
                    raise ValueError(row)
                rows.append(row)
            datasets[f"{domain}_karpathy_test"] = rows
    ann_zip = download("http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                       root / "metadata" / "annotations_trainval2017.zip")
    with zipfile.ZipFile(ann_zip) as z:
        annotations = json.loads(z.read("annotations/captions_val2017.json"))
    captions = {}
    for row in annotations["annotations"]:
        captions.setdefault(row["image_id"], []).append(row["caption"])
    datasets["coco_val2017"] = [
        {"id": str(im["id"]), "image": str(root / "benchmark_images" / im["file_name"]),
         "url": im["coco_url"], "captions": captions[im["id"]][:5]}
        for im in sorted(annotations["images"], key=lambda r: r["id"])]
    expected = {"coco_karpathy_test": 5000, "flickr30k_karpathy_test": 1000, "coco_val2017": 5000}
    for name, rows in datasets.items():
        if len(rows) != expected[name]:
            raise ValueError((name, len(rows)))
        todo = [r for r in rows if not Path(r["image"]).is_file()]
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(lambda r: download(r["url"], r["image"]), todo))
        write_json(root / "manifests" / f"{name}.json", rows)
    write_json(root / "results" / "benchmark_counts.json",
               {name: {"images": len(rows), "captions": sum(len(r["captions"]) for r in rows)}
                for name, rows in datasets.items()})
    ann_zip.unlink()


def iter_tar(path):
    """Read each WebDataset record once, including records that do not need images."""
    with tarfile.open(path) as tf:
        current, record = None, {}
        for member in tf:
            if not member.isfile():
                continue
            key, ext = member.name.rsplit(".", 1)
            if current is not None and key != current:
                yield current, record
                record = {}
            current = key
            record[ext] = tf.extractfile(member).read()
        if current is not None:
            yield current, record


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("setup", "benchmarks", "select"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--image-root")
    args = parser.parse_args()
    if args.stage == "setup":
        setup(args.root)
    elif args.stage == "select":
        select_sources(args.root)
    else:
        benchmarks(args.root, args.data_root, args.image_root)


if __name__ == "__main__":
    main()
