"""Extract pinned protocol features; keep CC3M features and captions only."""
import argparse
import concurrent.futures
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import time

import h5py
import numpy as np
from PIL import Image
import torch

from align.sotalign_protocol_data import CC_ID, VISION_REV, download, iter_tar, write_json


def upstream_file(source, relative, name):
    spec = importlib.util.spec_from_file_location(name, Path(source) / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Vision:
    def __init__(self, model_dir, source):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
        self.n_registers = self.model.config.num_register_tokens
        module = upstream_file(source, "src/models/vision_model.py", "sotalign_vision_extractor")
        self.transform = module._dinov3_transform()
        self.calls = 0

    @torch.inference_mode()
    def __call__(self, images):
        inputs = torch.stack([self.transform(im.convert("RGB")) for im in images]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model(pixel_values=inputs)
        seq = output.last_hidden_state
        features = torch.cat([seq[:, 0], seq[:, 1 + self.n_registers:].mean(dim=1)], dim=1)
        if features.shape[1] != 2048 or not torch.isfinite(features).all():
            raise ValueError(f"Invalid DINOv3 features: {features.shape}")
        self.calls += 1
        if self.calls == 1:
            torch.cuda.synchronize()
            print(f"[first-batch] images={len(images)} dim=2048 "
                  f"peak_bytes={torch.cuda.max_memory_allocated()} revision={VISION_REV}", flush=True)
        return features.half().cpu().numpy()


def image_shards(args):
    root = Path(args.root)
    info = json.loads((root / "cc3m_info.json").read_text())["splits"][args.split]
    revision = json.loads((root / "results/access_and_sources.json").read_text())["datasets"][CC_ID]
    roles = np.load(root / "source_roles.npy", mmap_mode="r") if args.split == "train" else None
    offsets = np.r_[0, np.cumsum(info["shard_lengths"])]
    out_dir = root / "cc_features" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    names = info["filenames"]
    selected = [i for i in range(args.rank, len(names), args.world)
                if not (out_dir / names[i].replace(".tar", ".json")).exists()]
    vision = Vision(args.model_dir, args.source)
    connection = sqlite3.connect(":memory:")
    with sqlite3.connect(f"file:{root / 'captions.sqlite'}?mode=ro", uri=True) as source_db:
        source_db.backup(connection)
    minimum_match_rate = 1000000 / int((roles == 2).sum()) if roles is not None else 0
    scratch = Path(os.environ["TMPDIR"]) / "sotalign_protocol"
    scratch.mkdir(parents=True, exist_ok=True)

    def fetch(index):
        return download(f"https://huggingface.co/datasets/{CC_ID}/resolve/{revision}/{names[index]}?download=true",
                        scratch / names[index])

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as prefetch:
        future = prefetch.submit(fetch, selected[0]) if selected else None
        for position, shard in enumerate(selected):
            path = future.result()
            future = prefetch.submit(fetch, selected[position + 1]) if position + 1 < len(selected) else None
            start = time.monotonic()
            vectors, image_ids, image_roles, image_keys = [], [], [], []
            batch_images, batch_records, captions = [], [], []
            stats = {"shard": names[shard], "split": args.split, "source_records": 0,
                     "matched_records": 0, "decode_errors": [], "unmatched_selected": 0}

            def flush():
                if not batch_images:
                    return
                vectors.append(vision(batch_images))
                for record in batch_records:
                    image_ids.append(record["source_index"])
                    image_roles.append(record["role"])
                    image_keys.append(record["key"])
                    if record["role"] in (1, 4):
                        captions.append(record)
                batch_images.clear()
                batch_records.clear()

            for local, (key, sample) in enumerate(iter_tar(path)):
                stats["source_records"] += 1
                index = int(offsets[shard]) + local
                role = int(roles[index]) if roles is not None else 4
                if role == 0:
                    continue
                if args.split == "train":
                    metadata = json.loads(sample["json"])
                    found = connection.execute("SELECT raw_caption FROM captions WHERE url=?",
                                               (metadata["url"],)).fetchone()
                    if found is None:
                        stats["unmatched_selected"] += 1
                        continue
                    caption = found[0]
                else:
                    caption = sample["txt"].decode("utf-8")
                stats["matched_records"] += 1
                record = {"source_index": index, "role": role, "key": key, "caption": caption}
                if role == 3:
                    captions.append(record)
                    continue
                formats = [ext for ext in ("jpg", "jpeg", "png", "webp") if ext in sample]
                if len(formats) != 1:
                    raise ValueError((key, sorted(sample)))
                try:
                    im = Image.open(io.BytesIO(sample[formats[0]])).convert("RGB")
                    im.load()
                except (OSError, ValueError) as error:
                    stats["decode_errors"].append({"key": key, "error": str(error)})
                    continue
                batch_images.append(im)
                batch_records.append(record)
                batch_size = min(8, args.batch) if vision.calls == 0 else args.batch
                if len(batch_images) == batch_size:
                    flush()
            flush()
            if stats["source_records"] != info["shard_lengths"][shard]:
                raise ValueError(f"Shard length mismatch: {stats}")
            selected_count = stats["matched_records"] + stats["unmatched_selected"]
            if args.split == "train" and stats["matched_records"] / selected_count < minimum_match_rate + 0.01:
                raise ValueError(f"DreamLIP overlap needs larger reserves before full extraction: {stats}")
            h5_path = out_dir / names[shard].replace(".tar", ".h5")
            temp = h5_path.with_suffix(".tmp.h5")
            with h5py.File(temp, "w") as f:
                f.create_dataset("embeddings", data=np.concatenate(vectors), compression="gzip", compression_opts=1)
                f.create_dataset("ids", data=np.asarray(image_ids, dtype=np.int64))
                f.create_dataset("roles", data=np.asarray(image_roles, dtype=np.uint8))
                f.create_dataset("keys", data=np.asarray(image_keys, dtype=h5py.string_dtype()))
            with h5py.File(temp) as f:
                if len(f["embeddings"]) != len(image_ids) or not np.isfinite(f["embeddings"][:]).all():
                    raise ValueError(f"Feature verification failed: {temp}")
            temp.replace(h5_path)
            text_path = out_dir / names[shard].replace(".tar", ".jsonl")
            text_temp = text_path.with_suffix(".jsonl.tmp")
            with text_temp.open("w") as f:
                for record in captions:
                    f.write(json.dumps(record) + "\n")
            text_temp.replace(text_path)
            stats.update({"image_features": len(image_ids), "text_records": len(captions),
                          "elapsed_seconds": time.monotonic() - start,
                          "peak_gpu_bytes": torch.cuda.max_memory_allocated()})
            write_json(out_dir / names[shard].replace(".tar", ".json"), stats)
            path.unlink()
            print(json.dumps(stats), flush=True)
    connection.close()
    write_json(root / "results" / f"cc_images_{args.split}_rank{args.rank}.json",
               {"rank": args.rank, "world": args.world, "split": args.split,
                "complete_shards": len(list(out_dir.glob("*.json"))),
                "peak_gpu_bytes": torch.cuda.max_memory_allocated()})


def benchmark_images(args):
    root = Path(args.root)
    vision = Vision(args.model_dir, args.source)
    for manifest in sorted((root / "manifests").glob("*_test.json")) + [root / "manifests/coco_val2017.json"]:
        rows = json.loads(manifest.read_text())
        path = root / "banks" / f"{manifest.stem}_images.h5"
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp.h5")
        with h5py.File(temp, "w") as f:
            bank = f.create_dataset("embeddings", (len(rows), 2048), dtype="float16")
            for start in range(0, len(rows), args.batch):
                images = [Image.open(r["image"]).convert("RGB") for r in rows[start:start + args.batch]]
                bank[start:start + len(images)] = vision(images)
            f.create_dataset("ids", data=np.asarray([r["id"] for r in rows], dtype=h5py.string_dtype()))
        temp.replace(path)
        print(f"[benchmark-image] {manifest.stem} count={len(rows)}", flush=True)
    write_json(root / "results/benchmark_image_features.json", {"complete": True,
               "peak_gpu_bytes": torch.cuda.max_memory_allocated()})


def merge_images(args):
    root = Path(args.root)
    bank_dir = root / "banks"
    bank_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "validation"):
        metadata = json.loads((root / "cc3m_info.json").read_text())["splits"][split]
        paths = [root / "cc_features" / split / n.replace(".tar", ".h5") for n in metadata["filenames"]]
        records = []
        source_count, matched_count, unmatched_count, decode_count = 0, 0, 0, 0
        for path in paths:
            marker = path.with_suffix(".json")
            if not marker.exists():
                raise FileNotFoundError(marker)
            stats = json.loads(marker.read_text())
            source_count += stats["source_records"]
            matched_count += stats["matched_records"]
            unmatched_count += stats["unmatched_selected"]
            decode_count += len(stats["decode_errors"])
            with h5py.File(path) as f:
                records.extend((int(i), int(r), path, j) for j, (i, r) in enumerate(zip(f["ids"][:], f["roles"][:])))
        counts[f"cc3m_{split}_source_records"] = source_count
        counts[f"cc3m_{split}_selected_matched"] = matched_count
        counts[f"cc3m_{split}_selected_unmatched"] = unmatched_count
        counts[f"cc3m_{split}_image_decode_errors"] = decode_count
        specifications = ((1, "pairs", None), (2, "unpaired", 1000000)) if split == "train" else ((4, "cc3m_validation", None),)
        keep_ids = {}
        for role, name, target in specifications:
            selected = sorted((r for r in records if r[1] == role), key=lambda r: r[0])
            counts[f"{name}_image_available"] = len(selected)
            if target is not None:
                if len(selected) < target:
                    raise ValueError((name, len(selected), target))
                take = np.random.RandomState(4200 + role).choice(len(selected), target, replace=False)
                selected = sorted((selected[i] for i in take), key=lambda r: r[0])
            keep_ids[role] = {r[0] for r in selected}
            dest = bank_dir / f"{name}_images.h5"
            with h5py.File(dest.with_suffix(".tmp.h5"), "w") as out:
                data = out.create_dataset("embeddings", (len(selected), 2048), dtype="float16")
                out.create_dataset("ids", data=np.asarray([r[0] for r in selected], dtype=np.int64))
                destination = {r[0]: i for i, r in enumerate(selected)}
                for path in paths:
                    with h5py.File(path) as f:
                        ids, values = f["ids"][:], f["embeddings"][:]
                        pairs = [(destination[int(k)], j) for j, k in enumerate(ids) if int(k) in destination]
                        if pairs:
                            positions, source_rows = map(np.asarray, zip(*pairs))
                            order = np.argsort(positions)
                            data[positions[order]] = values[source_rows[order]]
            dest.with_suffix(".tmp.h5").replace(dest)
            counts[f"{name}_images"] = len(selected)
        texts = []
        for path in paths:
            with path.with_suffix(".jsonl").open() as f:
                texts.extend(json.loads(line) for line in f)
        for role, name, target in (((1, "pairs", None), (3, "unpaired", 1000000)) if split == "train"
                                   else ((4, "cc3m_validation", None),)):
            rows = sorted((r for r in texts if r["role"] == role), key=lambda r: r["source_index"])
            if role in keep_ids:
                rows = [r for r in rows if r["source_index"] in keep_ids[role]]
            counts[f"{name}_text_available"] = len(rows)
            if target is not None:
                if len(rows) < target:
                    raise ValueError((name, len(rows), target))
                if role == 3:
                    take = np.random.RandomState(4200 + role).choice(len(rows), target, replace=False)
                    rows = sorted((rows[i] for i in take), key=lambda r: r["source_index"])
                else:
                    rows = rows[:target]
            write_json(root / "manifests" / f"{name}_text.json", rows)
            counts[f"{name}_texts"] = len(rows)
    write_json(root / "results/data_counts.json", counts)
    print(json.dumps(counts), flush=True)


def text_features(args):
    root = Path(args.root)
    module = upstream_file(args.source, "src/models/language_model.py", "sotalign_text_extractor")
    model = module.SentenceEmbedding(str(root / "models/NV-Embed-v2")).eval()
    for manifest in sorted((root / "manifests").glob("*.json")):
        if manifest.stem.endswith("_text"):
            rows = json.loads(manifest.read_text())
            captions = [r["caption"] for r in rows]
            ids = [r["source_index"] for r in rows]
            name = manifest.stem.removesuffix("_text")
        elif manifest.stem.endswith("_test") or manifest.stem == "coco_val2017":
            rows = json.loads(manifest.read_text())
            captions = [caption for r in rows for caption in r["captions"]]
            ids = [r["id"] for r in rows for _ in r["captions"]]
            name = manifest.stem
        else:
            raise ValueError(manifest)
        indices = np.arange(args.rank, len(captions), args.world)
        path = root / "text_features" / f"{name}_rank{args.rank}.h5"
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        lengths = np.empty(len(indices), dtype=np.int32)
        tokenizer = model.model.tokenizer
        for start in range(0, len(indices), 1024):
            selected = indices[start:start + 1024]
            tokens = tokenizer([captions[i] + tokenizer.eos_token for i in selected],
                               padding=False, truncation=True, max_length=2048)["input_ids"]
            lengths[start:start + len(selected)] = [len(ids_) for ids_ in tokens]
        temp = path.with_suffix(".tmp.h5")
        with h5py.File(temp, "w") as f:
            bank = f.create_dataset("embeddings", (len(indices), 4096), dtype="float16")
            f.create_dataset("indices", data=indices)
            start, batches = 0, 0
            while start < len(indices):
                count, longest = 0, 0
                for length in lengths[start:start + args.batch]:
                    proposed = max(longest, int(length))
                    if count and proposed * (count + 1) > 4096:
                        break
                    count, longest = count + 1, proposed
                selected = indices[start:start + count]
                with torch.inference_mode():
                    vectors = model.get_sentence_embeddings([captions[i] for i in selected])
                if vectors.shape != (len(selected), 4096) or not torch.isfinite(vectors).all():
                    raise ValueError((name, selected.tolist(), vectors.shape))
                bank[start:start + len(selected)] = vectors.cpu().numpy()
                if batches % 100 == 0:
                    print(f"[text] {name} rank={args.rank} rows={start + len(selected)}/{len(indices)} "
                          f"batch={len(selected)} max_tokens={longest} "
                          f"peak_bytes={torch.cuda.max_memory_allocated()}", flush=True)
                start += count
                batches += 1
            f.create_dataset("ids", data=np.asarray([str(ids[i]) for i in indices], dtype=h5py.string_dtype()))
        temp.replace(path)
    write_json(root / "results" / f"text_rank{args.rank}.json",
               {"rank": args.rank, "world": args.world, "peak_gpu_bytes": torch.cuda.max_memory_allocated()})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("cc-images", "benchmark-images", "merge-images", "texts"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    if not 0 <= args.rank < args.world:
        raise ValueError((args.rank, args.world))
    torch.set_num_threads(min(8, int(os.environ["SLURM_CPUS_ON_NODE"])))
    {"cc-images": image_shards, "benchmark-images": benchmark_images,
     "merge-images": merge_images, "texts": text_features}[args.stage](args)


if __name__ == "__main__":
    main()
