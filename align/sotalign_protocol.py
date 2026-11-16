"""Compare existing few-pair maps with the upstream SOTAlign entry point."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np
import pandas as pd

from align.sotalign_protocol_data import SOURCE_REV, write_json

DATASETS = ("coco_karpathy_test", "flickr30k_karpathy_test", "coco_val2017")
PAIR_COUNTS = (100, 1000, 10000, 100000)
METRICS = tuple(f"{direction}_r{k}" for direction in ("t2i", "i2t") for k in (1, 5, 10)) + ("mean_r1",)


def load_bank(root, name, modality):
    with h5py.File(Path(root) / "banks" / f"{name}_{modality}.h5") as f:
        values = f["embeddings"][:]
        ids = f["ids"][:].astype(str)
    if not np.isfinite(values).all():
        raise ValueError((name, modality, "non-finite features"))
    return values, ids


def merge_texts(args):
    root = Path(args.root)
    manifests = sorted((root / "manifests").glob("*.json"))
    for manifest in manifests:
        rows = json.loads(manifest.read_text())
        if manifest.stem.endswith("_text"):
            name = manifest.stem.removesuffix("_text")
            ids = np.asarray([str(r["source_index"]) for r in rows])
        else:
            name = manifest.stem
            ids = np.asarray([r["id"] for r in rows for _ in r["captions"]])
        dest = root / "banks" / f"{name}_text.h5"
        temp = dest.with_suffix(".tmp.h5")
        seen = np.zeros(len(ids), dtype=bool)
        values = np.empty((len(ids), 4096), dtype=np.float16)
        with h5py.File(temp, "w") as out:
            out.create_dataset("ids", data=np.asarray(ids, dtype=h5py.string_dtype()))
            for rank in range(args.world):
                path = root / "text_features" / f"{name}_rank{rank}.h5"
                with h5py.File(path) as f:
                    indices = f["indices"][:]
                    features = f["embeddings"][:]
                    if seen[indices].any() or not np.array_equal(f["ids"][:].astype(str), ids[indices]):
                        raise ValueError(f"Duplicate or wrong text rows: {path}")
                    if not np.isfinite(features).all():
                        raise ValueError(f"Non-finite text rows: {path}")
                    values[indices] = features
                    seen[indices] = True
            if not seen.all():
                raise ValueError((name, "missing text rows", int((~seen).sum())))
            out.create_dataset("embeddings", data=values)
        temp.replace(dest)
        del values
        if name != "unpaired":
            with h5py.File(root / "banks" / f"{name}_images.h5") as f:
                image_ids = f["ids"][:].astype(str)
            expected = np.repeat(image_ids, 5) if name in DATASETS else image_ids
            if not np.array_equal(ids, expected):
                raise ValueError((name, "image-text identity mismatch"))
    with h5py.File(root / "banks/pairs_images.h5") as f:
        pairs = set(f["ids"][:].astype(str))
    with h5py.File(root / "banks/unpaired_images.h5") as f:
        images = set(f["ids"][:].astype(str))
    with h5py.File(root / "banks/unpaired_text.h5") as f:
        texts = set(f["ids"][:].astype(str))
    if pairs & images or pairs & texts or images & texts:
        raise ValueError("Supervised and unpaired source identities overlap")
    for alias, target in (("text-unpaired.h5", "unpaired_text.h5"), ("image-unpaired.h5", "unpaired_images.h5")):
        link = root / "banks" / alias
        destination = root / "banks" / target
        if link.is_symlink():
            if link.resolve() != destination.resolve():
                raise ValueError(f"Wrong upstream bank alias: {link}")
        elif link.exists():
            raise ValueError(f"Expected a bank alias, found a file: {link}")
        else:
            link.symlink_to(target)
    np.savez(root / "centering.npz", image_mean=mean_unit_rows(root, "images"),
             text_mean=mean_unit_rows(root, "text"))
    write_json(root / "results/banks_verified.json", {"pairs": len(pairs), "unpaired_images": len(images),
               "unpaired_texts": len(texts), "source_ids_disjoint": True,
               "text_model_revision": json.loads((root / "results/setup.json").read_text())["models"]})


def mean_unit_rows(root, modality):
    from align.align import normalize

    with h5py.File(Path(root) / "banks" / f"unpaired_{modality}.h5") as f:
        dataset = f["embeddings"]
        total = np.zeros(dataset.shape[1], dtype=np.float64)
        for start in range(0, len(dataset), 32768):
            total += normalize(dataset[start:start + 32768].astype(np.float64)).sum(axis=0)
        return total / len(dataset)


def retrieval(images, texts, captions_per_image=5):
    from align.align import normalize

    if len(texts) != len(images) * captions_per_image:
        raise ValueError((len(images), len(texts), captions_per_image))
    if not np.isfinite(images).all() or not np.isfinite(texts).all():
        raise ValueError("Non-finite mapped retrieval features")
    images, texts = normalize(images), normalize(texts)
    labels_image = np.arange(len(images))
    labels_text = np.repeat(labels_image, captions_per_image)
    result = {}
    for direction, queries, targets, query_labels, target_labels in (
        ("t2i", texts, images, labels_text, labels_image),
        ("i2t", images, texts, labels_image, labels_text),
    ):
        hits = np.zeros(3, dtype=np.int64)
        for start in range(0, len(queries), 128):
            scores = queries[start:start + 128] @ targets.T
            top = np.argsort(-scores, axis=1, kind="stable")[:, :10]
            matches = target_labels[top] == query_labels[start:start + len(top), None]
            hits += np.asarray([matches[:, :k].any(axis=1).sum() for k in (1, 5, 10)])
        result.update({f"{direction}_r{k}": 100 * float(count) / len(queries)
                       for k, count in zip((1, 5, 10), hits)})
    result["mean_r1"] = (result["t2i_r1"] + result["i2t_r1"]) / 2
    return result


def baselines(args):
    from align.align import normalize, polar
    from align.anchor_followup import ridge
    from align.fewpair import _torch_space

    root = Path(args.root)
    start = time.monotonic()
    means = np.load(root / "centering.npz")
    mx, my = means["image_mean"], means["text_mean"]
    with h5py.File(root / "banks/pairs_images.h5") as f:
        pair_ids = f["ids"][:].astype(str)
        if args.n > len(pair_ids):
            raise ValueError((args.n, len(pair_ids)))
        indices = np.random.RandomState(args.seed).permutation(len(pair_ids))[:args.n]
        order = np.argsort(indices)
        inverse = np.argsort(order)
        bank = f["embeddings"]
        x = bank[:][indices] if args.n > len(pair_ids) // 4 else bank[indices[order]][inverse]
    with h5py.File(root / "banks/pairs_text.h5") as f:
        if not np.array_equal(pair_ids, f["ids"][:].astype(str)):
            raise ValueError("Paired identities differ")
        bank = f["embeddings"]
        y = bank[:][indices] if args.n > len(pair_ids) // 4 else bank[indices[order]][inverse]
    pair_hash = hashlib.sha256(pair_ids[indices].astype("S").tobytes()).hexdigest()
    center_x = lambda v: normalize(normalize(v.astype(np.float64)) - mx)
    center_y = lambda v: normalize(normalize(v.astype(np.float64)) - my)
    xp, yp = center_x(x), center_y(y)
    del x, y
    alpha = None
    if args.method == "lotoalign":
        fit = ridge(xp, yp)
        alpha = float(fit.alpha_)
        f, g = (lambda v: normalize(v @ fit.coef_.T)), normalize
    elif args.method == "procrustes":
        q = polar(xp.T @ yp)
        f, g = (lambda v: normalize(v @ q)), normalize
    elif args.method == "cca":
        f, g = _torch_space("cca", xp, yp, None, None, args.seed)
    else:
        raise ValueError(args.method)
    fit_seconds = time.monotonic() - start
    rows = []
    for dataset in DATASETS:
        xv, _ = load_bank(root, dataset, "images")
        yv, _ = load_bank(root, dataset, "text")
        scores = retrieval(f(center_x(xv)), g(center_y(yv)))
        row = {"dataset": dataset, "method": args.method, "n": args.n, "seed": args.seed,
               "ridge_alpha": alpha, "pair_draw_sha256": pair_hash, "fit_seconds": fit_seconds, **scores}
        rows.append(row)
        print(json.dumps(row), flush=True)
    path = root / "results" / f"{args.method}_n{args.n}_seed{args.seed}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)


def train_sotalign(args):
    root, source = Path(args.root), Path(args.source)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCE_REV:
        raise ValueError((revision, SOURCE_REV))
    run_name = f"sotalign_n{args.n}_seed{args.seed}_alpha{args.alpha:g}"
    run = root / "training" / run_name
    bank = root / "banks"
    audit_path = root / "results" / f"{run_name}_data.json"
    cmd = [sys.executable, str(Path(__file__).with_name("sotalign_protocol_entry.py")),
           "--protocol-source", str(source), "--protocol-audit", str(audit_path),
           "--supervised_text_embedding", str(bank / "pairs_text.h5"),
           "--supervised_image_embedding", str(bank / "pairs_images.h5"),
           "--unsupervised_text_embedding", str(bank / "text-unpaired.h5"),
           "--unsupervised_image_embedding", str(bank / "image-unpaired.h5"),
           "--val_image_embedding", str(bank / "cc3m_validation_images.h5"),
           "--val_text_embedding", str(bank / "cc3m_validation_text.h5"),
           "--unsupervised_index_mode", "disjoint", "--seed", str(args.seed), "--resume", "latest",
           "--batch-size", "32768", "--batch-size-supervised", str(min(args.n, 10000)),
           "--lr", "1e-4", "--optimizer", "lion", "--lr-scheduler", "cosine",
           "--beta1", "0.9", "--beta2", "0.99", "--eps", "1e-8", "--precision", "amp", "--wd", "1e-5",
           "--target-dimension", "1024", "--linear-type", "linear", "--width-factor", "1",
           "--log-every-n-steps", "5", "--name", run_name, "--logs", str(root / "training"),
           "--logit_scale", "20", "--logit_bias", "-10", "--ot", "--semisupervised",
           "--n_supervised_pairs", str(args.n), "--n_unsupervised_image", "1000000",
           "--n_unsupervised_text", "1000000", "--cca_lam_x", "0.1", "--cca_lam_y", "0.1", "--eig_eps", "1e-6",
           "--alpha_semisupervised_sail", "1", "--alpha_semisupervised_klot", str(args.alpha),
           "--epsilon_sinkhorn_student", "0.05", "--n_iters_sinkhorn_student", "100",
           "--epsilon_sinkhorn_teacher", "0.01", "--n_iters_sinkhorn_teacher", "100", "--affinity", "cca",
           "--p_noise", "0", "--p_noise_unsupervised", "0", "--sigma_K", "0",
           "--target-steps", "2000", "--save-frequency", "0", "--save-most-recent"]
    write_json(root / "results" / f"{run_name}_command.json", {"argv": cmd, "upstream_revision": revision})
    best = run / "checkpoints/epoch_best.pt"
    backup = run / "checkpoints/epoch_best_before_resume.pt"
    if not (run / "timing_summary.json").exists():
        if best.exists() and not backup.exists():
            shutil.copy2(best, backup)
        subprocess.run(cmd, check=True, cwd=source)
    import torch

    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    if backup.exists():
        previous = torch.load(backup, map_location="cpu", weights_only=False)
        if previous["best_metric"] > checkpoint["best_metric"]:
            shutil.copy2(backup, best)
            checkpoint = previous
        backup.unlink()
    sys.path.insert(0, str(source / "src"))
    from models.sail_model import AlignmentLayer

    model = AlignmentLayer(2048, 4096, 1024, linear_type="linear", logit_scale=20,
                           logit_bias=-10, width_factor=1).cuda().eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    audit = json.loads(audit_path.read_text())
    pair_hash = audit["pair_draw_sha256"]
    rows = []
    for dataset in DATASETS:
        x, _ = load_bank(root, dataset, "images")
        y, _ = load_bank(root, dataset, "text")
        outputs_x, outputs_y = [], []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for start in range(0, len(x), 1024):
                t = torch.as_tensor(x[start:start + 1024], device="cuda")
                outputs_x.append(model(image_features=t)["image_features"].float().cpu().numpy())
            for start in range(0, len(y), 1024):
                t = torch.as_tensor(y[start:start + 1024], device="cuda")
                outputs_y.append(model(text_features=t)["text_features"].float().cpu().numpy())
        row = {"dataset": dataset, "method": "sotalign", "n": args.n, "seed": args.seed,
               "alpha": args.alpha, "cc3m_val_mean_r1": checkpoint["best_metric"] * 100,
               "checkpoint": str(best), "pair_draw_sha256": pair_hash,
               "step_to_best": checkpoint["step_to_best"],
               **retrieval(np.concatenate(outputs_x), np.concatenate(outputs_y))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    pd.DataFrame(rows).to_csv(root / "results" / f"{run_name}.csv", index=False)
    write_json(root / "results" / f"{run_name}_timing.json", json.loads((run / "timing_summary.json").read_text()))


def summarize(args):
    root = Path(args.root)
    paths = [p for p in (root / "results").glob("*.csv") if p.stem.startswith(("lotoalign_n", "procrustes_n", "cca_n", "sotalign_n"))]
    if not paths:
        raise ValueError("No method results found")
    cells = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    selected = []
    for (n, seed), frame in cells[cells.method == "sotalign"].groupby(["n", "seed"]):
        ranking = frame.groupby("alpha").cc3m_val_mean_r1.first()
        if set(ranking.index) != {1e-3, 1e-4, 1e-5}:
            raise ValueError((n, seed, "incomplete CC3M alpha selection"))
        alpha = ranking.idxmax()
        selected.append(frame[frame.alpha == alpha])
    final = pd.concat([cells[cells.method != "sotalign"], *selected], ignore_index=True)
    expected = {(dataset, method, n) for dataset in DATASETS for method in
                ("lotoalign", "procrustes", "cca", "sotalign") for n in PAIR_COUNTS}
    if set(final.groupby(["dataset", "method", "n"]).groups) != expected:
        raise ValueError("Incomplete method, dataset, or pair-count grid")
    for key, frame in final.groupby(["dataset", "method", "n"]):
        required = set(range(5)) if key[1] != "sotalign" else set(range(args.sotalign_seeds))
        if set(frame.seed) != required or len(frame) != len(required):
            raise ValueError((key, "incomplete or duplicate seed results"))
    for (n, seed), frame in final.groupby(["n", "seed"]):
        if frame.pair_draw_sha256.nunique() != 1:
            raise ValueError((n, seed, "methods used different pair draws"))
    final.to_csv(root / "results/cells.csv", index=False)
    summary = final.groupby(["dataset", "method", "n"]).agg(
        **{f"{metric}_{stat}": (metric, stat) for metric in METRICS for stat in ("mean", "std")},
        seeds=("seed", "nunique")).reset_index()
    summary.to_csv(root / "results/summary.csv", index=False)
    paper = pd.DataFrame([
        {"dataset": "coco_val2017", "n": 10000, "t2i_r1": 26.5, "t2i_r5": 49.8,
         "i2t_r1": 34.1, "i2t_r5": 59.4, "mean_r1": 30.3},
        {"dataset": "flickr30k_karpathy_test", "n": 10000, "t2i_r1": 51.7, "t2i_r5": 79.2,
         "i2t_r1": 60.8, "i2t_r5": 85.7, "mean_r1": (51.7 + 60.8) / 2},
    ])
    paper.to_csv(root / "results/sotalign_paper.csv", index=False)
    check = summary[(summary.method == "sotalign") & (summary.n == 10000) & (summary.dataset == "coco_val2017")]
    if len(check) != 1:
        raise ValueError("Missing 10k-pair SOTAlign rerun")
    gap = float(check.iloc[0].mean_r1_mean) - 30.3
    write_json(root / "results/paper_reproduction_gate.json",
               {"paper_mean_r1": 30.3, "rerun_mean_r1": float(check.iloc[0].mean_r1_mean),
                "difference_pp": gap, "within_2pp": abs(gap) <= 2,
                "source": "https://arxiv.org/html/2602.23353v3#A3.T11"})
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("merge-texts", "baselines", "sotalign", "summary"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--source")
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--method", choices=("lotoalign", "procrustes", "cca"))
    parser.add_argument("--n", type=int, choices=PAIR_COUNTS)
    parser.add_argument("--seed", type=int, choices=range(5))
    parser.add_argument("--alpha", type=float, choices=(1e-3, 1e-4, 1e-5))
    parser.add_argument("--sotalign-seeds", type=int, choices=(1, 5), default=5)
    args = parser.parse_args()
    {"merge-texts": merge_texts, "baselines": baselines,
     "sotalign": train_sotalign, "summary": summarize}[args.stage](args)


if __name__ == "__main__":
    main()
