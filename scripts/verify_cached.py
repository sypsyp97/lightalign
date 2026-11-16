"""Fit the released weights from cached embeddings and check the chest-unit table.

The cache layout is the one read by align.official_ablation.run and align.fewpair.official:
each of --train and --valid has cases.jsonl, labels.json, and voxelfm_b11/<basename(key)>.npy.
The training directory also has textmid_9B.npy and promptsmid_9B.npy. The prompt file name is
the text name with the first "text" replaced by "prompts".

Ridge, report classifiers, AUROC, and the CT-CLIP ROC-corner F1 and accuracy are the
lightalign.align functions. Written weights are loaded with load_release and scored with
score_feature, the same calls lightalign-predict uses.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ENCODER = "voxelfm_b11"
TEXT_NAME = "textmid_9B"
AUROC_TOL = 5e-4
RELOAD_TOL = 1e-9
EXPECTED_AUROC = {50: 0.687, 100: 0.727, 200: 0.754, 400: 0.782, 1600: 0.809, 5994: 0.824}
EXPECTED_F1 = {50: 0.685, 100: 0.709, 200: 0.726, 400: 0.750, 1600: 0.769, 5994: 0.777}
EXPECTED_ACC = {50: 0.642, 100: 0.670, 200: 0.690, 400: 0.718, 1600: 0.740, 5994: 0.750}


def cached_image_path(emb, key):
    """align.official_ablation.run: {emb}/{encoder}/{basename(key)}.npy."""
    return os.path.join(emb, ENCODER, os.path.basename(str(key)) + ".npy")


def read_label_names(emb):
    """align.align.label_names: labels.json['labels'] when the file is present."""
    path = os.path.join(emb, "labels.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    if "labels" not in meta:
        raise KeyError(f"{path} has no labels")
    names = meta["labels"]
    if len(names) == 0:
        raise ValueError(f"{path} has no finding names")
    return names


def stage_split(src, dst, link_text):
    """Expose one cache directory in the layout fit_release and reproduction_table read."""
    cases_path = os.path.join(src, "cases.jsonl")
    labels_path = os.path.join(src, "labels.json")
    if not os.path.isfile(cases_path):
        raise FileNotFoundError(cases_path)
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(labels_path)
    import pandas as pd

    cases = pd.read_json(cases_path, lines=True)
    if "key" not in cases.columns:
        raise KeyError(f"{cases_path} has no key column")
    os.makedirs(os.path.join(dst, "image"), exist_ok=True)
    os.symlink(os.path.abspath(cases_path), os.path.join(dst, "cases.jsonl"))
    os.symlink(os.path.abspath(labels_path), os.path.join(dst, "labels.json"))
    seen = set()
    for key in cases["key"].tolist():
        base = os.path.basename(str(key))
        if base in seen:
            raise ValueError(f"two cases share the volume basename {base}")
        seen.add(base)
        src_npy = cached_image_path(src, key)
        if not os.path.isfile(src_npy):
            raise FileNotFoundError(src_npy)
        os.symlink(os.path.abspath(src_npy), os.path.join(dst, "image", base + ".npy"))
    if link_text:
        text_path = os.path.join(src, TEXT_NAME + ".npy")
        # align.fewpair._load: replace the first "text" in the text file name.
        prompt_stem = TEXT_NAME.replace("text", "prompts", 1)
        if prompt_stem == TEXT_NAME:
            raise RuntimeError(f"prompt file name is unchanged for {TEXT_NAME}")
        prompts_path = os.path.join(src, prompt_stem + ".npy")
        if not os.path.isfile(text_path):
            raise FileNotFoundError(text_path)
        if not os.path.isfile(prompts_path):
            raise FileNotFoundError(prompts_path)
        os.symlink(os.path.abspath(text_path), os.path.join(dst, "text.npy"))
        os.symlink(os.path.abspath(prompts_path), os.path.join(dst, "prompts.npy"))
    return cases


def rounded_3(value):
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite metric {value!r}")
    return format(value, ".3f")


def index_rows(rows, reader):
    found = {}
    for row in rows:
        if row["reader"] != reader:
            continue
        n = row["n"]
        if n in found:
            raise RuntimeError(f"duplicate {reader} row for n={n}")
        found[n] = row
    return found


def check_table(rows, failures):
    for n, exp in EXPECTED_AUROC.items():
        row = rows[n]
        got = row["auroc"]
        if got != got or abs(got - exp) > AUROC_TOL:
            failures.append(
                f"mismatch TC n={n} auroc got {got:.10f} expected {exp:.3f} "
                f"abs diff {abs(got - exp):.10f} tol {AUROC_TOL}"
            )
        else:
            print(f"TC n={n} auroc {got:.6f} expected {exp:.3f} OK", flush=True)
        f1 = rounded_3(row["f1_weighted"])
        acc = rounded_3(row["accuracy"])
        exp_f1 = format(EXPECTED_F1[n], ".3f")
        exp_acc = format(EXPECTED_ACC[n], ".3f")
        if f1 != exp_f1 or acc != exp_acc:
            failures.append(
                f"mismatch TC n={n} f1/acc got {row['f1_weighted']:.10f}/{row['accuracy']:.10f} "
                f"rounded {f1}/{acc} expected {exp_f1}/{exp_acc}"
            )
        else:
            print(f"TC n={n} f1/acc {f1}/{acc} OK", flush=True)


def reload_auroc(valid_cases, cache_dir, out_dir, reference, failures):
    import numpy as np

    from lightalign.align import auc_matrix, chest_mask, label_matrix, load_release, score_feature

    weights, _meta = load_release(out_dir)
    mask = chest_mask(valid_cases)
    vectors = []
    for key in valid_cases["key"].tolist():
        path = cached_image_path(cache_dir, key)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        vectors.append(np.load(path))
    raw = np.stack(vectors).astype(np.float64)
    if raw.ndim != 2 or len(raw) != len(valid_cases) or not np.isfinite(raw).all():
        raise ValueError(f"bad cached image matrix shape={raw.shape}")
    labels = label_matrix(valid_cases)[mask]
    scored = np.stack([score_feature(v, weights)[0] for v in raw[mask]])
    got = float(np.nanmean(auc_matrix(scored, labels)))
    if got != got or abs(got - reference) > RELOAD_TOL:
        failures.append(
            f"mismatch reloaded TC auroc got {got:.12f} table {reference:.12f} "
            f"abs diff {abs(got - reference):.3e} tol {RELOAD_TOL}"
        )
    else:
        print(f"reloaded weights TC auroc {got:.10f} matches n=5994 table OK", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Fit the n=5994 seed-0 release from cached CT-RATE embeddings and check "
            "chest-unit mean AUROC, weighted F1, and accuracy."
        ),
    )
    parser.add_argument("--train", required=True, help="cached training directory")
    parser.add_argument("--valid", required=True, help="cached validation directory (every reconstruction)")
    parser.add_argument("--out", required=True, help="directory for the release .npz and .json")
    args = parser.parse_args(argv)

    from lightalign.align import (
        PAPER_NS,
        RELEASE_N,
        RELEASE_SEED,
        RELEASE_STEM,
        fit_release,
        reproduction_table,
    )

    if tuple(PAPER_NS) != (50, 100, 200, 400, 1600, 5994):
        raise RuntimeError(f"PAPER_NS is {tuple(PAPER_NS)}")
    if RELEASE_N != 5994 or RELEASE_SEED != 0:
        raise RuntimeError(f"release is n={RELEASE_N} seed={RELEASE_SEED}")
    if set(EXPECTED_AUROC) != set(PAPER_NS) or set(EXPECTED_F1) != set(PAPER_NS) or set(EXPECTED_ACC) != set(PAPER_NS):
        raise RuntimeError("expected tables do not cover PAPER_NS")
    train_names = read_label_names(args.train)
    valid_names = read_label_names(args.valid)
    if list(train_names) != list(valid_names):
        raise ValueError("train and valid finding names differ")

    os.makedirs(args.out, exist_ok=True)
    npz_path = os.path.join(args.out, RELEASE_STEM + ".npz")
    stage_root = tempfile.mkdtemp(prefix="lightalign-cache-")
    try:
        train_stage = os.path.join(stage_root, "train")
        valid_stage = os.path.join(stage_root, "valid")
        stage_split(args.train, train_stage, link_text=True)
        valid_cases = stage_split(args.valid, valid_stage, link_text=False)
        fit_release(train_stage, npz_path)
        rows = reproduction_table(train_stage, valid_stage)
    finally:
        shutil.rmtree(stage_root)

    header = f"{'n':>6} {'seeds':>5} {'volumes':>8} {'reader':<6} {'auroc':>8} {'f1_weighted':>12} {'accuracy':>9}"
    print(header, flush=True)
    for row in rows:
        print(
            f"{row['n']:6d} {row['seeds']:5d} {row['volumes']:8d} {row['reader']:<6} "
            f"{row['auroc']:8.4f} {row['f1_weighted']:12.4f} {row['accuracy']:9.4f}",
            flush=True,
        )
    failures = []
    tc = index_rows(rows, "TC")
    prompt = index_rows(rows, "P")
    for n in PAPER_NS:
        if n not in tc or n not in prompt:
            raise RuntimeError(f"missing reader row for n={n}")
        for key in ("auroc", "f1_weighted", "accuracy"):
            value = prompt[n][key]
            if value != value or value in (float("inf"), float("-inf")):
                failures.append(f"non-finite P n={n} {key}={value!r}")
    check_table(tc, failures)
    reload_auroc(valid_cases, os.path.abspath(args.valid), args.out, tc[RELEASE_N]["auroc"], failures)
    if failures:
        for line in failures:
            print(line, flush=True)
        return 1
    print("VERIFY_CACHED_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
