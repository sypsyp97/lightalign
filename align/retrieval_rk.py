"""Retrieval R@1/5/10 on the cells of align.domain_scaling.

Same cases, train/test split, train-mean centering, seeds and ridge maps as
align.domain_scaling.run. n is align.domain_scaling.NS. That tuple is
(100, 400, 1600) and does not contain 0, so this module emits no n = 0 row
unless NS gains one.

Cosine similarity is the matrix align.domain_scaling passes to FOSCTTM:
row-normalized mapped test images times row-normalized test texts. The true
match is the paired case row, the diagonal of that square matrix
(align.align.foscttm). cases.jsonl stores one text per item, and the text
matrix has one row per case in that order:

- COCO: first caption in annotation order (align.coco.build_cases)
- Flickr30k: first caption (align.domains.cases_flickr)
- Clotho: caption_1 (align.domains.cases_clotho)
- IU X-ray: one report per study (align.domains.cases_iuxray)

A caption that was not stored is neither a match nor a candidate. Image-to-text
ranks texts for each image. Text-to-image ranks images for each text. With
unique scores this is the usual top-k hit. Ties use the expected hit of
align.baselines.recall_at.

    python -m align.retrieval_rk run --emb DIR --encoders E1,E2 --texts T1,T2 --out DIR [--check CELLS.csv]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from align.align import foscttm, normalize
from align.anchor_followup import ridge
from align.baselines import recall_at
from align.domain_scaling import NS, SEEDS

KS = (1, 5, 10)
FOSCTTM_ATOL = 1e-9
COLS = ("data", "encoder", "text", "seed", "n", "foscttm",
        "i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10")


def assert_same_foscttm(df, path):
    """Each output cell's foscttm equals the stored domain_scaling cell, within FOSCTTM_ATOL."""
    ref = pd.read_csv(path)
    keys = ["data", "encoder", "text", "seed", "n"]
    missing = [c for c in keys + ["foscttm"] if c not in ref.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    if ref.duplicated(keys).any():
        raise ValueError(f"{path} has duplicate cells")
    merged = df.merge(ref[keys + ["foscttm"]], on=keys, how="left", suffixes=("", "_ref"))
    if merged["foscttm_ref"].isna().any():
        n_missing = int(merged["foscttm_ref"].isna().sum())
        raise ValueError(f"{path} has no foscttm for {n_missing} cells")
    delta = np.abs(merged["foscttm"].to_numpy() - merged["foscttm_ref"].to_numpy())
    worst = int(np.nanargmax(delta))
    assert np.isfinite(delta).all() and float(delta[worst]) <= FOSCTTM_ATOL, (
        f"foscttm differs from {path} by {delta[worst]} at "
        + ", ".join(f"{k}={merged.iloc[worst][k]}" for k in keys))


def run(a):
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    train, test = (np.flatnonzero(cases.split == s) for s in ("train", "test"))
    rows = []
    for t in a.texts.split(","):
        y = normalize(np.load(f"{a.emb}/{t}.npy").astype(np.float64))
        if len(y) != len(cases):
            raise ValueError(f"{t}: {len(y)} rows for {len(cases)} cases")
        my = y[train].mean(0)
        fy = lambda v: normalize(v - my)
        yt = fy(y[test])
        for e in a.encoders.split(","):
            x = normalize(np.stack([np.load(f"{a.emb}/{e}/{os.path.basename(k)}.npy") for k in cases["key"]])
                          .astype(np.float64))
            mx = x[train].mean(0)
            fx = lambda v: normalize(v - mx)
            xt = fx(x[test])
            for seed in SEEDS:
                perm = np.random.default_rng(seed).permutation(len(train))
                for n in NS:
                    idx = train[perm[:n]]
                    u = xt @ ridge(fx(x[idx]), fy(y[idx])).coef_.T
                    sim = normalize(u) @ yt.T
                    i2t = recall_at(sim, KS)
                    t2i = recall_at(sim.T, KS)
                    rows.append({"data": os.path.basename(a.emb.rstrip("/")), "encoder": e, "text": t,
                                 "seed": seed, "n": n, "foscttm": foscttm(sim),
                                 "i2t_r1": i2t["r@1"], "i2t_r5": i2t["r@5"], "i2t_r10": i2t["r@10"],
                                 "t2i_r1": t2i["r@1"], "t2i_r5": t2i["r@5"], "t2i_r10": t2i["r@10"]})
            print(json.dumps(rows[-1]), flush=True)
    if not rows:
        raise ValueError("no cells")
    df = pd.DataFrame(rows)[list(COLS)]
    if a.check is not None:
        if not os.path.isfile(a.check):
            raise FileNotFoundError(a.check)
        assert_same_foscttm(df, a.check)
    os.makedirs(a.out, exist_ok=True)
    tag = os.path.basename(a.emb.rstrip("/"))
    df.to_csv(f"{a.out}/{tag}_rk.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run"])
    ap.add_argument("--emb", required=True)
    ap.add_argument("--encoders", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check")
    a = ap.parse_args()
    run(a)


if __name__ == "__main__":
    main()
