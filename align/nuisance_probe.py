"""Nuisance-dominated geometry probe: does removing modality-specific nuisance on each side
(no pairs) raise image-report similarity and let geometry-only zero-pair alignment work?

Step 1, oracle diagnostic: unbiased linear CKA between image and report features of the same train patients, per
variant (true pairs only measure, never fit). Step 2, zero pairs: Gromov-Wasserstein on the disjoint train halves,
evaluated on the test patients.

    python -m align.nuisance_probe --emb DIR --out DIR [--encoder voxelfm] [--model 4B]
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from align.align import SEEDS, foscttm, gromov_wasserstein, label_names, linear_cka, normalize, svd
from align.baselines import ALPHAS, auc_matrix, zero_shot_scores

KS = (1, 2, 4, 8, 16)
IMAGE_NUISANCE = [f"ts_total_{i}" for i in [*range(25, 51), *range(69, 90), *range(91, 118)]]


def image_covariates(path, keys):
    m = pd.read_csv(path).set_index("key").loc[keys]
    return np.log1p(m[IMAGE_NUISANCE].to_numpy(float))


def report_covariates(reports):
    return np.array([[np.log1p(len(r)), np.log1p(len(r.split())), np.log1p(len(re.findall(r"[.!?](\s|$)", r))),
                      np.log1p(r.count("\n")), float("IMPRESSION" in r)] for r in reports])


class Cleaner:
    """Fitted on one side's own rows only (covariate standardisation, optional nuisance residualisation, optional
    removal of the top-k PCs); applied to any rows of that side with their raw covariates."""

    def __init__(self, feats, cov, rows, residualise, k):
        self.cmu, csd = cov[rows].mean(0), cov[rows].std(0)
        self.keep = csd > 0
        self.csd = csd[self.keep]
        self.mu = feats[rows].mean(0)
        z = feats[rows] - self.mu
        self.reg = RidgeCV(alphas=ALPHAS).fit(self.std(cov[rows]), z) if residualise else None
        if self.reg is not None:
            z = z - self.reg.predict(self.std(cov[rows]))
        self.top = svd(z, full_matrices=False)[2][:k] if k else None

    def std(self, cov):
        return (cov - self.cmu)[:, self.keep] / self.csd

    def __call__(self, feats, cov):
        z = feats - self.mu
        if self.reg is not None:
            z = z - self.reg.predict(self.std(cov))
        if self.top is not None:
            z = z - (z @ self.top.T) @ self.top
        return normalize(z)


def r2(feats, cov, rows):
    z = feats[rows] - feats[rows].mean(0)
    pred = RidgeCV(alphas=ALPHAS).fit(cov[rows], z).predict(cov[rows])
    return float(1 - ((z - pred) ** 2).sum() / (z ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="voxelfm")
    ap.add_argument("--model", default="4B")
    a = ap.parse_args()
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    names = label_names(a.emb)
    lab = cases[names].to_numpy(float)
    x = normalize(np.stack([np.load(f"{a.emb}/{a.encoder}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64))
    y = normalize(np.load(f"{a.emb}/textmid_{a.model}.npy").astype(np.float64))
    prompts = normalize(np.load(f"{a.emb}/promptsmid_{a.model}.npy").astype(np.float64))
    train, test = (np.flatnonzero(cases.split == s) for s in ("train", "test"))
    cx = image_covariates(f"{a.emb}/measures.csv", cases["key"])
    cy = report_covariates(cases["report"].tolist())
    cp = report_covariates(json.load(open(f"{a.emb}/labels.json"))["prompts"])
    variants = {"raw": (False, 0), "N": (True, 0), **{f"P{k}": (False, k) for k in KS}, **{f"N+P{k}": (True, k) for k in KS}}

    diag = {"nuisance_r2_image": r2(x, cx, train), "nuisance_r2_report": r2(y, cy, train), "cka": {}}
    for name, (res, k) in variants.items():
        fx, fy = Cleaner(x, cx, train, res, k), Cleaner(y, cy, train, res, k)
        diag["cka"][name] = linear_cka(fx(x[train], cx[train]), fy(y[train], cy[train]))
    print(json.dumps(diag, indent=1), flush=True)

    rows = []
    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(train)   # same halves as align.anchor_zero
        half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]
        for name, (res, k) in variants.items():
            fx, fy = Cleaner(x, cx, half_a, res, k), Cleaner(y, cy, half_b, res, k)
            w, gx, gy = gromov_wasserstein(fx(x[half_a], cx[half_a]), fy(y[half_b], cy[half_b]), seed)
            fx, fy = (lambda v, c, f=fx, g=gx: g(f(v, c))), (lambda v, c, f=fy, g=gy: g(f(v, c)))   # the aligner's own centering on top
            u = normalize(fx(x[test], cx[test]) @ w)
            auc = auc_matrix(zero_shot_scores(u, fy(prompts, cp)), lab[test])
            rows.append({"seed": seed, "variant": name, "foscttm": foscttm(u @ fy(y[test], cy[test]).T),
                         "mean_auc": float(np.nanmean(auc))})
            print(json.dumps(rows[-1]), flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/{a.encoder}_cells.csv", index=False)
    summ = df.groupby("variant")[["foscttm", "mean_auc"]].agg(["mean", "std"])
    raw_cka, raw_fos = diag["cka"]["raw"], summ.loc["raw", ("foscttm", "mean")]
    means = summ[("foscttm", "mean")]
    out = {"encoder": a.encoder, "model": a.model, **diag,
           "gw": {v: {"foscttm": float(summ.loc[v, ("foscttm", "mean")]), "foscttm_sd": float(summ.loc[v, ("foscttm", "std")]),
                      "mean_auc": float(summ.loc[v, ("mean_auc", "mean")])} for v in summ.index},
           "signal_cka": bool(raw_cka > 0 and max(v for n, v in diag["cka"].items() if n != "raw") >= 1.5 * raw_cka),
           "signal_gw": bool(((means <= 0.45) & (means <= raw_fos - 0.03)).any())}
    json.dump(out, open(f"{a.out}/{a.encoder}.json", "w"), indent=1)
    print(summ.to_string())
    print("NUISANCE_PROBE_EXIT rc=0", flush=True)


if __name__ == "__main__":
    main()
