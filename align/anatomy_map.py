"""Anatomy-subspace map: few-pair image -> report map estimated inside the feature directions that predict anatomy.

The subspace comes from images and their segmentation measurements only (the "sel" patients, no report): a ridge map
B from the image features to the measurements, and the top-r left singular vectors of B. The image -> report map is a
ridge regression from the r projected coordinates, fitted on the k known pairs. Controls on the same pairs: ridge and
Procrustes on the full features, ridge inside the top-r principal components of the same "sel" images (same
dimension reduction without the measurements), and a per-label logistic head trained on the report labels.

dev: pairs and evaluation both inside the "train" patients (pairs from half A, evaluation on half B), r swept.
test: one configuration, pairs as in align.t2 / align.run_cell, evaluation on the "test" patients, paired bootstrap.

    python -m align.anatomy_map dev  --emb DIR --out DIR [--encoder voxelfm] [--model 4B]
    python -m align.anatomy_map test --emb DIR --out DIR --rank R [--encoder voxelfm] [--model 4B]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV

from align.align import SEEDS, foscttm, label_names, normalize, polar, svd
from align.baselines import ALPHAS, N_BOOT, auc_matrix, zero_shot_scores
from align.t2 import K, measurement_matrix

RANKS = (8, 16, 32, 64, 123)


def load(emb, encoder, model):
    cases = pd.read_json(f"{emb}/cases.jsonl", lines=True)
    names = label_names(emb)
    x = normalize(np.stack([np.load(f"{emb}/{encoder}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64))
    y = normalize(np.load(f"{emb}/textmid_{model}.npy").astype(np.float64))
    prompts = normalize(np.load(f"{emb}/promptsmid_{model}.npy").astype(np.float64))
    sel = np.flatnonzero(cases.split == "sel")
    meas, _ = measurement_matrix(f"{emb}/measures.csv", cases["key"], sel)
    return cases, x, y, prompts, cases[names].to_numpy(float), meas


def subspaces(xs, ms, ranks):
    """Image-only bases on the "sel" rows (already centred): anatomy = top singular vectors of the ridge map to the
    measurements; pca = top principal directions of the same rows."""
    b = RidgeCV(alphas=ALPHAS, fit_intercept=False).fit(xs, ms).coef_.T          # (d, m)
    ua = svd(b, full_matrices=False)[0]
    up = svd(xs, full_matrices=False)[2].T
    return {r: {"anatomy": ua[:, :r], "pca": up[:, :r]} for r in ranks}


def fit_maps(xp, yp, lab_p, bases):
    """Every method as a function from centred, normalised image rows to report-space scores."""
    out = {"procrustes": (lambda w: lambda v: normalize(v @ w))(polar(xp.T @ yp)),
           "ridge": (lambda w: lambda v: normalize(v @ w))(RidgeCV(alphas=ALPHAS, fit_intercept=False).fit(xp, yp).coef_.T)}
    for r, bs in bases.items():
        for kind, u in bs.items():
            w = RidgeCV(alphas=ALPHAS, fit_intercept=False).fit(xp @ u, yp).coef_.T
            out[f"{kind}_r{r}"] = (lambda u, w: lambda v: normalize(v @ u @ w))(u, w)
    heads = []
    for j in range(lab_p.shape[1]):
        heads.append(LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(xp, lab_p[:, j])
                     if 0 < lab_p[:, j].sum() < len(lab_p) else None)
    return out, heads


def scores(maps, heads, xe, ye, pr):
    res = {}
    for name, f in maps.items():
        u = f(xe)
        res[name] = (zero_shot_scores(u, pr), foscttm(u @ ye.T))
    res["head_C1"] = (np.stack([h.decision_function(xe) if h is not None else np.zeros(len(xe)) for h in heads], 1), np.nan)
    return res


def run(a, mode):
    cases, x, y, prompts, lab, meas = load(a.emb, a.encoder, a.model)
    sel, train, test = (np.flatnonzero(cases.split == s) for s in ("sel", "train", "test"))
    ranks = RANKS if mode == "dev" else (a.rank,)
    rows, kept = [], {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(train)
        half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]
        pairs = rng.choice(half_a, K, replace=False)
        ev = half_b if mode == "dev" else test
        mx, my = x[half_a].mean(0), y[half_b].mean(0)
        fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
        bases = subspaces(fx(x[sel]), meas[sel], ranks)
        maps, heads = fit_maps(fx(x[pairs]), fy(y[pairs]), lab[pairs], bases)
        res = scores(maps, heads, fx(x[ev]), fy(y[ev]), fy(prompts))
        for name, (s, fos) in res.items():
            auc = auc_matrix(s, lab[ev])
            rows.append({"mode": mode, "seed": seed, "method": name, "mean_auc": float(np.nanmean(auc)), "foscttm": fos})
            kept.setdefault(name, []).append(s)
        print(f"[anatomy-map] {mode} seed={seed} " + " ".join(f"{r['method']}={r['mean_auc']:.3f}" for r in rows[-len(res):]), flush=True)
    df = pd.DataFrame(rows)
    summ = df.groupby("method").agg(mean_auc=("mean_auc", "mean"), sd=("mean_auc", "std"), foscttm=("foscttm", "mean"))
    summ = summ.sort_values("mean_auc", ascending=False)
    out = {"mode": mode, "encoder": a.encoder, "model": a.model, "k": K, "summary": summ.reset_index().to_dict("records")}
    if mode == "test":
        ours = f"anatomy_r{a.rank}"
        ev_lab = lab[test]
        boot = np.random.default_rng(12345).integers(0, len(test), (N_BOOT, len(test)))
        for ref in ("ridge", "head_C1", f"pca_r{a.rank}", "procrustes"):
            point = np.mean([np.nanmean(auc_matrix(o, ev_lab)) - np.nanmean(auc_matrix(r, ev_lab))
                             for o, r in zip(kept[ours], kept[ref])])
            d = np.mean([[np.nanmean(auc_matrix(o[b], ev_lab[b])) - np.nanmean(auc_matrix(r[b], ev_lab[b])) for b in boot]
                         for o, r in zip(kept[ours], kept[ref])], axis=0)
            out[f"{ours}_minus_{ref}"] = {"point": float(point), "lo": float(np.quantile(d, 0.025)),
                                          "hi": float(np.quantile(d, 0.975))}
        out["pass"] = out[f"{ours}_minus_ridge"]["lo"] > 0
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/{mode}_{a.encoder}_cells.csv", index=False)
    json.dump(out, open(f"{a.out}/{mode}_{a.encoder}.json", "w"), indent=1)
    with pd.option_context("display.width", 200):
        print(summ.to_string())
    print(json.dumps({k: v for k, v in out.items() if k != "summary"}, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dev", "test"])
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="voxelfm")
    ap.add_argument("--model", default="4B")
    ap.add_argument("--rank", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "test" and a.rank not in RANKS:
        raise ValueError(f"--rank must be one of the dev ranks {RANKS}")
    run(a, a.mode)


if __name__ == "__main__":
    main()
