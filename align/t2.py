"""T2: does an image-only score predict which frozen CT encoder aligns with radiology reports?

Predictor (no report is read): linear CKA between an encoder's features and the segmentation measurements
(align.measures) on the "sel" patients. Baseline predictor: 5-fold cross-validated ridge R^2 from the features to the
same measurements. Target (reads reports, "train" pairs and "test" patients only): mean zero-shot AUC over the 18
CT-RATE labels after a Procrustes map from k = 100 pairs, 5 seeds; secondary targets FOSCTTM and the ridge-map AUC.
Reference that needs reports: CKA image-report on the test patients.

    python -m align.t2 --emb DIR --out DIR --encoders voxelfm,voxelfm_patch,... --model 4B --layer mid
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from align.align import SEEDS, foscttm, label_names, linear_cka, normalize, polar
from align.baselines import ALPHAS, auc_matrix, zero_shot_scores

K = 100
N_PERM = 100000


def measurement_matrix(path, keys, rows):
    """log1p(ml) of every measurement, z-scored on the given rows; measurements constant on those rows are dropped."""
    m = pd.read_csv(path).set_index("key").loc[keys].to_numpy(float)
    m = np.log1p(m)
    mu, sd = m[rows].mean(0), m[rows].std(0)
    keep = sd > 0
    return (m[:, keep] - mu[keep]) / sd[keep], int(keep.sum())


def ridge_r2(x, m, seed=0):
    """Mean over measurements of the 5-fold cross-validated R^2 of a ridge map from features to measurements."""
    pred = np.zeros_like(m)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(x):
        pred[te] = RidgeCV(alphas=ALPHAS).fit(x[tr], m[tr]).predict(x[te])
    return float(np.mean(1 - ((m - pred) ** 2).sum(0) / ((m - m.mean(0)) ** 2).sum(0)))


def target(x, y, prompts, lab, train, test):
    """Procrustes and ridge maps from K pairs drawn from the train patients (as align.run_cell draws its pairs)."""
    out = {"procrustes_auc": [], "procrustes_foscttm": [], "ridge_auc": []}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(train)
        half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]
        pairs = rng.choice(half_a, K, replace=False)
        mx, my = x[half_a].mean(0), y[half_b].mean(0)
        fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
        xp, yp, xt, yt, pr = fx(x[pairs]), fy(y[pairs]), fx(x[test]), fy(y[test]), fy(prompts)
        w = polar(xp.T @ yp)
        u = normalize(xt @ w)
        out["procrustes_auc"].append(float(np.nanmean(auc_matrix(zero_shot_scores(u, pr), lab[test]))))
        out["procrustes_foscttm"].append(foscttm(u @ yt.T))
        r = RidgeCV(alphas=ALPHAS, fit_intercept=False).fit(xp, yp).coef_.T
        out["ridge_auc"].append(float(np.nanmean(auc_matrix(zero_shot_scores(normalize(xt @ r), pr), lab[test]))))
    return {k: float(np.mean(v)) for k, v in out.items()}


def perm_p(a, b, rng):
    """One-sided permutation p-value of Spearman rho(a, b) > observed under random pairing."""
    obs = spearmanr(a, b)[0]
    null = np.array([spearmanr(a, rng.permutation(b))[0] for _ in range(N_PERM)])
    return float(obs), float((1 + (null >= obs).sum()) / (1 + N_PERM))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoders", required=True)
    ap.add_argument("--model", default="4B")
    ap.add_argument("--layer", default="mid", choices=["last", "mid"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    sel, train, test = (np.flatnonzero(cases.split == s) for s in ("sel", "train", "test"))
    names = label_names(a.emb)
    lab = cases[names].to_numpy(float)
    meas, n_meas = measurement_matrix(f"{a.emb}/measures.csv", cases["key"], sel)
    sfx = {"last": "", "mid": "mid"}[a.layer]
    y = normalize(np.load(f"{a.emb}/text{sfx}_{a.model}.npy").astype(np.float64))
    prompts = normalize(np.load(f"{a.emb}/prompts{sfx}_{a.model}.npy").astype(np.float64))
    if len(y) != len(cases) or len(prompts) != 2 * len(names):
        raise ValueError("text embeddings do not match the case list")
    print(f"[t2] sel={len(sel)} train={len(train)} test={len(test)} measurements={n_meas}", flush=True)
    rows = []
    os.makedirs(f"{a.out}/rows", exist_ok=True)
    for enc in a.encoders.split(","):
        cached = f"{a.out}/rows/{enc}.json"     # one row per encoder, so a requeued run continues
        if os.path.exists(cached):
            rows.append(json.load(open(cached)))
            print(f"[t2] cached {enc}", flush=True)
            continue
        x = normalize(np.stack([np.load(f"{a.emb}/{enc}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64))
        xs = normalize(x[sel] - x[sel].mean(0))
        row = {"encoder": enc, "dim": x.shape[1],
               "pred_cka_measures": linear_cka(xs, meas[sel]),
               "pred_ridge_r2": ridge_r2(xs, meas[sel]),
               "ref_cka_report_test": linear_cka(x[test], y[test]),
               **target(x, y, prompts, lab, train, test)}
        rows.append(row)
        json.dump(row, open(cached + ".tmp", "w"))
        os.replace(cached + ".tmp", cached)
        print(json.dumps(row), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/encoders.csv", index=False)
    rng = np.random.default_rng(0)
    result = {"n_encoders": len(df), "n_sel": len(sel), "n_train": len(train), "n_test": len(test),
              "n_measurements": n_meas, "model": a.model, "layer": a.layer, "k": K}
    for p in ("pred_cka_measures", "pred_ridge_r2", "ref_cka_report_test"):
        for t in ("procrustes_auc", "ridge_auc"):
            rho, pv = perm_p(df[p].to_numpy(), df[t].to_numpy(), rng)
            result[f"rho_{p}__{t}"], result[f"p_{p}__{t}"] = rho, pv
        rho, pv = perm_p(df[p].to_numpy(), -df["procrustes_foscttm"].to_numpy(), rng)
        result[f"rho_{p}__neg_procrustes_foscttm"], result[f"p_{p}__neg_procrustes_foscttm"] = rho, pv
    rho, p = result["rho_pred_cka_measures__procrustes_auc"], result["p_pred_cka_measures__procrustes_auc"]
    result["primary_pass"] = bool(rho >= 0.6 and p < 0.05)
    json.dump(result, open(f"{a.out}/result.json", "w"), indent=1)
    with pd.option_context("display.width", 200):
        print(df.to_string(index=False))
    print(json.dumps(result, indent=1), flush=True)
    print("T2_EXIT rc=0", flush=True)


if __name__ == "__main__":
    main()
