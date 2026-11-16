"""T2 follow-up after the CheapCT comparison: is report-free anatomical readability a predictor of few-pair alignability
beyond plain disease-label readability, and does it help to choose an encoder?

Per encoder, on the "sel" patients only (the "test" patients are not read):
- label readability L: mean 5-fold cross-validated AUC of a per-label logistic probe on the 18 CT-RATE labels;
- LogME (You et al. 2021, regression form) of the features for the segmentation measurements, mean over measurements.
Then, with the T2 rows (encoders.csv: pred_ridge_r2 = anatomical readability A, procrustes_auc = target Y):
- Spearman of each predictor with Y; partial Spearman of A with Y given L (ranks residualised on L), permutation p;
- selection regret of each predictor's top pick, and of always choosing voxelfm.

    python -m align.t2_control --emb DIR --rows DIR/encoders.csv --out FILE.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from align.align import label_names, normalize, svd
from align.t2 import measurement_matrix

N_PERM = 100000


def fold_scores(x, y, tr, te):
    return te, LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(x[tr], y[tr]).decision_function(x[te])


def label_readability(x, lab):
    from joblib import Parallel, delayed

    folds = [(j, tr, te) for j in range(lab.shape[1])
             for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(x, lab[:, j])]
    done = Parallel(n_jobs=32)(delayed(fold_scores)(x, lab[:, j], tr, te) for j, tr, te in folds)
    aucs = []
    for j in range(lab.shape[1]):
        y = lab[:, j]
        score = np.zeros(len(y))
        for (jj, _, _), (te, s) in zip(folds, done):
            if jj == j:
                score[te] = s
        r = rankdata(score)
        pos = y.sum()
        aucs.append((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos)))
    return float(np.mean(aucs))


def logme(f, y, iters=100):
    """LogME evidence per sample for one regression target (You et al. 2021, Algorithm 1)."""
    n, d = f.shape
    u, s, _ = svd(f, full_matrices=False)
    s2 = s ** 2
    z = u.T @ y
    z2 = z ** 2
    delta = (y ** 2).sum() - z2.sum()
    alpha, beta = 1.0, 1.0
    for _ in range(iters):
        t = alpha / beta
        gamma = (beta * s2 / (alpha + beta * s2)).sum()
        m2 = (s2 * z2 / (t + s2) ** 2).sum()
        res2 = (z2 / (1 + s2 / t) ** 2).sum() + delta
        a_new, b_new = gamma / (m2 + 1e-9), (n - gamma) / (res2 + 1e-9)
        if abs(a_new / b_new - t) / t < 1e-3:
            alpha, beta = a_new, b_new
            break
        alpha, beta = a_new, b_new
    t = alpha / beta
    m2 = (s2 * z2 / (t + s2) ** 2).sum()
    res2 = (z2 / (1 + s2 / t) ** 2).sum() + delta
    evidence = (n / 2 * np.log(beta) + d / 2 * np.log(alpha) - n / 2 * np.log(2 * np.pi)
                - beta / 2 * res2 - alpha / 2 * m2 - 0.5 * np.log(alpha + beta * s2).sum()
                - 0.5 * (d - len(s)) * np.log(alpha))
    return evidence / n


def partial_spearman(a, y, c, rng):
    ra, ry, rc = rankdata(a), rankdata(y), rankdata(c)
    design = np.c_[np.ones_like(rc), rc]
    res = lambda v: v - design @ np.linalg.lstsq(design, v, rcond=None)[0]
    ea, ey = res(ra), res(ry)
    obs = float(np.corrcoef(ea, ey)[0, 1])
    null = np.array([np.corrcoef(rng.permutation(ea), ey)[0, 1] for _ in range(N_PERM)])
    return obs, float((1 + (null >= obs).sum()) / (1 + N_PERM))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    sel = np.flatnonzero(cases.split == "sel")
    lab = cases[label_names(a.emb)].to_numpy(int)[sel]
    meas, _ = measurement_matrix(f"{a.emb}/measures.csv", cases["key"], sel)
    meas = meas[sel]
    df = pd.read_csv(a.rows).set_index("encoder")
    cache_dir = os.path.splitext(a.out)[0] + "_rows"     # one row per encoder, so a requeued run continues
    os.makedirs(cache_dir, exist_ok=True)
    for enc in df.index:
        cached = f"{cache_dir}/{enc}.json"
        if os.path.exists(cached):
            c = json.load(open(cached))
            df.loc[enc, "label_readability"], df.loc[enc, "logme_measures"] = c["label_readability"], c["logme_measures"]
            continue
        x = normalize(np.stack([np.load(f"{a.emb}/{enc}/{os.path.basename(k)}.npy") for k in cases["key"].iloc[sel]]).astype(np.float64))
        x = normalize(x - x.mean(0))
        df.loc[enc, "label_readability"] = label_readability(x, lab)
        df.loc[enc, "logme_measures"] = float(np.mean([logme(x, meas[:, j]) for j in range(meas.shape[1])]))
        json.dump({k: float(df.loc[enc, k]) for k in ("label_readability", "logme_measures")}, open(cached + ".tmp", "w"))
        os.replace(cached + ".tmp", cached)
        print(f"[t2-control] {enc} L={df.loc[enc, 'label_readability']:.4f} logme={df.loc[enc, 'logme_measures']:.4f}", flush=True)
    rng = np.random.default_rng(0)
    y = df["procrustes_auc"].to_numpy()
    out = {"n_encoders": len(df)}
    for p in ("pred_ridge_r2", "pred_cka_measures", "label_readability", "logme_measures", "ref_cka_report_test"):
        out[f"spearman_{p}"] = float(spearmanr(df[p], y)[0])
        pick = df[p].idxmax()
        out[f"regret_{p}"] = {"pick": pick, "regret": float(y.max() - df.loc[pick, "procrustes_auc"])}
    out["regret_always_voxelfm"] = float(y.max() - df.loc["voxelfm", "procrustes_auc"])
    out["spearman_ridge_r2_vs_label_readability"] = float(spearmanr(df["pred_ridge_r2"], df["label_readability"])[0])
    for p in ("pred_ridge_r2", "pred_cka_measures"):
        rho, pv = partial_spearman(df[p].to_numpy(), y, df["label_readability"].to_numpy(), rng)
        out[f"partial_{p}_given_label_readability"] = {"rho": rho, "p_one_sided": pv}
    rho, pv = partial_spearman(df["label_readability"].to_numpy(), y, df["pred_ridge_r2"].to_numpy(), rng)
    out["partial_label_readability_given_ridge_r2"] = {"rho": rho, "p_one_sided": pv}
    df.to_csv(os.path.splitext(a.out)[0] + "_encoders.csv")
    json.dump(out, open(a.out, "w"), indent=1)
    with pd.option_context("display.width", 200):
        print(df[["pred_ridge_r2", "pred_cka_measures", "label_readability", "logme_measures", "procrustes_auc"]].to_string())
    print(json.dumps(out, indent=1), flush=True)
    print("T2_CONTROL_EXIT rc=0", flush=True)


if __name__ == "__main__":
    main()
