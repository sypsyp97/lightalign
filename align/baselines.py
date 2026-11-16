"""Same-budget baselines for the few-pair language readout (CT-RATE).

T1: on the exact pairs of align.run_cell (same seed, same draw), compare
    - procrustes: orthogonal map, zero-shot readout with the "present / No x." prompts,
    - ridge: linear map image -> report embedding (RidgeCV, leave-one-out alpha), same zero-shot readout,
    - label head: one logistic regression per label on the image features of the k pairs, trained on the report
      labels of those pairs (C grid reported in full; best-of-grid favours the head).
    Mean AUC over labels, per-label AUC, FOSCTTM and Recall@1/5/10 (maps only), and a paired bootstrap over test
    patients of the mean-AUC difference (procrustes minus each competitor), averaged over seeds.
T3: logistic probe per label on all training cases (upper reference for what the frozen features hold).

    python -m align.baselines --emb DIR --out DIR --models 0.8B,4B,27B --encoder voxelfm --layer mid
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, RidgeCV

from align.align import SEEDS, foscttm, label_names, load, normalize, polar

KS = (10, 100, 1000)
CS = (0.1, 1.0, 10.0, 100.0)
ALPHAS = np.logspace(-3, 3, 13)
N_BOOT = 1000


def auc_matrix(scores, y):
    """AUC per column via tie-averaged ranks; scores and y are (n, L). Columns with a single class give nan."""
    n = len(y)
    ranks = rankdata(scores, axis=0)
    pos = y.sum(0)
    neg = n - pos
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = ((ranks * y).sum(0) - pos * (pos + 1) / 2) / (pos * neg)
    return np.where((pos > 0) & (neg > 0), auc, np.nan)


def recall_at(sim, ks=(1, 5, 10)):
    """Image -> report retrieval: fraction of test images whose own report is in the top k. Ties with the own report
    count as a random order among the tied candidates (expected hit rate)."""
    true = np.diag(sim)[:, None]
    above = (sim > true).sum(1)
    tied = (sim == true).sum(1)
    return {f"r@{k}": float(np.clip((k - above) / tied, 0, 1).mean()) for k in ks}


def zero_shot_scores(u, prompts):
    """Prompt reader: cosine of each mapped item with the positive prompt minus the negative one. The rows of u are
    unit-normed here, so a map whose output length varies by item (ridge) does not reorder items by their length."""
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    return u @ prompts[0::2].T - u @ prompts[1::2].T


def split(cases, seed, k):
    """The draw of align.run_cell for a procrustes cell: permutation of train, then the k pairs from half A."""
    train = np.flatnonzero(cases.split == "train")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train)
    half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]
    k_used = min(k, len(half_a))
    pairs = rng.choice(half_a, k_used, replace=False)
    return half_a, half_b, pairs


def logistic_scores(xtr, ytr, xte, c):
    """A label with one class among the training pairs gets a constant score (AUC 0.5): the head cannot rank it."""
    out = np.zeros((len(xte), ytr.shape[1]))
    for j in range(ytr.shape[1]):
        if 0 < ytr[:, j].sum() < len(ytr):
            m = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(xtr, ytr[:, j])
            out[:, j] = m.decision_function(xte)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="0.8B,4B,27B")
    ap.add_argument("--encoder", default="voxelfm")
    ap.add_argument("--layer", default="mid", choices=["last", "mid"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    models = tuple(a.models.split(","))
    names = label_names(a.emb)
    cases, x, ys, ps = load(a.emb, models, a.encoder, a.layer)
    lab = cases[names].to_numpy().astype(float)
    test = np.flatnonzero(cases.split == "test")
    train = np.flatnonzero(cases.split == "train")
    yt_lab = lab[test]
    rng_boot = np.random.default_rng(12345)
    boot = rng_boot.integers(0, len(test), (N_BOOT, len(test)))
    rows, per_label = [], []

    # T3: all training labels, image features centred on the training mean as in run_cell.
    mx = x[train].mean(0)
    xtr_all, xte_all = normalize(x[train] - mx), normalize(x[test] - mx)
    for c in CS:
        auc = auc_matrix(logistic_scores(xtr_all, lab[train], xte_all, c), yt_lab)
        rows.append({"test": "T3", "method": f"probe_all_C{c}", "model": "-", "k": len(train), "seed": -1,
                     "mean_auc": float(np.nanmean(auc))})
        per_label.append({"test": "T3", "method": f"probe_all_C{c}", "k": len(train), **dict(zip(names, auc.tolist()))})
        print(json.dumps(rows[-1]), flush=True)

    # T1: per seed and k, the same pairs for every method.
    boot_diff, point_diff = {}, {}
    for k in KS:
        for seed in SEEDS:
            half_a, half_b, pairs = split(cases, seed, k)
            mxa = x[half_a].mean(0)
            fx = lambda v: normalize(v - mxa)
            xp, xt = fx(x[pairs]), fx(x[test])
            head = {}
            for c in CS:
                s = logistic_scores(xp, lab[pairs], xt, c)
                head[f"head_C{c}"] = s
            for model in models:
                y, p = ys[model], ps[model]
                myb = y[half_b].mean(0)
                fy = lambda v: normalize(v - myb)
                yp, yt, pr = fy(y[pairs]), fy(y[test]), fy(p)
                scores = dict(head)
                maps = {"procrustes": polar(xp.T @ yp),
                        "ridge": RidgeCV(alphas=ALPHAS, fit_intercept=False).fit(xp, yp).coef_.T}
                for name, w in maps.items():
                    u = normalize(xt @ w)
                    scores[name] = zero_shot_scores(u, pr)
                    sim = u @ yt.T
                    extra = {"foscttm": foscttm(sim), **recall_at(sim)}
                    scores[name + "_extra"] = extra
                for name, s in scores.items():
                    if name.endswith("_extra"):
                        continue
                    auc = auc_matrix(s, yt_lab)
                    rows.append({"test": "T1", "method": name, "model": model, "k": k, "k_used": len(pairs), "seed": seed,
                                 "mean_auc": float(np.nanmean(auc)), **scores.get(name + "_extra", {})})
                    per_label.append({"test": "T1", "method": name, "model": model, "k": k, "seed": seed,
                                      **dict(zip(names, auc.tolist()))})
                    print(json.dumps(rows[-1]), flush=True)
                ref = scores["procrustes"]
                for name, s in scores.items():
                    if name.endswith("_extra") or name == "procrustes":
                        continue
                    d = np.array([np.nanmean(auc_matrix(ref[b], yt_lab[b])) - np.nanmean(auc_matrix(s[b], yt_lab[b]))
                                  for b in boot])
                    boot_diff.setdefault((model, k, name), []).append(d)
                    point_diff.setdefault((model, k, name), []).append(
                        np.nanmean(auc_matrix(ref, yt_lab)) - np.nanmean(auc_matrix(s, yt_lab)))

    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/cells.csv", index=False)
    pd.DataFrame(per_label).to_csv(f"{a.out}/per_label.csv", index=False)
    summ = df.groupby(["test", "method", "model", "k"], dropna=False).agg(
        mean_auc=("mean_auc", "mean"), sd_auc=("mean_auc", "std"),
        foscttm=("foscttm", "mean"), r1=("r@1", "mean"), r5=("r@5", "mean"), r10=("r@10", "mean")).reset_index()
    summ.to_csv(f"{a.out}/summary.csv", index=False)
    ci = []
    for (model, k, name), ds in boot_diff.items():
        d = np.mean(ds, axis=0)          # seed-averaged difference per bootstrap resample of test patients
        ci.append({"model": model, "k": k, "vs": name, "procrustes_minus": float(np.mean(point_diff[(model, k, name)])),
                   "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975))})
    pd.DataFrame(ci).to_csv(f"{a.out}/paired_ci.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(summ.to_string(index=False))
        print(pd.DataFrame(ci).to_string(index=False))
    print("BASELINES_EXIT rc=0", flush=True)


if __name__ == "__main__":
    main()
