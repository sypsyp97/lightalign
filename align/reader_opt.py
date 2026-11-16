"""Reader optimisation on held-out patients: the report classifier TC as in align.decisive readout
(base) against two variants, one change each.
  A: the regularisation of each finding's logistic regression chosen by 5-fold cross-validation on the train reports
     (and solved to convergence).
  B: base logistic regression, fitted and applied only in the k report-side directions of the ridge map (top-k right
     singular vectors of W), k = the number of shared components the spectrum estimate gives for this encoder and text
     model (align.spiked / align.textlm estimates, d = 64).
The classifiers see reports only, so base and A are fitted once per text model and shared by all encoders.

    python -m align.reader_opt run --emb DIR --spiked DIR --out DIR --model M
    python -m align.reader_opt summary --out DIR
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from align.align import SEEDS, label_names, normalize, svd
from align.anchor_followup import ridge
from align.baselines import auc_matrix
from align.budget import AMP_D
from align.decisive import stratified_auc, test_strata
from align.spiked import ENCODERS

NS = (100, 400, 1600)
CS = np.logspace(-4, 2, 13)


def run(a):
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    names = label_names(a.emb)
    train, test = (np.flatnonzero(cases.split == s) for s in ("train", "test"))
    y = normalize(np.load(f"{a.emb}/textmid_{a.model}.npy").astype(np.float64))
    lab_all = cases[names].to_numpy(float)
    lab, lt = lab_all[test], lab_all[train]
    group = test_strata(cases, lab, test)
    my = y[train].mean(0)
    fy = lambda v: normalize(v - my)
    yt = fy(y[train])
    base = [make_pipeline(StandardScaler(), LogisticRegression()).fit(yt, lt[:, j]) for j in range(len(names))]
    tuned = [make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=CS, cv=5, scoring="roc_auc", max_iter=5000,
                                                                  n_jobs=-1)).fit(yt, lt[:, j])
             for j in range(len(names))]
    chosen_c = {n: float(c[-1].C_[0]) for n, c in zip(names, tuned)}
    print(json.dumps({"model": a.model, "C": chosen_c}), flush=True)
    key = f"{a.model}/mid/{AMP_D}"
    rows = []
    for e in ENCODERS:
        k = max(1, len(json.load(open(f"{a.spiked}/{e}_estimates.json"))[key]["r_hat"]))   # at least one direction
        x = normalize(np.stack([np.load(f"{a.emb}/{e}/{os.path.basename(c)}.npy") for c in cases["key"]]).astype(np.float64))
        mx = x[train].mean(0)
        fx = lambda v: normalize(v - mx)
        for seed in SEEDS:
            perm = np.random.default_rng(seed).permutation(train)      # as align.decisive readout
            for n in NS:
                w = ridge(fx(x[perm[:n]]), fy(y[perm[:n]])).coef_.T
                u = fx(x[test]) @ w
                vt = svd(w, full_matrices=False)[2][:k]
                sub = [make_pipeline(StandardScaler(), LogisticRegression()).fit(yt @ vt.T, lt[:, j])
                       for j in range(len(names))]
                scores = {"base": np.stack([c.decision_function(u) for c in base], 1),
                          "A": np.stack([c.decision_function(u) for c in tuned], 1),
                          "B": np.stack([c.decision_function(u @ vt.T) for c in sub], 1)}
                for r, sc in scores.items():
                    rows.append({"encoder": e, "model": a.model, "k": k, "seed": seed, "n": n, "reader": r,
                                 "strat_auc": float(np.nanmean(stratified_auc(sc, lab, group))),
                                 "mean_auc": float(np.nanmean(auc_matrix(sc, lab)))})
                print(json.dumps(rows[-1]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/reader_opt_{a.model}.csv", index=False)
    json.dump(chosen_c, open(f"{a.out}/reader_opt_{a.model}_C.json", "w"), indent=1)


def summary(a):
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{a.out}/reader_opt_*.csv"))])
    m = df.groupby(["model", "encoder", "n", "reader"])[["strat_auc", "mean_auc"]].mean()
    res = {}
    for model in df.model.unique():
        res[model] = {}
        for v in ("A", "B"):
            d = {n: (m.xs((model, n, v), level=("model", "n", "reader")) -
                     m.xs((model, n, "base"), level=("model", "n", "reader"))) for n in NS}
            gain = {int(n): {"strat_median": float(g.strat_auc.median()), "mean_median": float(g.mean_auc.median()),
                             "strat_better": int((g.strat_auc > 0).sum())} for n, g in d.items()}
            res[model][v] = {"by_n": gain, "pass": bool(gain[1600]["strat_median"] >= 0.01
                                                        and min(gain[n]["strat_median"] for n in NS) >= -0.005)}
    json.dump(res, open(f"{a.out}/reader_opt_summary.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "summary"])
    ap.add_argument("--emb")
    ap.add_argument("--spiked")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model")
    a = ap.parse_args()
    {"run": run, "summary": summary}[a.step](a)


if __name__ == "__main__":
    main()
