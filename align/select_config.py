"""Configuration selection on held-out patients: the report classifier TC (as align.decisive readout,
full map) for one image feature set and one report embedding file, at every pair budget of NS, 5 seeds of the RO
permutation, scored on the CT-RATE train split "test" patients. The official validation set is not read.

    python -m align.select_config run --emb DIR --out DIR --image voxelfm --text textmid_4B
    python -m align.select_config summary --out DIR --stage S2
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from align.align import SEEDS, label_names, normalize
from align.anchor_followup import ridge
from align.baselines import auc_matrix
from align.decisive import stratified_auc, test_strata

NS = (25, 50, 100, 200, 400, 800, 1600)


def run(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    names = label_names(a.emb)
    train, test = (np.flatnonzero(cases.split == s) for s in ("train", "test"))
    x = normalize(np.stack([np.load(f"{a.emb}/{a.image}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64))
    y = normalize(np.load(f"{a.emb}/{a.text}.npy").astype(np.float64))
    assert len(x) == len(y) == len(cases)
    lab_all = cases[names].to_numpy(float)
    lab = lab_all[test]
    group = test_strata(cases, lab, test)
    mx, my = x[train].mean(0), y[train].mean(0)
    fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
    clfs = [make_pipeline(StandardScaler(), LogisticRegression()).fit(fy(y[train]), lab_all[train, j])
            for j in range(len(names))]
    rows = []
    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(train)       # as align.decisive readout
        for n in NS:
            u = fx(x[test]) @ ridge(fx(x[perm[:n]]), fy(y[perm[:n]])).coef_.T
            sc = np.stack([c.decision_function(u) for c in clfs], 1)
            rows.append({"image": a.image, "text": a.text, "seed": seed, "n": n,
                         "mean_auc": float(np.nanmean(auc_matrix(sc, lab))),
                         "strat_auc": float(np.nanmean(stratified_auc(sc, lab, group)))})
            print(json.dumps(rows[-1]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/{a.image}__{a.text}.csv", index=False)


def summary(a):
    """Rank cells by the mean over NS of the seed-mean held-out mean AUC (budget-averaged). A cell within two paired
    standard errors of the leader (paired over seeds, on the budget-averaged score) counts as tied; ties go to the
    higher mean AUC at n = 1,600."""
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{a.out}/*__*.csv"))])
    df["cell"] = df.image + " | " + df.text
    per_seed = df.groupby(["cell", "seed"]).mean_auc.mean().unstack("seed")        # budget-averaged, per seed
    score = per_seed.mean(1).sort_values(ascending=False)
    lead = score.index[0]
    at1600 = df[df.n == 1600].groupby("cell").mean_auc.mean()
    rows = []
    for c in score.index:
        d = per_seed.loc[lead] - per_seed.loc[c]
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if c != lead else 0.0
        rows.append({"cell": c, "budget_avg": float(score[c]), "auc1600": float(at1600[c]),
                     "lead_minus": float(d.mean()), "se": se, "tied": bool(c == lead or d.mean() <= 2 * se)})
    t = pd.DataFrame(rows)
    tied = t[t.tied]
    pick = tied.sort_values("auc1600", ascending=False).iloc[0].cell
    curves = df.groupby(["cell", "n"]).mean_auc.mean().unstack("n")
    res = {"stage": a.stage, "pick": pick, "table": t.to_dict("records"),
           "curves": {c: {int(n): float(v) for n, v in curves.loc[c].items()} for c in curves.index}}
    json.dump(res, open(f"{a.out}/summary_{a.stage}.json", "w"), indent=1)
    print(t.to_string(index=False))
    print("PICK", pick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "summary"])
    ap.add_argument("--emb")
    ap.add_argument("--out", required=True)
    ap.add_argument("--image")
    ap.add_argument("--text")
    ap.add_argument("--stage", default="S2")
    a = ap.parse_args()
    {"run": run, "summary": summary}[a.step](a)


if __name__ == "__main__":
    main()
