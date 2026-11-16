"""V2: the locked official confirmation matrix. For one encoder and one text model: ridge maps on n pairs
of the CT-RATE train split, readers P / RP / TC with the full and the rank-one map, scored on every evaluation unit of
the official validation split (align.official.units). Pools: "ro" = exactly the RO fits (pairs from split "train" in
the RO permutation, readers fitted on split "train" reports); "all" = all 5,994 train-split patients (the V anchor).
At n = 1,600 also: patient-clustered bootstrap intervals (one set of patient resamples shared by every reader, unit and
statistic; each resample scores every seed and averages, so the intervals belong to the seed-mean AUC of the CSV) of the
mean AUC over 18 and 16 findings for P, RP and TC with the full map, on every unit; the E metrics (axis-1 weight share,
effective rank of the score matrix) on the a_1 unit; and the per-seed scores of every validation volume (float64) for
paired comparisons between cells (align.official_matrix_summary).

    python -m align.official_matrix --train DIR --valid DIR --out DIR --encoder E --model M --pool ro|all
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from align.align import SEEDS, label_names, normalize, svd
from align.anchor_followup import ridge
from align.baselines import auc_matrix
from align.decisive import rank_one, stratified_auc, test_strata
from align.official import units

NS = (25, 50, 100, 200, 400, 800, 1600)
BOOT_N, BOOT = 1600, 1000
DROP16 = ("Lymphadenopathy", "Medical material")


def patient_resamples(patients, seed=0):
    """BOOT index arrays; each draws patients with replacement and takes all their volumes. Fixed seed, so every
    cell of the matrix sees the same resamples and paired differences between cells are valid."""
    uniq, inv = np.unique(patients, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))]) for _ in range(BOOT)]


def boot_auc(per_seed, lab, cols, resamples):
    """Seed-mean of the mean AUC over the findings in cols, for each resample."""
    return np.array([np.mean([np.nanmean(auc_matrix(sc[idx], lab[idx])[cols]) for sc in per_seed])
                     for idx in resamples])


def e_metrics(s, vt, d, sc):
    e = (s[:, None] * (vt @ d)) ** 2
    scc = sc - sc.mean(0)
    sv = svd(scc, compute_uv=False) ** 2
    p = sv / sv.sum()
    return float(np.median(e[0] / e.sum(0))), float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", choices=["ro", "all"], required=True)
    a = ap.parse_args()
    names = label_names(a.train)
    assert names == label_names(a.valid)
    keep16 = np.array([l not in DROP16 for l in names])
    tc = pd.read_json(f"{a.train}/cases.jsonl", lines=True)
    vc = pd.read_json(f"{a.valid}/cases.jsonl", lines=True)
    load = lambda emb, cases: np.stack([np.load(f"{emb}/{a.encoder}/{os.path.basename(k)}.npy")
                                        for k in cases["key"]]).astype(np.float64)
    xt, xv = normalize(load(a.train, tc)), normalize(load(a.valid, vc))
    y = normalize(np.load(f"{a.train}/textmid_{a.model}.npy").astype(np.float64))
    prompts = normalize(np.load(f"{a.train}/promptsmid_{a.model}.npy").astype(np.float64))
    assert len(y) == len(tc)
    pool = np.flatnonzero(tc.split == "train") if a.pool == "ro" else np.arange(len(tc))
    lt = tc[names].to_numpy(float)
    lv = vc[names].to_numpy(float)
    masks = units(vc)
    groups = {u: test_strata(vc[m].reset_index(drop=True), lv[m], np.arange(int(m.sum()))) for u, m in masks.items()}
    patients = vc["id"].str.split("_").str[1].to_numpy()
    mx, my = xt[pool].mean(0), y[pool].mean(0)          # as RO: centering from the training pool
    fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
    yp, lp = fy(y[pool]), lt[pool]
    pr = fy(prompts)
    dirs = {"P": (pr[0::2] - pr[1::2]).T,
            "RP": np.stack([yp[lp[:, j] == 1].mean(0) - yp[lp[:, j] == 0].mean(0) for j in range(len(names))], 1)}
    clfs = [make_pipeline(StandardScaler(), LogisticRegression()).fit(yp, lp[:, j]) for j in range(len(names))]
    dirs_tc = np.stack([c[-1].coef_[0] / c[0].scale_ for c in clfs], 1)
    xvn = fx(xv)
    rows, erows, keep = [], [], {}
    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(pool)
        for n in NS:
            w = ridge(fx(xt[perm[:n]]), fy(y[perm[:n]])).coef_.T
            for mname, m in (("full", w), ("rank_one", rank_one(w))):
                u = xvn @ m
                scores = {r: normalize(u) @ d for r, d in dirs.items()}      # prompt readers: cosine
                scores["TC"] = np.stack([c.decision_function(u) for c in clfs], 1)
                for r, sc in scores.items():
                    for un, mk in masks.items():
                        auc = auc_matrix(sc[mk], lv[mk])
                        rows.append({"encoder": a.encoder, "model": a.model, "pool": a.pool, "unit": un,
                                     "volumes": int(mk.sum()), "seed": seed, "n": n, "map": mname, "reader": r,
                                     "mean_auc": float(np.nanmean(auc)), "mean_auc16": float(np.nanmean(auc[keep16])),
                                     "strat_auc": float(np.nanmean(stratified_auc(sc[mk], lv[mk], groups[un])))})
                    if n == BOOT_N and mname == "full":
                        keep.setdefault(r, []).append(sc)
            if n == BOOT_N:
                _, s, vt = svd(w, full_matrices=False)
                a1 = masks["patient_a1"]
                for r, d in (("P", dirs["P"]), ("TC", dirs_tc)):
                    share, rank = e_metrics(s, vt, d, keep[r][-1][a1])
                    erows.append({"encoder": a.encoder, "model": a.model, "pool": a.pool, "seed": seed, "reader": r,
                                  "axis1_share": share, "eff_rank": rank})
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.encoder}_{a.model}_{a.pool}"
    pd.DataFrame(rows).to_csv(f"{a.out}/{tag}.csv", index=False)
    pd.DataFrame(erows).to_csv(f"{a.out}/{tag}_e.csv", index=False)
    np.savez_compressed(f"{a.out}/{tag}_scores.npz", ids=vc["id"].to_numpy(),
                        **{r: np.stack(v) for r, v in keep.items()})
    boot = {}
    for un, mk in masks.items():
        idx_un = np.flatnonzero(mk)
        res = patient_resamples(patients[idx_un])
        boot[un] = {}
        for lname, cols in (("mean_auc", np.ones(len(names), bool)), ("mean_auc16", keep16)):
            draws = {r: boot_auc([sc[idx_un] for sc in keep[r]], lv[idx_un], cols, res) for r in ("P", "RP", "TC")}
            for r, d in draws.items():
                boot[un][f"{r}/{lname}"] = [float(v) for v in np.percentile(d, [2.5, 97.5])]
            boot[un][f"TC_minus_P/{lname}"] = [float(v) for v in np.percentile(draws["TC"] - draws["P"], [2.5, 97.5])]
    json.dump(boot, open(f"{a.out}/{tag}_boot.json", "w"), indent=1)
    print(json.dumps({"tag": tag, "rows": len(rows), "boot": boot}), flush=True)


if __name__ == "__main__":
    main()
