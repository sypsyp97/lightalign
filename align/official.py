"""V: report-trained and prompt readouts of a ridge map, trained on the CT-RATE train split and scored on the official
CT-RATE validation split.

    python -m align.official --train DIR --valid DIR --out DIR [--encoder voxelfm]

A valid directory built with `align.ctrate cases --subset valid-all` (every reconstruction) is scored on the units of
the published CT-RATE numbers: all 3,039 reconstructions, the 3,002 without the dataset's non-chest list, one
reconstruction per study (the lowest number), and the pre-registered one a_1 reconstruction per patient without
non-chest volumes. A directory without those columns is that last unit only.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from align.align import SEEDS, label_names, normalize
from align.anchor_followup import ridge
from align.baselines import auc_matrix
from align.budget import CURVE_MODEL
from align.decisive import stratified_auc, test_strata

NS = (100, 400, 1600)


def load(emb, encoder, text=True):
    cases = pd.read_json(f"{emb}/cases.jsonl", lines=True)
    x = np.stack([np.load(f"{emb}/{encoder}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64)
    if not text:
        return cases, normalize(x), None
    y = np.load(f"{emb}/textmid_{CURVE_MODEL}.npy").astype(np.float64)
    assert len(x) == len(y) == len(cases), (len(x), len(y), len(cases))
    return cases, normalize(x), normalize(y)


def units(cases):
    """Row masks of the evaluation units present in this validation directory."""
    if "no_chest" not in cases:
        return {"patient_a1": np.ones(len(cases), bool)}
    first = cases.groupby("study").recon.transform("min") == cases.recon
    a1 = (cases.study.str.endswith("_a") & (cases.recon == 1) & ~cases.no_chest).to_numpy()
    return {"all": np.ones(len(cases), bool), "chest": ~cases.no_chest.to_numpy(), "study": first.to_numpy(),
            "patient_a1": a1}


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="voxelfm")
    a = ap.parse_args()
    names = label_names(a.train)
    assert names == label_names(a.valid)
    tcases, xt, yt_raw = load(a.train, a.encoder)
    vcases, xv, _ = load(a.valid, a.encoder, text=False)
    masks = units(vcases)
    prompts = normalize(np.load(f"{a.train}/promptsmid_{CURVE_MODEL}.npy").astype(np.float64))
    lt = tcases[names].to_numpy(float)
    lv = vcases[names].to_numpy(float)
    mx, my = xt.mean(0), yt_raw.mean(0)
    fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
    yt = fy(yt_raw)
    pr = fy(prompts)
    diff = (pr[0::2] - pr[1::2]).T
    fit = lambda z: [make_pipeline(StandardScaler(), LogisticRegression()).fit(z, lt[:, j]) for j in range(len(names))]
    text_clfs, image_clfs = fit(yt), fit(fx(xt))
    rows = []

    keep16 = np.array([l not in ("Lymphadenopathy", "Medical material") for l in names])   # 16-label papers (fVLM, HLIP)
    groups = {u: test_strata(vcases[m].reset_index(drop=True), lv[m], np.arange(int(m.sum()))) for u, m in masks.items()}

    def score(reader, n, seed, sc):
        for u, m in masks.items():
            auc = auc_matrix(sc[m], lv[m])
            rows.append({"encoder": a.encoder, "unit": u, "volumes": int(m.sum()), "reader": reader, "n": n,
                         "seed": seed, "mean_auc": float(np.nanmean(auc)), "mean_auc16": float(np.nanmean(auc[keep16])),
                         "strat_auc": float(np.nanmean(stratified_auc(sc[m], lv[m], groups[u]))),
                         **{f"auc|{l}": float(v) for l, v in zip(names, auc)}})
            print(json.dumps({k: v for k, v in rows[-1].items() if "|" not in k}), flush=True)

    every = np.arange(len(xt))
    plan = [(n, seed, np.random.default_rng(seed).permutation(every)[:n]) for n in NS for seed in SEEDS]
    plan.append((len(xt), 0, every))
    for n, seed, idx in plan:
        u = fx(xv) @ ridge(fx(xt[idx]), yt[idx]).coef_.T
        score("P", n, seed, u @ diff)
        score("TC", n, seed, np.stack([c.decision_function(u) for c in text_clfs], 1))
    score("IL", len(xt), 0, np.stack([c.decision_function(fx(xv)) for c in image_clfs], 1))

    os.makedirs(a.out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/{a.encoder}_official.csv", index=False)
    table = df.groupby(["unit", "volumes", "reader", "n"])[["mean_auc", "mean_auc16", "strat_auc"]].agg(["mean", "std"]).round(4)
    pa = df[df.unit == "patient_a1"]
    tc, p = (pa[(pa.reader == r) & (pa.n == 1600)].mean_auc.mean() for r in ("TC", "P"))
    res = {"table": {f"{u}({v})/{r}/{n}": {f"{c}_{s}": x for (c, s), x in row.items()}
                     for (u, v, r, n), row in table.iterrows()},
           "V1": {"tc_minus_p_1600": float(tc - p), "pass": bool(tc - p >= 0.03)},
           "n_train": int(len(xt)), "n_valid": int(len(xv))}
    json.dump(res, open(f"{a.out}/{a.encoder}_official_summary.json", "w"), indent=1)
    print(table.to_string(), flush=True)
    print(json.dumps(res["V1"]), flush=True)


if __name__ == "__main__":
    main()
