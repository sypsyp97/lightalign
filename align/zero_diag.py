"""Why zero-pair alignment leaves chance on V-JEPA2.

Maps per seed, as in align.downstream / align.minivec2vec: n = 0 Gromov-Wasserstein, n = 0 mini-vec2vec, ridge on the
first 100 train pairs. On the test patients, per finding:
- auc:       zero-shot AUC of the finding's prompt pair;
- auc_resid: the same after removing the common axis (each finding's score regressed on the mean score over the 18
             findings, fitted on the test scores without labels): above 0.5 only if the map carries finding-specific
             information, not just one "abnormal vs normal" direction;
plus, per map, Spearman of the mean score with the number of positive findings (burden) and with report length in
words, and the report-length-only AUC per finding (a text-side confound that needs no image).

    python -m align.zero_diag --emb DIR --out DIR --encoder E
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from align.align import SEEDS, gromov_wasserstein, label_names, mini_vec2vec_official, normalize
from align.anchor_followup import ridge
from align.baselines import auc_matrix, zero_shot_scores
from align.downstream import MODEL, residual
from align.minivec2vec import basis, mini_vec2vec

N_PAIRS = 100


def official_mv2v(gx, gy, xa, yb, seed):
    """Official mini-vec2vec on the same halves and centring; its own centring composed on top (see gromov_wasserstein)."""
    w, ax, ay = mini_vec2vec_official(gx(xa), gy(yb), seed)
    return w, (lambda v: ax(gx(v))), (lambda v: ay(gy(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", required=True)
    a = ap.parse_args()
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    names = label_names(a.emb)
    train, test = (np.flatnonzero(cases.split == s) for s in ("train", "test"))
    x = normalize(np.stack([np.load(f"{a.emb}/{a.encoder}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64))
    y = normalize(np.load(f"{a.emb}/textmid_{MODEL}.npy").astype(np.float64))
    prompts = normalize(np.load(f"{a.emb}/promptsmid_{MODEL}.npy").astype(np.float64))
    lab = cases[names].to_numpy(float)[test]
    burden = lab.sum(1)
    length = cases["report"].str.split().str.len().to_numpy(float)[test]
    rows = []
    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(train)
        half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]
        hx, hy = x[half_a].mean(0), y[half_b].mean(0)
        gx, gy = (lambda v: normalize(v - hx)), (lambda v: normalize(v - hy))
        vx, vy = basis(gx(x[half_a])), basis(gy(y[half_b]))
        tx, ty = x[train].mean(0), y[train].mean(0)
        rx, ry = (lambda v: normalize(v - tx)), (lambda v: normalize(v - ty))
        w_gw, ax, ay = gromov_wasserstein(gx(x[half_a]), gy(y[half_b]), seed)   # ax, ay: the aligner's own centering
        maps = {
            "gw": (w_gw, (lambda v: ax(gx(v))), (lambda v: ay(gy(v)))),
            "mini_vec2vec": (vx @ mini_vec2vec(normalize(gx(x[half_a]) @ vx), normalize(gy(y[half_b]) @ vy),
                                               np.random.default_rng(seed + 3000)) @ vy.T, gx, gy),
            "mini_vec2vec_official": official_mv2v(gx, gy, x[half_a], y[half_b], seed),
            "ridge100": (ridge(rx(x[perm[:N_PAIRS]]), ry(y[perm[:N_PAIRS]])).coef_.T, rx, ry),
        }
        for name, (w, fx, fy) in maps.items():
            s = zero_shot_scores(normalize(fx(x[test]) @ w), fy(prompts))
            mean = s.mean(1)
            rows.append({"encoder": a.encoder, "seed": seed, "map": name,
                         **{f"auc|{l}": v for l, v in zip(names, auc_matrix(s, lab))},
                         **{f"auc_resid|{l}": v for l, v in zip(names, auc_matrix(residual(s), lab))},
                         "rho_mean_burden": float(spearmanr(mean, burden).statistic),
                         "rho_mean_length": float(spearmanr(mean, length).statistic)})
            print(json.dumps({k: v for k, v in rows[-1].items() if "|" not in k}), flush=True)
    df = pd.DataFrame(rows)
    length_auc = dict(zip(names, auc_matrix(np.repeat(length[:, None], len(names), 1), lab)))
    summ = {}
    for name, g in df.groupby("map"):
        auc = g[[f"auc|{l}" for l in names]].mean()
        res = g[[f"auc_resid|{l}" for l in names]].mean()
        summ[name] = {"median_auc": float(auc.median()), "median_auc_resid": float(res.median()),
                      "rho_mean_burden": float(g.rho_mean_burden.mean()), "rho_mean_length": float(g.rho_mean_length.mean()),
                      "per_finding": {l: [round(float(auc[f"auc|{l}"]), 3), round(float(res[f"auc_resid|{l}"]), 3)]
                                      for l in names}}
    summ["report_length_only"] = {"median_auc": float(np.nanmedian(list(length_auc.values()))),
                                  "per_finding": {l: round(float(v), 3) for l, v in length_auc.items()}}
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/{a.encoder}_zero_diag.csv", index=False)
    json.dump(summ, open(f"{a.out}/{a.encoder}_zero_diag.json", "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_finding"} for k, v in summ.items()}, indent=1))
    print("ZERO_DIAG_EXIT rc=0", flush=True)


if __name__ == "__main__":
    main()
