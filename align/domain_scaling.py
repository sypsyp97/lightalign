"""Image model vs text model, and the 200-pair pilot, in every domain: for each image encoder x
text model of one domain, a ridge map from n paired train items (n = 100, 400, 1,600; three seeds) scored on the test
split by FOSCTTM, and, on labelled domains, the report-classifier AUC (align.official_ablation reader). The pilot: r_hat
from the first 200 pairs of the same permutation (align.spiked.estimate on top-64 principal coordinates, as
align.planner) and its law score S(1,600).

Summary per domain: share of the variance of the score over the encoder x text-model grid explained by the encoder,
the text model and the rest (two-way sums of squares of the cell means), at each n; per seed (one 200-pair pilot each),
Spearman of the pilot score S with the cell's score at 1,600 pairs, and the loss of the pilot's pick against the best
cell. `summary` recomputes this from <tag>_cells.csv.

    python -m align.domain_scaling run --emb DIR --encoders E1,E2 --texts textmid_4B,text_mpnet,... --out DIR
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from align.align import foscttm, label_names, normalize
from align.anchor_followup import ridge
from align.baselines import auc_matrix
from align.planner import D, law
from align.spiked import estimate, pcs

NS = (100, 400, 1600)
PILOT = 200
SEEDS = range(3)


def run(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    train, test = (np.flatnonzero(cases.split == s) for s in ("train", "test"))
    names = label_names(a.emb) if os.path.exists(f"{a.emb}/labels.json") else []
    lab = cases[names].to_numpy(float) if names else None
    rows = []
    for t in a.texts.split(","):
        y = normalize(np.load(f"{a.emb}/{t}.npy").astype(np.float64))
        if len(y) != len(cases):
            raise ValueError(f"{t}: {len(y)} rows for {len(cases)} cases")
        my = y[train].mean(0)
        fy = lambda v: normalize(v - my)
        yt, y_pc = fy(y[test]), pcs(y[train], D)
        clfs = [make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(fy(y[train]), lab[train, j])
                for j in range(len(names))] if names else []
        for e in a.encoders.split(","):
            x = normalize(np.stack([np.load(f"{a.emb}/{e}/{os.path.basename(k)}.npy") for k in cases["key"]])
                          .astype(np.float64))
            mx = x[train].mean(0)
            fx = lambda v: normalize(v - mx)
            xt, x_pc = fx(x[test]), pcs(x[train], D)
            for seed in SEEDS:
                perm = np.random.default_rng(seed).permutation(len(train))
                r_hat = estimate(x_pc[perm[:PILOT]], y_pc[perm[:PILOT]], np.random.default_rng(seed + 4000))[0]
                r_hat = r_hat[r_hat > 0]
                k, s = law(r_hat, max(NS))
                for n in NS:
                    idx = train[perm[:n]]
                    u = xt @ ridge(fx(x[idx]), fy(y[idx])).coef_.T
                    row = {"data": os.path.basename(a.emb.rstrip("/")), "encoder": e, "text": t, "seed": seed, "n": n,
                           "foscttm": foscttm(normalize(u) @ yt.T), "S_pilot": s, "k_pilot": k,
                           "r1_pilot": float(r_hat[0]) if len(r_hat) else 0.0}
                    if names:
                        sc = np.stack([c.decision_function(u) for c in clfs], 1)
                        row["auc"] = float(np.nanmean(auc_matrix(sc, lab[test])))
                    rows.append(row)
            print(json.dumps(rows[-1]), flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    tag = os.path.basename(a.emb.rstrip("/"))
    df.to_csv(f"{a.out}/{tag}_cells.csv", index=False)
    summarize(df, a.out, tag)


def summarize(df, out, tag):
    """Variance shares of the cell means; the pilot judged per seed, since each seed draws its own 200 pilot pairs."""
    res = {}
    for metric, sign in (("foscttm", -1), ("auc", 1)):
        if metric not in df:
            continue
        m = df.groupby(["encoder", "text", "n"])[[metric]].mean().reset_index()
        for n, g in m.groupby("n"):
            grid = g.pivot(index="encoder", columns="text", values=metric)
            tot = ((grid - grid.values.mean()) ** 2).values.sum()
            enc = grid.shape[1] * ((grid.mean(1) - grid.values.mean()) ** 2).sum()
            txt = grid.shape[0] * ((grid.mean(0) - grid.values.mean()) ** 2).sum()
            res[f"{metric}/n{n}"] = {"share_encoder": float(enc / tot), "share_text": float(txt / tot),
                                     "share_rest": float(1 - (enc + txt) / tot),
                                     "range_over_encoders": float(np.ptp(grid.mean(1))),
                                     "range_over_texts": float(np.ptp(grid.mean(0))),
                                     "cell_means": grid.round(4).to_dict()}
        last = df[df.n == max(NS)]
        score = last.groupby(["encoder", "text"])[metric].mean()          # the cell's score at 1,600 pairs
        best = (sign * score).idxmax()
        per_seed = []
        for seed, g in last.groupby("seed"):
            s = g.set_index(["encoder", "text"])["S_pilot"].reindex(score.index)
            pick = s.idxmax()
            per_seed.append({"seed": int(seed), "rho_S_vs_score": float(sign * spearmanr(s, score).statistic),
                             "pilot_pick": list(pick), "score_loss_of_pick": float(sign * (score[best] - score[pick]))})
        res[f"{metric}/pilot"] = {"best_cell": list(best), "per_seed": per_seed,
                                  "rho_min": min(p["rho_S_vs_score"] for p in per_seed),
                                  "rho_median": float(np.median([p["rho_S_vs_score"] for p in per_seed])),
                                  "loss_max": max(p["score_loss_of_pick"] for p in per_seed)}
    json.dump(res, open(f"{out}/{tag}_summary.json", "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "cell_means"} for k, v in res.items()}, indent=1),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "summary"])
    ap.add_argument("--emb", required=True)
    ap.add_argument("--encoders")
    ap.add_argument("--texts")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.step == "run":
        run(a)
    else:
        tag = os.path.basename(a.emb.rstrip("/"))
        summarize(pd.read_csv(f"{a.out}/{tag}_cells.csv"), a.out, tag)


if __name__ == "__main__":
    main()
