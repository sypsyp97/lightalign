"""Anchor follow-up: does zero-pair anchor alignment carry information beyond the anchors?

Same halves, pseudo pairs and 100 true pairs as align.anchor_zero. Methods:
  anchor_ridge   ridge on the anchor pseudo pairs (reproduces align.anchor_zero)
  true100_ridge  ridge on 100 true pairs (reference)
  bottleneck     ridge features -> image-anchor ranks (half A) composed with ridge text-anchor ranks -> report (half B)
  rematch_r1..5  re-matching: ridge on current pairs, Hungarian on anchor-rank cost + (1 - cos(prediction, report)),
                 each term divided by its mean; round 5 is primary
Outputs per-method cells, per-label AUC, bootstrap intervals of the pre-registered differences, the co-occurrence
analysis, and the per-round non-anchor agreement of the pseudo pairs (train labels, diagnostic only).

    python -m align.anchor_followup --emb DIR --out DIR [--encoder voxelfm] [--model 4B]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

from align.align import SEEDS, label_names, normalize
from align.anchor_zero import ANCHORS, image_anchors, ranks
from align.baselines import ALPHAS, N_BOOT, auc_matrix, zero_shot_scores
from align.t2 import K

ROUNDS = 5
PHI_WITH = ("Pleural effusion", "Cardiomegaly")


def ridge(a, b, intercept=False):
    return RidgeCV(alphas=ALPHAS, fit_intercept=intercept).fit(a, b)


def foscttm_per_patient(sim):
    """Per-patient FOSCTTM (mean of its row and column term); the mean over patients equals align.foscttm."""
    if not np.isfinite(sim).all():
        raise ValueError("non-finite similarity")
    d = np.diag(sim)
    n = len(d)
    off = ~np.eye(n, dtype=bool)
    row = (((sim > d[:, None]) + 0.5 * (sim == d[:, None])) * off).sum(1) / (n - 1)
    col = (((sim > d[None, :]) + 0.5 * (sim == d[None, :])) * off).sum(0) / (n - 1)
    return (row + col) / 2


def phi(a, b):
    a, b = a.astype(float), b.astype(float)
    return np.corrcoef(a, b)[0, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="voxelfm")
    ap.add_argument("--model", default="4B")
    a = ap.parse_args()
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    names = label_names(a.emb)
    lab = cases[names].to_numpy(float)
    x = normalize(np.stack([np.load(f"{a.emb}/{a.encoder}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64))
    y = normalize(np.load(f"{a.emb}/textmid_{a.model}.npy").astype(np.float64))
    prompts = normalize(np.load(f"{a.emb}/promptsmid_{a.model}.npy").astype(np.float64))
    img_anchor = image_anchors(f"{a.emb}/measures.csv", cases["key"])
    idx = [names.index(n) for n in ANCHORS]
    other = [k for k in range(len(names)) if k not in idx]
    train, test = (np.flatnonzero(cases.split == s) for s in ("train", "test"))
    boot = np.random.default_rng(12345).integers(0, len(test), (N_BOOT, len(test)))
    lab_t = lab[test]

    rows, agree, scores, fos = [], [], {}, {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)   # same draws as align.anchor_zero: permutation, then the true pairs
        perm = rng.permutation(train)
        half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]
        pairs = rng.choice(half_a, K, replace=False)
        mx, my = x[half_a].mean(0), y[half_b].mean(0)
        fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
        pr = fy(prompts)
        xa, yb, xt, yt = fx(x[half_a]), fy(y[half_b]), fx(x[test]), fy(y[test])
        ra, rb = ranks(img_anchor[half_a]), ranks(zero_shot_scores(yb, pr)[:, idx])
        anchor_cost = ((ra[:, None, :] - rb[None, :, :]) ** 2).sum(-1)
        i, j = linear_sum_assignment(anchor_cost)          # on the raw cost, as align.anchor_zero (same tie breaks)
        anchor_cost = anchor_cost / anchor_cost.mean()
        rand_j = np.random.default_rng(seed + 1000).permutation(j)

        def agreement(jj, rnd=rand_j, ii=i):
            ok = lab[half_a[ii]][:, other] == lab[half_b[jj]][:, other]
            base = lab[half_a[ii]][:, other] == lab[half_b[rnd]][:, other]
            return float(ok.mean() - base.mean())

        preds = {"anchor_ridge": xt @ ridge(xa[i], yb[j]).coef_.T,
                 "true100_ridge": xt @ ridge(fx(x[pairs]), fy(y[pairs])).coef_.T}
        to_rank, from_rank = ridge(xa, ra, True), ridge(rb, yb, True)
        preds["bottleneck"] = from_rank.predict(to_rank.predict(xt))
        agree.append({"seed": seed, "round": 0, "nonanchor_agreement_minus_random": agreement(j),
                      "anchor_rank_cost": float(anchor_cost[i, j].mean())})
        jj = j
        for r in range(1, ROUNDS + 1):
            w = ridge(xa[i], yb[jj]).coef_.T
            dist = 1 - normalize(xa @ w) @ yb.T
            ii, jj = linear_sum_assignment(anchor_cost + dist / dist.mean())
            order = np.argsort(ii)                       # keep image order aligned with i (ii is 0..n-1 sorted)
            ii, jj = ii[order], jj[order]
            agree.append({"seed": seed, "round": r, "nonanchor_agreement_minus_random": agreement(jj, ii=ii),
                          "anchor_rank_cost": float(anchor_cost[ii, jj].mean())})
            preds[f"rematch_r{r}"] = xt @ ridge(xa[ii], yb[jj]).coef_.T
            i = ii
        for name, p in preds.items():
            u = normalize(p)
            s = zero_shot_scores(u, pr)
            f = foscttm_per_patient(u @ yt.T)
            auc = auc_matrix(s, lab_t)
            scores.setdefault(name, []).append(s)
            fos.setdefault(name, []).append(f)
            rows.append({"seed": seed, "method": name, "mean_auc": float(np.nanmean(auc)),
                         "auc_anchor_labels": float(np.nanmean(auc[idx])), "auc_other_labels": float(np.nanmean(auc[other])),
                         "foscttm": float(f.mean()), **{f"auc_{n}": float(v) for n, v in zip(names, auc)}})
        print(f"[followup] seed={seed} " + " ".join(f"{r['method']}={r['auc_other_labels']:.3f}/{r['foscttm']:.3f}"
                                                   for r in rows[-len(preds):]), flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/{a.encoder}_cells.csv", index=False)
    pd.DataFrame(agree).to_csv(f"{a.out}/{a.encoder}_agreement.csv", index=False)

    def boot_other(name):   # (N_BOOT,) seed-averaged non-anchor AUC per patient resample
        return np.mean([[np.nanmean(auc_matrix(s[b], lab_t[b])[other]) for b in boot] for s in scores[name]], 0)

    def boot_fos(name):
        return np.mean([[f[b].mean() for b in boot] for f in fos[name]], 0)

    primary = f"rematch_r{ROUNDS}"
    bo = {m: boot_other(m) for m in ("anchor_ridge", "bottleneck", primary)}
    bf = {m: boot_fos(m) for m in ("anchor_ridge", primary)}
    summ = df.groupby("method")[["auc_other_labels", "mean_auc", "foscttm"]].mean()

    def interval(d, point):
        return {"point": float(point), "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975))}

    pt = summ.auc_other_labels
    d_b = interval(bo["anchor_ridge"] - bo["bottleneck"], pt["anchor_ridge"] - pt["bottleneck"])
    d_ra = interval(bo[primary] - bo["anchor_ridge"], pt[primary] - pt["anchor_ridge"])
    d_rb = interval(bo[primary] - bo["bottleneck"], pt[primary] - pt["bottleneck"])
    d_rf = interval(bf[primary] - bf["anchor_ridge"], summ.foscttm[primary] - summ.foscttm["anchor_ridge"])
    if abs(d_b["point"]) <= 0.01 and d_b["lo"] >= -0.02 and d_b["hi"] <= 0.02:
        bottleneck_verdict = "explicit concept bottleneck"
    elif d_b["point"] > 0.01 and d_b["lo"] > 0:
        bottleneck_verdict = "estimator advantage, not extra information"
    else:
        bottleneck_verdict = "inconclusive"
    rematch_pass = bool(d_ra["point"] >= 0.01 and d_ra["lo"] > 0 and d_rb["point"] >= 0.01 and d_rb["lo"] > 0
                        and d_rf["point"] <= 0.01)

    tr = lab[train]
    max_phi = [max(abs(phi(tr[:, k], tr[:, names.index(n)])) for n in PHI_WITH) for k in other]
    gap = [df[df.method == "anchor_ridge"][f"auc_{names[k]}"].mean() - df[df.method == "true100_ridge"][f"auc_{names[k]}"].mean()
           for k in other]
    rho, p = spearmanr(gap, max_phi)
    out = {"encoder": a.encoder, "model": a.model, "rounds": ROUNDS,
           "summary": {m: {c: float(summ.loc[m, c]) for c in summ.columns} for m in summ.index},
           "bottleneck": {"anchor_minus_bottleneck": d_b, "verdict": bottleneck_verdict},
           "rematch": {"r5_minus_anchor": d_ra, "r5_minus_bottleneck": d_rb, "foscttm_r5_minus_anchor": d_rf,
                       "pass": rematch_pass},
           "cooccurrence": {"spearman": float(rho), "p": float(p), "pass": bool(rho >= 0.5),
                            "per_label": [{"label": names[k], "anchor_minus_true100": float(g), "max_abs_phi": float(f)}
                                          for k, g, f in zip(other, gap, max_phi)]},
           "agreement_by_round": pd.DataFrame(agree).groupby("round").mean(numeric_only=True)
                                   .drop(columns="seed").to_dict(orient="index")}
    json.dump(out, open(f"{a.out}/{a.encoder}.json", "w"), indent=1)
    print(summ.to_string())
    print(json.dumps(out, indent=1), flush=True)
    print("ANCHOR_FOLLOWUP_EXIT rc=0", flush=True)


if __name__ == "__main__":
    main()
