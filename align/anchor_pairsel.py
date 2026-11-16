"""Anchor-informed pair selection: do a few true pairs chosen to cover the directions the
anchors cannot fix beat random pairs and the same design rule without the anchors?

Same halves and anchor pseudo pairs as align.anchor_zero. True pairs come from half A (image with its own report).
Arms per m: anchors + random / residual design / anchor-blind design, random alone, OT refinement on random pairs;
plus a random-alone reference curve to read an equivalent pair count. Weighting: true pairs share the total weight of
all pseudo pairs.

    python -m align.anchor_pairsel --emb DIR --out DIR [--encoder voxelfm] [--model 4B]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import RidgeCV

from align.align import SEEDS, label_names, normalize, svd
from align.anchor_followup import foscttm_per_patient, ridge
from align.anchor_zero import ANCHORS, image_anchors, ranks
from align.baselines import ALPHAS, N_BOOT, auc_matrix, zero_shot_scores

MS = (10, 25, 50)
REF_MS = (10, 25, 50, 100, 200, 400)
PCS = 32
OT_ROUNDS, OT_EPS, SINKHORN_MAX_ITERS, SINKHORN_TOL = 3, 0.05, 5000, 1e-6


def greedy_design(z, m):
    """Greedy maximisation of log det(lambda I + Z_S^T Z_S), lambda = mean column variance of z."""
    lam = z.var(0).mean()
    minv = np.eye(z.shape[1]) / lam
    chosen = []
    for _ in range(m):
        gain = np.einsum("ij,jk,ik->i", z, minv, z)
        gain[chosen] = -np.inf
        k = int(np.argmax(gain))
        chosen.append(k)
        v = minv @ z[k]
        minv -= np.outer(v, v) / (1 + z[k] @ v)
    return np.array(chosen)


def pcs(v, r=PCS):
    v = v - v.mean(0)
    u, s, _ = svd(v, full_matrices=False)
    return u[:, :r] * s[:r] / np.sqrt(len(v))


def sinkhorn(cost, eps=OT_EPS):
    """Entropic OT with uniform marginals; raises unless both marginals match to SINKHORN_TOL (relative)."""
    k = np.exp(-(cost - cost.min()) / eps)
    a, b = np.full(cost.shape[0], 1 / cost.shape[0]), np.full(cost.shape[1], 1 / cost.shape[1])
    v = np.ones(cost.shape[1])
    for _ in range(SINKHORN_MAX_ITERS):
        u = a / (k @ v)
        v = b / (k.T @ u)
        p = u[:, None] * k * v[None, :]
        if not np.isfinite(p).all():
            raise ValueError("sinkhorn produced non-finite plan")
        if max(np.abs(p.sum(1) / a - 1).max(), np.abs(p.sum(0) / b - 1).max()) < SINKHORN_TOL:
            return p
    raise ValueError(f"sinkhorn did not converge in {SINKHORN_MAX_ITERS} iterations")


def weighted_fit(xs, ys, xt, yt, m):
    """Ridge on pseudo (xs, ys, weight 1 each) plus m true pairs (xt, yt) carrying the same total weight."""
    w = np.r_[np.ones(len(xs)), np.full(m, len(xs) / m)]
    return RidgeCV(alphas=ALPHAS, fit_intercept=False).fit(np.r_[xs, xt], np.r_[ys, yt], sample_weight=w).coef_.T


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

    rows, scores, fos = [], {}, {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)   # same halves as align.anchor_zero
        perm = rng.permutation(train)
        half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]
        mx, my = x[half_a].mean(0), y[half_b].mean(0)
        fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
        pr = fy(prompts)
        xa, ya, yb, xt, yt = fx(x[half_a]), fy(y[half_a]), fy(y[half_b]), fx(x[test]), fy(y[test])
        ra, rb = ranks(img_anchor[half_a]), ranks(zero_shot_scores(yb, pr)[:, idx])
        i, j = linear_sum_assignment(((ra[:, None, :] - rb[None, :, :]) ** 2).sum(-1))
        xs, ys = xa[i], yb[j]

        to_rank = ridge(xa, ra, True).coef_            # (4, d): feature directions that carry the anchors
        q, _ = np.linalg.qr(to_rank.T)
        xc = xa - xa.mean(0)
        z_res, z_full = pcs(xc - (xc @ q) @ q.T), pcs(xc)
        pick = np.random.default_rng(seed + 2000)
        rand_order = pick.permutation(len(half_a))

        maps = {}
        for m in MS:
            sel = {"random": rand_order[:m], "residual": greedy_design(z_res, m), "blind": greedy_design(z_full, m)}
            for name, s in sel.items():
                maps[(f"anchors+{name}", m)] = weighted_fit(xs, ys, xa[s], ya[s], m)
            s = rand_order[:m]
            w = ridge(xa[s], ya[s]).coef_.T
            for _ in range(OT_ROUNDS):
                p = sinkhorn(1 - normalize(xa @ w) @ yb.T)
                bary = p @ yb / p.sum(1, keepdims=True)
                w = weighted_fit(xa, bary, xa[s], ya[s], m)
            maps[("ot_refine", m)] = w
        for m in REF_MS:
            s = rand_order[:m]
            maps[("random_alone", m)] = ridge(xa[s], ya[s]).coef_.T
        for (name, m), w in maps.items():
            u = normalize(xt @ w)
            sc = zero_shot_scores(u, pr)
            f = foscttm_per_patient(u @ yt.T)
            auc = auc_matrix(sc, lab_t)
            scores.setdefault((name, m), []).append(sc)
            fos.setdefault((name, m), []).append(f)
            rows.append({"seed": seed, "method": name, "m": m, "mean_auc": float(np.nanmean(auc)),
                         "auc_other_labels": float(np.nanmean(auc[other])), "foscttm": float(f.mean())})
        print(f"[pairsel] {a.encoder} seed={seed} " + " ".join(
            f"{r['method']}@{r['m']}={r['auc_other_labels']:.3f}" for r in rows[-len(maps):]), flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/{a.encoder}_cells.csv", index=False)
    summ = df.groupby(["method", "m"])[["auc_other_labels", "mean_auc", "foscttm"]].mean()

    def boot_auc(key):
        return np.mean([[np.nanmean(auc_matrix(s[b], lab_t[b])[other]) for b in boot] for s in scores[key]], 0)

    def boot_fos(key):
        return np.mean([[f[b].mean() for b in boot] for f in fos[key]], 0)

    def diff(k1, k2, fn, col):
        d = fn(k1) - fn(k2)
        return {"point": float(summ.loc[k1, col] - summ.loc[k2, col]),
                "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975))}

    contrasts = {}
    for m in MS:
        contrasts[f"m{m}_residual_minus_random"] = diff(("anchors+residual", m), ("anchors+random", m), boot_auc, "auc_other_labels")
        contrasts[f"m{m}_residual_minus_blind"] = diff(("anchors+residual", m), ("anchors+blind", m), boot_auc, "auc_other_labels")
    contrasts["m25_residual_minus_random100"] = diff(("anchors+residual", 25), ("random_alone", 100), boot_auc, "auc_other_labels")
    contrasts["m25_residual_minus_random100_foscttm"] = diff(("anchors+residual", 25), ("random_alone", 100), boot_fos, "foscttm")
    c = contrasts
    passed = bool(c["m25_residual_minus_random"]["point"] >= 0.01 and c["m25_residual_minus_random"]["lo"] > 0
                  and c["m25_residual_minus_blind"]["lo"] > 0
                  and c["m25_residual_minus_random100"]["point"] >= 0
                  and c["m25_residual_minus_random100_foscttm"]["point"] <= 0.005)
    ref = summ.loc["random_alone"].auc_other_labels
    target = summ.loc[("anchors+residual", 25), "auc_other_labels"]
    if not ref.is_monotonic_increasing:
        equiv = "reference curve not monotone"
    elif not ref.values[0] <= target <= ref.values[-1]:
        equiv = f"outside reference range [{REF_MS[0]}, {REF_MS[-1]}] pairs"
    else:   # pairs of random_alone with the same non-anchor AUC, log-linear between reference points
        equiv = float(np.exp(np.interp(target, ref.values, np.log(ref.index.values.astype(float)))))
    out = {"encoder": a.encoder, "model": a.model,
           "summary": [{"method": k[0], "m": int(k[1]), **{col: float(v) for col, v in r.items()}} for k, r in summ.iterrows()],
           "contrasts": contrasts, "pass_voxelfm_criteria": passed,
           "equivalent_random_pairs_residual_m25": equiv,
           "residual_minus_blind_m25_sign": float(np.sign(c["m25_residual_minus_blind"]["point"]))}
    json.dump(out, open(f"{a.out}/{a.encoder}.json", "w"), indent=1)
    print(summ.to_string())
    print(json.dumps({k: out[k] for k in ("contrasts", "pass_voxelfm_criteria", "equivalent_random_pairs_residual_m25")}, indent=1))
    print("ANCHOR_PAIRSEL_EXIT rc=0", flush=True)


if __name__ == "__main__":
    main()
