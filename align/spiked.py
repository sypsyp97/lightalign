"""Spectral detectability of shared image-report components from the finite-rank high-dimensional CCA limits
. Theory: Bao, Hu, Pan, Zhou, Ann. Statist. 2019 (arXiv 1704.02408), eqs. (1.5), (1.7), (1.10),
Theorem 1.8: with n pairs of x in R^p and y in R^q (c1 = p/n, c2 = q/n), a shared component with population squared
canonical correlation r gives a sample outlier (asymptotically, Gaussian data) iff r > rc(c1, c2); its sample value then
tends to gamma(r), otherwise to the null edge. For p = q = d this needs n > n* = d (1 + 1/r). The theorem is about
sample canonical correlations, not a recovery guarantee for the subspace; held-out correlation is reported beside it.

Test on CT-RATE: image and report embeddings projected on their own top-d principal components (fitted on all cases,
unpaired). Population values estimated once on the "sel" patients (no other use of them), predictions checked on
subsamples of the "train" patients.

    python -m align.spiked --emb DIR --out DIR --encoder E
    python -m align.spiked --out DIR --summary
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from align.align import svd

MODELS = ("0.8B", "2B", "4B", "9B", "27B")
LAYERS = {"mid": "textmid", "last": "text"}
DIMS = (16, 32, 64, 128)
NS = (100, 200, 400, 800, 1600)
DRAWS = 20
PERMS = 100
TOP = 3
ENCODERS = ("voxelfm", "voxelfm_patch", "ctfm", "vjepa2_l", "vjepa2_h", "vjepa2_g", "wan_vae", "wan_dit", "ltx_vae",
            "dinov3_2d", "dinov3_25d", "dalect0", "dalect1s", "videomae_b", "dinov2_2d")


def check(c1, c2):
    if not (c1 > 0 and c2 > 0 and c1 + c2 < 1):
        raise ValueError(f"need 0 < c1, c2 and c1 + c2 < 1, got {c1}, {c2}")


def rc(c1, c2):
    check(c1, c2)
    return float(np.sqrt(c1 * c2 / ((1 - c1) * (1 - c2))))


def edge(c1, c2):
    check(c1, c2)
    return float((np.sqrt(c1 * (1 - c2)) + np.sqrt(c2 * (1 - c1))) ** 2)


def gamma(r, c1, c2):
    return r * (1 - c1 + c1 / r) * (1 - c2 + c2 / r)


def invert_gamma(lam, c1, c2):
    lo, e = rc(c1, c2), edge(c1, c2)
    if not (e < lam <= 1):
        raise ValueError(f"{lam} is not an outlier above the edge {e}")
    return 1.0 if lam == 1 else brentq(lambda r: gamma(r, c1, c2) - lam, lo, 1.0)


def predict(r, c1, c2):
    t, e = rc(c1, c2), edge(c1, c2)
    return np.sort([gamma(v, c1, c2) if v > t else e for v in np.asarray(r, float)])[::-1]


def budget(r, d):
    return d * (1 + 1 / r)


def sample_cc2(x, y):
    """Squared sample canonical correlations, descending: squared singular values of Qx' Qy (thin QR of the centred
    data), which are the eigenvalues of Sxx^-1 Sxy Syy^-1 Syx."""
    qx = np.linalg.qr(x - x.mean(0))[0]
    qy = np.linalg.qr(y - y.mean(0))[0]
    return np.clip(svd(qx.T @ qy, compute_uv=False) ** 2, 0, 1)


def simulate(r, p, q, n, rng):
    r = np.asarray(r, float)
    x = rng.standard_normal((n, p))
    y = rng.standard_normal((n, q))
    k = len(r)
    y[:, :k] = np.sqrt(r) * x[:, :k] + np.sqrt(1 - r) * y[:, :k]
    return x, y


def pcs(z, d):
    """Top-d principal coordinates of z (fitted on all its rows; no pairing involved)."""
    zc = z - z.mean(0)
    return zc @ svd(zc, full_matrices=False)[2][:d].T


def qr_centred(z):
    zc = z - z.mean(0)
    return np.linalg.qr(zc)


def top_null(qx, qy, rng, perms):
    """Top squared canonical correlation after shuffling the pairing; QR commutes with the row permutation."""
    return np.array([svd(qx.T @ qy[rng.permutation(len(qy))], compute_uv=False)[0] ** 2 for _ in range(perms)])


def n_significant(lam, null, alpha=0.05):
    """Leading values with permutation p = (1 + #{null >= value}) / (len(null) + 1) <= alpha, counted in order."""
    k = 0
    for v in lam:
        if (1 + (null >= v).sum()) / (len(null) + 1) > alpha:
            break
        k += 1
    return k


def estimate(x, y, rng):
    """Population squared canonical correlations of the leading components that are significant against the
    permutation null and above the null edge; the others are below the threshold at this n and are returned as 0."""
    n, p = x.shape
    q = y.shape[1]
    (qx, _), (qy, _) = qr_centred(x), qr_centred(y)
    lam = np.clip(svd(qx.T @ qy, compute_uv=False) ** 2, 0, 1)
    null = top_null(qx, qy, rng, PERMS)
    k = n_significant(lam, null)
    r = np.zeros(len(lam))
    for i in range(k):
        if lam[i] > edge(p / n, q / n):
            r[i] = invert_gamma(lam[i], p / n, q / n)
    return r, lam, float(np.percentile(null, 95))


def draw_stats(x, y, rng, x_test=None, y_test=None):
    """Top squared sample canonical correlations, the count significant against 20 permutations (which then needs
    the value above all of them), and the held-out correlation of the leading canonical pair."""
    mx, my = x.mean(0), y.mean(0)
    (qx, rx), (qy, ry) = qr_centred(x), qr_centred(y)
    u, sv, vt = svd(qx.T @ qy)
    lam = np.clip(sv ** 2, 0, 1)
    k = n_significant(lam, top_null(qx, qy, rng, 20))
    held = np.nan
    if x_test is not None:
        a = (x_test - mx) @ np.linalg.solve(rx, u[:, 0])
        b = (y_test - my) @ np.linalg.solve(ry, vt[0])
        held = abs(float(np.corrcoef(a, b)[0, 1]))
    return lam[:TOP], k, held


def run(a):
    """Splits: r_hat on a.est, subsamples from a.pool, held-out correlation on a.heldout. Volumes listed in
    <emb>/rejected (BIMCV) are dropped from both sides; the report embeddings cover every case of cases.jsonl."""
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    rejected = {os.path.basename(p)[:-5] for p in glob.glob(f"{a.emb}/rejected/*.json")}
    keep = ~cases.key.isin(rejected).to_numpy()
    cases = cases[keep].reset_index(drop=True)
    sel, train, test = (np.flatnonzero(cases.split == s) for s in (a.est, a.pool, a.heldout))
    img = np.stack([np.load(f"{a.emb}/{a.encoder}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64)
    dims, ns = [int(v) for v in a.dims.split(",")], [int(v) for v in a.ns.split(",")]
    img_pc = pcs(img, max(dims))
    rows, est = [], {}
    for model in a.models.split(","):
        for layer in a.layers.split(","):
            prefix = LAYERS[layer] if layer in LAYERS else f"text{layer}"   # L<k>: align.extract depth
            txt = np.load(f"{a.emb}/{prefix}_{model}.npy").astype(np.float64)
            if len(txt) != len(keep):
                raise ValueError(f"{prefix}_{model}: {len(txt)} rows for {len(keep)} cases")
            txt_pc = pcs(txt[keep], max(dims))
            for d in dims:
                x, y = img_pc[:, :d], txt_pc[:, :d]
                rng = np.random.default_rng(0)
                r_hat, lam_sel, cut_sel = estimate(x[sel], y[sel], rng)
                est[f"{model}/{layer}/{d}"] = {"r_hat": r_hat[r_hat > 0].tolist(), "lam_sel": lam_sel[:8].tolist(),
                                               "null95_sel": cut_sel, "n_sel": len(sel)}
                for n in ns:
                    if 2 * d >= n:
                        continue
                    c = d / n
                    pred = predict(r_hat, c, c)[:TOP]
                    obs, k_obs, held, sims, k_sim = [], [], [], [], []
                    for _ in range(DRAWS):
                        idx = rng.choice(train, n, replace=False)
                        o, k, h = draw_stats(x[idx], y[idx], rng, x[test], y[test])
                        obs.append(o)
                        k_obs.append(k)
                        held.append(h)
                        o, k, _ = draw_stats(*simulate(r_hat[r_hat > 0], d, d, n, rng), rng)
                        sims.append(o)
                        k_sim.append(k)
                    obs, sims = np.mean(obs, 0), np.mean(sims, 0)
                    rows.append({"encoder": a.encoder, "model": model, "layer": layer, "d": d, "n": n,
                                 "k_pred": int((r_hat > rc(c, c)).sum()), "k_obs": float(np.mean(k_obs)),
                                 "k_sim": float(np.mean(k_sim)), "heldout1": float(np.mean(held)),
                                 **{f"obs{i + 1}": float(obs[i]) for i in range(TOP)},
                                 **{f"pred{i + 1}": float(pred[i]) for i in range(TOP)},
                                 **{f"sim{i + 1}": float(sims[i]) for i in range(TOP)}})
                print(f"[spiked] {a.encoder} {model} {layer} d={d} r_hat={np.round(r_hat[r_hat > 0], 3).tolist()}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/{a.encoder}_cells.csv", index=False)
    json.dump(est, open(f"{a.out}/{a.encoder}_estimates.json", "w"), indent=1)


def summary(out):
    """Criterion T1: the closed-form limits describe real embeddings about as well as they describe
    Gaussian data with the same r_hat, d and n (every part of the criterion calibrated on that reference), and the
    test is informative only if enough series cross the threshold within the tested n."""
    df = pd.concat([pd.read_csv(f"{out}/{e}_cells.csv") for e in ENCODERS])
    err = np.abs(np.concatenate([df[f"obs{i}"] - df[f"pred{i}"] for i in range(1, TOP + 1)]))
    sim = np.abs(np.concatenate([df[f"sim{i}"] - df[f"pred{i}"] for i in range(1, TOP + 1)]))
    tol_med = max(0.03, 1.5 * float(np.median(sim)))
    tol_p90 = max(0.08, 1.5 * float(np.percentile(sim, 90)))
    k_ok = float(((df.k_obs - df.k_pred).abs() <= 1).mean())
    k_ok_sim = float(((df.k_sim - df.k_pred).abs() <= 1).mean())
    crossing = float(df.groupby(["encoder", "model", "layer", "d"]).k_pred.nunique().gt(1).mean())
    res = {"cells": len(df), "median_abs_err": float(np.median(err)), "p90_abs_err": float(np.percentile(err, 90)),
           "sim_median_abs_err": float(np.median(sim)), "sim_p90_abs_err": float(np.percentile(sim, 90)),
           "tol_median": tol_med, "tol_p90": tol_p90, "k_within_1": k_ok, "k_within_1_sim": k_ok_sim,
           "series_crossing_threshold": crossing, "informative": bool(crossing >= 0.2),
           "T1_pass": bool(crossing >= 0.2 and np.median(err) <= tol_med and np.percentile(err, 90) <= tol_p90
                           and k_ok >= k_ok_sim - 0.1),
           "by_layer": {l: float(np.median(np.abs(g.obs1 - g.pred1))) for l, g in df.groupby("layer")},
           "by_d": {int(d): float(np.median(np.abs(g.obs1 - g.pred1))) for d, g in df.groupby("d")}}
    json.dump(res, open(f"{out}/summary.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--est", default="sel")
    ap.add_argument("--pool", default="train")
    ap.add_argument("--heldout", default="test")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--layers", default=",".join(LAYERS))
    ap.add_argument("--dims", default=",".join(map(str, DIMS)))
    ap.add_argument("--ns", default=",".join(map(str, NS)))
    a = ap.parse_args()
    summary(a.out) if a.summary else run(a)
    print("SPIKED_EXIT rc=0", flush=True)


if __name__ == "__main__":
    main()
