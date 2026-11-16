"""How Gaussian are the frozen features? The detection law (align.spiked) is derived for Gaussian
data, and a sign of a shared direction cannot be recovered from a sign-symmetric (e.g. Gaussian) distribution. So,
per embedding directory and feature set:
  - the skewness and excess kurtosis of each of the top-D principal components (all cases, centred), against a null of
    Gaussian samples with the same covariance and size (DRAWS draws): share of components outside the null's 99% band;
  - the skewness of the train projections on each oracle shared direction (align.budget.shared_directions on the sel
    split), image and report side, with the same Gaussian null.

    python -m align.gauss --emb DIR --image E [E ...] --text T --out CSV
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from align.budget import shared_directions
from align.align import normalize

D, DRAWS = 64, 200


def moments(z):
    return skew(z, axis=0), kurtosis(z, axis=0)          # excess kurtosis: 0 for a Gaussian


def null_band(n, k, rng):
    """99% band of |skew| and of excess kurtosis for n i.i.d. Gaussian samples (scale-free, so one band per n)."""
    s, q = [], []
    for _ in range(DRAWS):
        a, b = moments(rng.standard_normal((n, k)))
        s.append(np.abs(a))
        q.append(b)
    s, q = np.concatenate(s), np.concatenate(q)
    return float(np.quantile(s, 0.99)), float(np.quantile(q, 0.005)), float(np.quantile(q, 0.995))


def pcs(z, d):
    zc = z - z.mean(0)
    v = np.linalg.svd(zc, full_matrices=False)[2][:d].T
    return zc @ v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--image", nargs="+", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    sel, train = (np.flatnonzero(cases.split == s) for s in ("sel", "train"))
    y = np.load(f"{a.emb}/{a.text}.npy").astype(np.float64)    # raw, as align.spiked and the oracle directions
    rng = np.random.default_rng(0)
    band_all, band_train = null_band(len(cases), D, rng), null_band(len(train), 16, rng)
    rows = []
    for side, feats in [("text", {a.text: y})] + [("image", {e: None for e in a.image})]:
        for name in feats:
            z = y if side == "text" else np.stack(
                [np.load(f"{a.emb}/{name}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64)
            s, q = moments(pcs(z, D))                     # raw features: the view of the detection law
            sn, qn = moments(pcs(normalize(z), D))        # unit-norm features: the view of the aligner
            row = {"data": os.path.basename(a.emb.rstrip("/")), "side": side, "features": name, "n": len(z),
                   "pc_abs_skew_median": float(np.median(np.abs(s))), "pc_exkurt_median": float(np.median(q)),
                   "pc_frac_skew_out": float(np.mean(np.abs(s) > band_all[0])),
                   "pc_frac_kurt_out": float(np.mean((q < band_all[1]) | (q > band_all[2]))),
                   "pc_abs_skew_median_unit": float(np.median(np.abs(sn))), "pc_exkurt_median_unit": float(np.median(qn)),
                   "pc_frac_skew_out_unit": float(np.mean(np.abs(sn) > band_all[0])),
                   "null_abs_skew_99": band_all[0], "null_kurt_lo": band_all[1], "null_kurt_hi": band_all[2]}
            if side == "image":
                try:
                    ux, uy, k = shared_directions(z[sel], y[sel], np.random.default_rng(0))
                except ValueError:                     # no significant shared component
                    k = 0
                if k:
                    sx = skew((z[train] - z[train].mean(0)) @ ux, axis=0)
                    sy = skew((y[train] - y[train].mean(0)) @ uy, axis=0)
                    # per-direction skewness; null_train_abs_skew_99 is the 99% band of ONE direction's |skew| for
                    # n_train Gaussian samples, so compare direction by direction (n_out), not the max or the mean
                    row.update({"k": k, "shared_img_skew": json.dumps(np.round(sx, 4).tolist()),
                                "shared_txt_skew": json.dumps(np.round(sy, 4).tolist()),
                                "shared_img_n_out": int(np.sum(np.abs(sx) > band_train[0])),
                                "shared_txt_n_out": int(np.sum(np.abs(sy) > band_train[0])),
                                "null_train_abs_skew_99": band_train[0]})
                else:
                    row["k"] = 0
            rows.append(row)
            print(row, flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
