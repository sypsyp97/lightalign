"""Pair-free shared directions.

align.budget amplify (B3) amplifies each side along its shared canonical directions taken from CCA on true pairs.
Here each side finds its directions alone: the image side from its own TotalSegmentator measurements, the report side
from its own zero-shot prompt scores in the frozen LLM. Then the same amplification and zero-pair Gromov-Wasserstein.

    python -m align.pairfree --emb DIR --out DIR [--encoder voxelfm_patch]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from align.align import SEEDS, foscttm, gromov_wasserstein, label_names, normalize, svd
from align.anchor_zero import ANCHORS, image_anchors
from align.baselines import auc_matrix, zero_shot_scores
from align.budget import AMP_D, CURVE_MODEL, FACTORS, amplify_map, shared_directions
from align.spiked import qr_centred

K = 10
PRIMARY = 8


def guided_directions(z, s, k):
    """Orthonormal basis (in z's feature space) of the top-k canonical directions between z (top-AMP_D principal
    coordinates) and side signals s of the same rows (standardised, constant columns dropped, full rank kept)."""
    s = s[:, s.std(0) > 0]
    s = (s - s.mean(0)) / s.std(0)
    zc = z - z.mean(0)
    vz = svd(zc, full_matrices=False)[2][:AMP_D].T
    _, sv, vt = svd(s, full_matrices=False)
    vs = vt[sv > sv[0] * 1e-10].T                          # full non-zero rank of the side signals
    (qz, rz), (qs, _) = qr_centred(zc @ vz), qr_centred(s @ vs)
    u = svd(qz.T @ qs, full_matrices=False)[0]
    if k > u.shape[1]:
        raise ValueError(f"k = {k} > {u.shape[1]} canonical directions")
    return np.linalg.qr(vz @ np.linalg.solve(rz, u[:, :k]))[0]


def random_directions(dim, k, rng):
    return np.linalg.qr(rng.standard_normal((dim, k)))[0]


def overlap(u, v):
    """Mean squared cosine of the principal angles between span(u) and span(v), over the smaller rank."""
    return float((svd(u.T @ v, compute_uv=False) ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="voxelfm_patch")
    a = ap.parse_args()
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    names = label_names(a.emb)
    sel, train, test = (np.flatnonzero(cases.split == s) for s in ("sel", "train", "test"))
    x = np.stack([np.load(f"{a.emb}/{a.encoder}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64)
    y = np.load(f"{a.emb}/textmid_{CURVE_MODEL}.npy").astype(np.float64)
    prompts = np.load(f"{a.emb}/promptsmid_{CURVE_MODEL}.npy").astype(np.float64)
    lab = cases[names].to_numpy(float)[test]
    meas = np.log1p(pd.read_csv(f"{a.emb}/measures.csv").set_index("key").loc[cases["key"]].to_numpy(np.float64))
    anchor_meas = image_anchors(f"{a.emb}/measures.csv", cases["key"])
    idx = [names.index(n) for n in ANCHORS]
    ox, oy, k_oracle = shared_directions(x[sel], y[sel], np.random.default_rng(0))   # diagnostic only
    rows = []
    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(train)
        half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]
        scores = zero_shot_scores(normalize(y[half_b] - y[half_b].mean(0)), normalize(prompts - y[half_b].mean(0)))
        rng = np.random.default_rng(seed + 5000)
        arms = {"A_generic": (guided_directions(x[half_a], meas[half_a], K), guided_directions(y[half_b], scores, K)),
                "B_anchor": (guided_directions(x[half_a], anchor_meas[half_a], len(ANCHORS)),
                             guided_directions(y[half_b], scores[:, idx], len(ANCHORS))),
                "R_random": (random_directions(x.shape[1], K, rng), random_directions(y.shape[1], K, rng))}
        for arm, (ux, uy) in arms.items():
            diag = {"overlap_image": overlap(ux, ox), "overlap_report": overlap(uy, oy)}
            for factor in FACTORS:
                gx, gy = amplify_map(ux, x[half_a].mean(0), factor), amplify_map(uy, y[half_b].mean(0), factor)
                xa, ya, pa = normalize(gx(x)), normalize(gy(y)), normalize(gy(prompts))
                mx, my = xa[half_a].mean(0), ya[half_b].mean(0)
                fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
                w, gx, gy = gromov_wasserstein(fx(xa[half_a]), fy(ya[half_b]), seed)
                fx, fy = (lambda v, f=fx, g=gx: g(f(v))), (lambda v, f=fy, g=gy: g(f(v)))   # the aligner's own centering on top
                u = normalize(fx(xa[test]) @ w)
                rows.append({"encoder": a.encoder, "arm": arm, "seed": seed, "factor": factor, **diag,
                             "mean_auc": float(np.nanmean(auc_matrix(zero_shot_scores(u, fy(pa)), lab))),
                             "foscttm": foscttm(u @ fy(ya[test]).T)})
                print(json.dumps(rows[-1]), flush=True)
    df = pd.DataFrame(rows)
    s = df.groupby(["arm", "factor"])[["mean_auc", "foscttm", "overlap_image", "overlap_report"]].mean()
    p = s.xs(PRIMARY, level="factor")
    ok = lambda arm, auc, fos: bool(p.loc[arm, "mean_auc"] >= auc and p.loc[arm, "foscttm"] <= fos)
    res = {"encoder": a.encoder, "k_oracle": k_oracle, "primary_factor": PRIMARY,
           "by_arm_factor": s.reset_index().to_dict("records"),
           "PF1": bool(any(ok(arm, 0.60, 0.40) and p.loc[arm, "mean_auc"] >= p.loc["R_random", "mean_auc"] + 0.05
                           for arm in ("A_generic", "B_anchor"))),
           "PF2": bool(any(ok(arm, 0.69, 0.35) for arm in ("A_generic", "B_anchor")))}
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/{a.encoder}_pairfree.csv", index=False)
    json.dump(res, open(f"{a.out}/{a.encoder}_pairfree.json", "w"), indent=1)
    print(s.round(3).to_string())
    print(json.dumps({k: v for k, v in res.items() if k != "by_arm_factor"}, indent=1))
    print("PAIRFREE_EXIT rc=0", flush=True)


if __name__ == "__main__":
    main()
