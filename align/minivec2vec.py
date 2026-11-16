"""Zero-pair relational baseline: mini-vec2vec (Dar, arXiv 2510.02348; author notebook linear_vec2vec.ipynb at
github.com/guy-dar/mini-vec2vec, settings copied from it).

Approximate matching: 30 rounds of k-means (20 clusters) on each side, centroid kernels matched by QAP (2-opt, best of
30 random starts), relative representations against the matched centroids, each image matched to the mean of its 50
nearest reports in that representation; orthogonal Procrustes; refinement 1: 100 steps of 1,000 sampled images to the
mean of their 50 nearest reports, W <- (W + W_new) / 2; refinement 2: one k-means round with 500 clusters.
Deviation from the notebook: the orthogonal map needs equal dimensions, so each side first goes to its own top
d = 64 principal coordinates (fitted on its own half; the paper and the notebook align equal-width text embeddings).
Same halves, centring and test evaluation as the n = 0 arm of align.downstream.

    python -m align.minivec2vec --emb DIR --out DIR --encoder E
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import quadratic_assignment
from sklearn.cluster import KMeans

from align.align import SEEDS, label_names, normalize, svd
from align.downstream import MODEL, metrics

D = 64
ROUNDS, CLUSTERS, QAP_RUNS, K = 30, 20, 30, 50
REFINE_STEPS, REFINE_SAMPLE, REFINE_CLUSTERS = 100, 1000, 500


def centred_kernel(c):
    h = np.eye(len(c)) - 1 / len(c)
    return h @ c @ c.T @ h


def aligned_centroids(xa, yb, rng):
    ca = KMeans(CLUSTERS, random_state=int(rng.integers(2 ** 31))).fit(xa).cluster_centers_
    cb = KMeans(CLUSTERS, random_state=int(rng.integers(2 ** 31))).fit(yb).cluster_centers_
    ka, kb = centred_kernel(ca), centred_kernel(cb)
    best = None
    for _ in range(QAP_RUNS):
        q = quadratic_assignment(ka, kb, method="2opt",
                                 options={"maximize": True, "rng": int(rng.integers(2 ** 31))})
        if best is None or q.fun > best.fun:
            best = q
    return ca, cb[best.col_ind]


def mini_vec2vec(xa, yb, rng):
    """xa, yb: unit rows of equal width, no shared patient. Returns W with xa @ W ~ matched reports."""
    cs = [aligned_centroids(xa, yb, rng) for _ in range(ROUNDS)]
    ra = normalize(xa) @ normalize(np.concatenate([c[0] for c in cs])).T
    rb = normalize(yb) @ normalize(np.concatenate([c[1] for c in cs])).T
    near = np.argsort(-(normalize(ra) @ normalize(rb).T), 1)[:, :K]
    w = orthogonal_procrustes(xa, yb[near].mean(1))[0]
    for _ in range(REFINE_STEPS):
        s = xa[rng.permutation(len(xa))[:REFINE_SAMPLE]]
        near = np.argsort(-(normalize(s @ w) @ yb.T), 1)[:, :K]
        w = 0.5 * w + 0.5 * orthogonal_procrustes(s, yb[near].mean(1))[0]
    k1 = KMeans(REFINE_CLUSTERS, random_state=int(rng.integers(2 ** 31))).fit(xa)
    k2 = KMeans(REFINE_CLUSTERS, init=k1.cluster_centers_ @ w, n_init=1).fit(yb)
    return 0.5 * w + 0.5 * orthogonal_procrustes(k1.cluster_centers_, k2.cluster_centers_)[0]


def basis(z):
    return svd(z - z.mean(0), full_matrices=False)[2][:D].T


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
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(train)       # the halves of the n = 0 arm of align.downstream
        half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]
        mx, my = x[half_a].mean(0), y[half_b].mean(0)
        fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
        vx, vy = basis(fx(x[half_a])), basis(fy(y[half_b]))
        w = mini_vec2vec(normalize(fx(x[half_a]) @ vx), normalize(fy(y[half_b]) @ vy), np.random.default_rng(seed + 3000))
        rows.append({"encoder": a.encoder, "seed": seed, "method": "mini_vec2vec",
                     **metrics(vx @ w @ vy.T, fx(x[test]), fy(y[test]), fy(prompts), lab)})
        print(json.dumps(rows[-1]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/{a.encoder}_minivec2vec.csv", index=False)
    print("MINIVEC2VEC_EXIT rc=0", flush=True)


if __name__ == "__main__":
    main()
