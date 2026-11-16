"""Unpaired and few-pair alignment of VoxelFM image features to Qwen3.5 language-model report features.

Wasserstein Procrustes with a cluster-level CKA initialization (arXiv 2610.09411, Alg. 1 and 2) against
a random map, Procrustes on known pairs, and Gromov-Wasserstein. Writes results.json and the gate verdict.

    python -m align.align --emb DIR --out DIR [--workers N]
"""
import argparse
import functools
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy.linalg
from scipy.optimize import linear_sum_assignment, quadratic_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

TEXT_MODELS = ("0.8B", "2B")
ORGANS = ["liver", "pancreas", "kidney"]


def label_names(emb):
    """Zero-shot labels: labels.json of the embedding directory when present (CT-RATE), else the AbdomenAtlas organs."""
    path = f"{emb}/labels.json"
    return json.load(open(path))["labels"] if os.path.exists(path) else ORGANS
SEEDS = range(5)
C, S, R, QAP_FAQ, QAP_RANDOM = 30, 30, 100, 20, 100


def svd(a, **kw):
    """np.linalg.svd; LAPACK gesdd (divide and conquer) does not converge on some finite, well-scaled matrices
    (CXR-BERT ridge maps, 864 x 768), so a LinAlgError falls back to gesvd, which computes the same decomposition."""
    try:
        return np.linalg.svd(a, **kw)
    except np.linalg.LinAlgError:
        return scipy.linalg.svd(a, lapack_driver="gesvd", **kw)


def polar(n):
    u, _, vt = svd(n, full_matrices=False)
    return u @ vt


def normalize(z):
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def centered_kernel(a):
    k = a @ a.T
    return k - k.mean(0, keepdims=True) - k.mean(1, keepdims=True) + k.mean()


def match_clusters(a, b, ap, bp, rng):
    """Permutation of the rows of b maximizing linear CKA with a; known pairs (ap, bp) stay matched.

    The paper stresses that QAP solution quality dominates the result. FAQ alone stalls in local optima (on a
    known-isomorphic test it reached 0.74-0.77 of the true objective), so every FAQ solution is polished by 2-opt
    and 2-opt also runs from random starts; the best objective wins."""
    ka, kb = centered_kernel(np.vstack([a, ap])), centered_kernel(np.vstack([b, bp]))
    c, p = len(a), len(ap)
    fixed = np.stack([np.arange(c, c + p)] * 2, axis=1) if p else None
    starts = []
    for _ in range(QAP_FAQ):
        r = quadratic_assignment(ka, kb, method="faq",
                                 options=dict(maximize=True, partial_match=fixed, P0="randomized", rng=rng))
        starts.append(r.col_ind[:c])
    starts += [rng.permutation(c) for _ in range(QAP_RANDOM)]
    best = None
    for s in starts:
        guess = np.stack([np.arange(c), s], axis=1)
        r = quadratic_assignment(ka, kb, method="2opt",
                                 options=dict(maximize=True, partial_match=fixed, partial_guess=guess, rng=rng))
        if best is None or r.fun > best.fun:
            best = r
    perm = best.col_ind[:c]
    if perm.max() >= len(a):
        raise RuntimeError("cluster matched to a known pair")
    return perm


def assign(scores):
    rows, cols = linear_sum_assignment(scores, maximize=True)
    return rows, cols


def procrustes_step(x, y, rows, cols, xp, yp):
    n = x[rows].T @ y[cols]
    if len(xp):
        n = n + len(rows) / len(xp) * (xp.T @ yp)
    return polar(n)


def wasserstein_procrustes(x, y, xp, yp, seed, w0=None):
    rng = np.random.default_rng(seed)
    if w0 is None:
        m = np.zeros((x.shape[1], y.shape[1]))
        for s in range(S):
            ka = KMeans(C, n_init=1, random_state=int(rng.integers(1 << 31))).fit(x).cluster_centers_
            kb = KMeans(C, n_init=1, random_state=int(rng.integers(1 << 31))).fit(y).cluster_centers_
            perm = match_clusters(ka, kb, xp, yp, rng)
            m += ka.T @ kb[perm]
        m /= S * C
        rows, cols = assign(x @ m @ y.T)
        w = procrustes_step(x, y, rows, cols, xp, yp)
    else:
        w = w0
    prev = None
    for _ in range(R):
        rows, cols = assign(x @ w @ y.T)
        if prev is not None and np.array_equal(cols, prev):
            break
        prev = cols
        w = procrustes_step(x, y, rows, cols, xp, yp)
    return w


def gromov_wasserstein(x, y, seed):
    """Zero-pair map W [d_x, d_y] from the official aligner of Schnaus et al. (unpaired-rosetta @ fdfad84,
    WassersteinProcrustes with the authors' defaults C = 30, S = 30, b = 10,000, R = 100; needs pylibmgm).
    It replaces POT's Gromov-Wasserstein from the product coupling, which stalls in local optima on COCO features
    (exact-isometry control, the authors' Table 1 reports the same).
    The aligner centers and row-normalizes x and y again with their own means, and W lives in those coordinates; so it
    returns W with that preprocessing of each side (gx, gy), which every use of the map (test images, reports, prompts,
    report classifiers) must apply on top of the caller's own."""
    from unpaired_rosetta import WassersteinProcrustes

    return _official_map(WassersteinProcrustes(num_workers=1, num_threads=4), x, y, seed)


def mini_vec2vec_official(x, y, seed):
    """Zero-pair map of mini-vec2vec (Dar, arXiv 2510.02348) as implemented in the same repository
    (unpaired_rosetta.baselines.mini_vec2vec.MiniVec2Vec, its defaults); returns (W, gx, gy) as gromov_wasserstein."""
    from unpaired_rosetta.baselines.mini_vec2vec import MiniVec2Vec

    return _official_map(MiniVec2Vec(), x, y, seed)


def _official_map(aligner, x, y, seed):
    import torch

    aligner.fit(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32), randomness=int(seed))
    mx, my = (m.double().numpy()[0] for m in (aligner.mean_x, aligner.mean_y))
    return aligner.weight.double().numpy(), (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))


def foscttm(sim):
    """Fraction of other candidates closer than the true match; ties count one half, so a constant map gives 0.5."""
    if not np.isfinite(sim).all():
        raise ValueError("non-finite similarity")
    d = np.diag(sim)
    n = len(d)
    off = ~np.eye(n, dtype=bool)
    row = (((sim > d[:, None]) + 0.5 * (sim == d[:, None])) * off).sum(1) / (n - 1)
    col = (((sim > d[None, :]) + 0.5 * (sim == d[None, :])) * off).sum(0) / (n - 1)
    return float((row.mean() + col.mean()) / 2)


def unbiased_hsic(k, l):
    n = len(k)
    k, l = k.copy(), l.copy()
    np.fill_diagonal(k, 0)
    np.fill_diagonal(l, 0)
    one = np.ones(n)
    return ((k * l).sum() + (one @ k @ one) * (one @ l @ one) / ((n - 1) * (n - 2))
            - 2 / (n - 2) * (one @ k @ l @ one)) / (n * (n - 3))


def linear_cka(x, y):
    """Linear CKA with the unbiased HSIC estimator (as in the geometry scores of arXiv 2610.09411)."""
    k, l = x @ x.T, y @ y.T
    return float(unbiased_hsic(k, l) / np.sqrt(unbiased_hsic(k, k) * unbiased_hsic(l, l)))


def evaluate(w, xt, yt, prompts, labels):
    u = normalize(xt @ w)
    out = {"foscttm": foscttm(u @ yt.T)}
    for j, organ in enumerate(labels):
        score = u @ prompts[2 * j] - u @ prompts[2 * j + 1]
        y = labels[organ]
        out[f"auc_{organ}"] = float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan")
    return out


@functools.lru_cache(maxsize=1)
def load(emb, models=TEXT_MODELS, encoder="voxelfm", layer="last"):
    cases = pd.read_json(f"{emb}/cases.jsonl", lines=True)
    cases = cases[cases.split != "none"].reset_index(drop=True)
    # Row L2 here, then train-mean centering and row L2 again in run_cell (order of arXiv 2610.09411).
    x = normalize(np.stack([np.load(f"{emb}/{encoder}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64))
    sfx = {"last": "", "mid": "mid"}[layer]
    ys = {m: normalize(np.load(f"{emb}/text{sfx}_{m}.npy").astype(np.float64)) for m in models}
    ps = {m: normalize(np.load(f"{emb}/prompts{sfx}_{m}.npy").astype(np.float64)) for m in models}
    for m in models:
        if len(ys[m]) != len(cases) or len(ps[m]) != 2 * len(label_names(emb)):
            raise ValueError(f"text embeddings {m} do not match the case list")
    return cases, x, ys, ps


def whitener(z, dims):
    """PCA whitening fitted on one side's own (centered) training rows: top `dims` directions, unit variance.
    Pair-free, as in unsupervised word-embedding mapping (Artetxe et al. 2018)."""
    _, sv, vt = svd(z, full_matrices=False)
    return vt[:dims].T / (sv[:dims] / np.sqrt(len(z)))


def run_cell(cell):
    """One (text model, seed, method, k) cell; the result is cached as one JSON file so a rerun skips it."""
    emb, models, encoder, model, seed, method, k, out, whiten, layer = cell
    path = f"{out}/cells/{model}_{seed}_{method}_{k}.json"
    if os.path.exists(path):
        return json.load(open(path))
    cases, x, ys, ps = load(emb, models, encoder, layer)
    y = ys[model]
    train = np.flatnonzero(cases.split == "train")
    test = np.flatnonzero(cases.split == "test")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train)
    half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]
    mx, my = x[half_a].mean(0), y[half_b].mean(0)
    if whiten:      # each side whitened with its own half only; no pair enters
        px, py = whitener(x[half_a] - mx, whiten), whitener(y[half_b] - my, whiten)
    else:
        px, py = np.eye(x.shape[1]), np.eye(y.shape[1])
    fx = lambda v: normalize((v - mx) @ px)
    fy = lambda v: normalize((v - my) @ py)
    xa, yb = fx(x[half_a]), fy(y[half_b])
    k_used = min(k, len(half_a))           # small datasets (CT-RATE subset): at most the whole image half
    pairs = rng.choice(half_a, k_used, replace=False) if k else np.array([], dtype=int)
    xp, yp = fx(x[pairs]), fy(y[pairs])
    xt, yt = fx(x[test]), fy(y[test])
    prompts = fy(ps[model])
    labels = {o: cases[o].to_numpy()[test] for o in label_names(emb)}
    if method == "random":
        w = polar(rng.standard_normal((xa.shape[1], yb.shape[1])))
    elif method == "procrustes":
        w = polar(xp.T @ yp)
    elif method in ("gw", "gw+wp"):     # every use of the map moves into the aligner's own coordinates
        w, gx, gy = gromov_wasserstein(xa, yb, seed)
        xa, yb, xp, yp, xt, yt, prompts = gx(xa), gy(yb), gx(xp), gy(yp), gx(xt), gy(yt), gy(prompts)
        if method == "gw+wp":
            w = wasserstein_procrustes(xa, yb, xp, yp, seed, w0=w)
    elif method == "wp":
        w = wasserstein_procrustes(xa, yb, xp, yp, seed)
    else:
        raise ValueError(method)
    row = {"model": model, "seed": seed, "method": method, "k": k, "k_used": k_used, "whiten": whiten,
           **evaluate(w, xt, yt, prompts, labels)}
    print(json.dumps(row), flush=True)
    json.dump(row, open(path + ".tmp", "w"))
    os.replace(path + ".tmp", path)
    return row


def gate(df, cka, models):
    """Signal check; with more than two sizes G3 compares the smallest and the largest model and also
    reports the Spearman correlation of size with CKA and with zero-pair FOSCTTM."""
    from scipy.stats import spearmanr

    mean = df.groupby(["model", "method", "k"])["foscttm"].mean()
    wp0 = {m: mean[(m, "wp", 0)] for m in models}
    best = min(wp0, key=wp0.get)
    g1 = wp0[best] <= mean[(best, "random", 0)] - 0.05
    g2 = mean[(best, "wp", 100)] < mean[(best, "procrustes", 100)]
    lo, hi = models[0], models[-1]
    g3 = cka[lo] < cka[hi] or wp0[hi] < wp0[lo]
    size = [float(re.match(r"[\d.]+", m).group()) for m in models]       # "27B", "9B-impression" -> 27, 9
    trend = {"spearman_size_cka": spearmanr(size, [cka[m] for m in models])[0] if len(models) > 2 else None,
             "spearman_size_wp0": spearmanr(size, [wp0[m] for m in models])[0] if len(models) > 2 else None}
    return {"best_model": best, "G1": bool(g1), "G2": bool(g2), "G3": bool(g3),
            "wp0": wp0, "random": {m: mean[(m, "random", 0)] for m in models}, **trend,
            "wp100": mean[(best, "wp", 100)], "procrustes100": mean[(best, "procrustes", 100)],
            "procrustes1000": mean[(best, "procrustes", 1000)], "cka": cka}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models", default=",".join(TEXT_MODELS), help="text model names, smallest first")
    ap.add_argument("--encoder", default="voxelfm", help="image feature directory under --emb")
    ap.add_argument("--whiten", type=int, default=0, help="PCA-whiten each side to this many dims (0: off)")
    ap.add_argument("--layer", default="last", choices=["last", "mid"], help="LLM layer of the report and prompt embeddings")
    a = ap.parse_args()
    models = tuple(a.models.split(","))
    cases, x, ys, _ = load(a.emb, models, a.encoder, a.layer)
    test = np.flatnonzero(cases.split == "test")
    cka = {m: linear_cka(x[test], ys[m][test]) for m in models}
    print(f"[cka] {cka} n_train={int((cases.split == 'train').sum())} n_test={len(test)}", flush=True)
    methods = [("random", 0), ("procrustes", 10), ("procrustes", 100), ("procrustes", 1000), ("gw", 0),
               ("gw+wp", 0), ("wp", 0), ("wp", 10), ("wp", 100)]
    os.makedirs(f"{a.out}/cells", exist_ok=True)
    cells = [(a.emb, models, a.encoder, m, s, meth, k, a.out, a.whiten, a.layer) for m in models for s in SEEDS for meth, k in methods]
    # Expensive cells first, so the pool does not end on a long tail.
    cells.sort(key=lambda c: c[5] in ("random", "procrustes"))
    if a.workers == 1:
        rows = [run_cell(c) for c in cells]
    else:
        with ProcessPoolExecutor(a.workers) as pool:
            rows = list(pool.map(run_cell, cells))
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/cells.csv", index=False)
    verdict = gate(df, cka, models)
    print(f"[gate] {json.dumps(verdict, default=float)}", flush=True)
    summary = df.drop(columns=["seed", "whiten", "k_used"], errors="ignore").groupby(["model", "method", "k"]).agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    print(summary.to_string(), flush=True)
    summary.reset_index().to_csv(f"{a.out}/summary.csv", index=False)
    json.dump({"verdict": verdict}, open(f"{a.out}/results.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
