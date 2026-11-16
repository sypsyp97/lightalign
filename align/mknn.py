"""Local versus global convergence: on paired test items, mutual k-nearest-neighbour overlap (Huh et al.
2024, k = 10, cosine) and unbiased linear CKA between image and text features, each calibrated against a permutation
null (the text rows shuffled, 200 draws): calibrated score = (observed - null mean) / (1 - null mean) for mutual k-NN and
observed - null mean for CKA, with p = (draws at or above the observed value + 1) / (draws + 1).

    python -m align.mknn --emb DIR --image E [E ...] --text T [T ...] --out CSV
"""
import argparse
import os

import numpy as np
import pandas as pd

from align.align import linear_cka, normalize

K, DRAWS = 10, 200


def knn(z, k=K):
    s = z @ z.T
    np.fill_diagonal(s, -np.inf)
    return np.argpartition(-s, k, axis=1)[:, :k]


def mutual_knn(a, b):
    """Mean over items of |kNN_a(i) & kNN_b(i)| / k."""
    hit = np.zeros(a.shape, bool)
    for j in range(a.shape[1]):
        hit[:, j] = (a[:, j:j + 1] == b).any(1)
    return float(hit.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--image", nargs="+", required=True)
    ap.add_argument("--text", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    test = np.flatnonzero(cases.split == "test")
    rng = np.random.default_rng(0)
    perms = [rng.permutation(len(test)) for _ in range(DRAWS)]
    rows = []
    for t in a.text:
        y = normalize(np.load(f"{a.emb}/{t}.npy").astype(np.float64)[test])
        y = normalize(y - y.mean(0))
        ny = knn(y)
        for e in a.image:
            x = normalize(np.stack([np.load(f"{a.emb}/{e}/{os.path.basename(k)}.npy")
                                    for k in cases["key"].iloc[test]]).astype(np.float64))
            x = normalize(x - x.mean(0))
            nx = knn(x)
            m = mutual_knn(nx, ny)
            m0 = np.array([mutual_knn(nx, np.argsort(p)[ny[p]]) for p in perms])   # text neighbours of shuffled rows
            c = linear_cka(x, y)
            c0 = np.array([linear_cka(x, y[p]) for p in perms])
            rows.append({"data": os.path.basename(a.emb.rstrip("/")), "image": e, "text": t, "n": len(test),
                         "mknn": m, "mknn_null": float(m0.mean()), "mknn_cal": float((m - m0.mean()) / (1 - m0.mean())),
                         "mknn_p": float(((m0 >= m).sum() + 1) / (len(m0) + 1)), "cka": float(c), "cka_null": float(c0.mean()),
                         "cka_cal": float(c - c0.mean()), "cka_p": float(((c0 >= c).sum() + 1) / (len(c0) + 1))})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
