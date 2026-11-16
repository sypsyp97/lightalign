"""Baselines for the pilot score S of align.pilot_curve: do simple similarity scores computed on the SAME pilot pairs pick
the best of the 15 CT encoders as well as the random-matrix score S?

Pilot rows, principal coordinates, S and the target (mean_auc at n = 1600) are those of align.pilot_curve. Scores per
(m, seed, encoder):
  S            budget law at n = 1600 on r_hat of align.spiked.estimate (recomputed; gated against pilot_curve_m<M>.csv)
  cc1_raw      top sample canonical correlation of the top-D principal coordinates of the pilot rows (no RMT correction)
  cc2sum_raw   sum of all D squared sample canonical correlations of the same coordinates
  cka_lin      align.align.linear_cka (unbiased HSIC) on the pilot rows of the paper-convention features
  mknn10       align.mknn.mutual_knn of align.mknn.knn neighbour sets (k = align.mknn.K = 10, cosine, row itself excluded)
               on the pilot rows of the same features
  Features of cka_lin and mknn10 (as in align.mknn): each train row to unit norm, minus the mean over ALL train rows,
  renormalised to unit norm.

run:     one pilot size m (array task), seeds 0 .. N-1. Output DIR/pilot_baselines_m<M>.csv, one row per (seed, encoder).
summary: all pilot sizes -> pilot_baselines_summary.csv, pilot_baselines_paired.csv, pilot_baselines_summary.json.

    python -m align.pilot_baselines run --emb DIR --downstream DIR --out DIR --pilot M --pilot-curve-dir DIR [--seeds N]
    python -m align.pilot_baselines summary --out DIR
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.stats import spearmanr

from align.align import linear_cka, normalize
from align.downstream import MODEL
from align.mknn import knn, mutual_knn
from align.planner import D, law
from align.pilot_curve import GAP_TOL, N_TARGET, PILOTS
from align.spiked import ENCODERS, estimate, pcs

SCORES = ("S", "cc1_raw", "cc2sum_raw", "cka_lin", "mknn10")
BASELINES = SCORES[1:]
BOOT = 10000


def canon_corr(x, y):
    """Sample canonical correlations (descending) of two column-centred-on-these-rows blocks."""
    qx = np.linalg.qr(x - x.mean(0))[0]
    qy = np.linalg.qr(y - y.mean(0))[0]
    return svd(qx.T @ qy, compute_uv=False)


def run(a):
    if a.pilot <= 2 * D:
        raise ValueError(f"pilot {a.pilot} must exceed 2 * D = {2 * D}")
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    train = np.flatnonzero(cases.split == "train")
    if a.pilot > len(train):
        raise ValueError(f"pilot {a.pilot} > {len(train)} train rows")
    y = np.load(f"{a.emb}/textmid_{MODEL}.npy").astype(np.float64)[train]
    y_pc = pcs(y, D)
    yn = normalize(y)
    yn = normalize(yn - yn.mean(0))
    curves = pd.read_csv(f"{a.downstream}/curves.csv")
    auc = curves[curves.n == N_TARGET].set_index("encoder").mean_auc
    perms = [np.random.default_rng(seed).permutation(len(train)) for seed in range(a.seeds)]
    rows = []
    for e in ENCODERS:
        x = np.stack([np.load(f"{a.emb}/{e}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64)[train]
        x_pc = pcs(x, D)
        xn = normalize(x)
        xn = normalize(xn - xn.mean(0))
        for seed in range(a.seeds):
            p = perms[seed][:a.pilot]
            r_hat = estimate(x_pc[p], y_pc[p], np.random.default_rng(seed + 4000))[0]
            r_hat = r_hat[r_hat > 0]
            s = law(r_hat, N_TARGET)[1]
            cc = canon_corr(x_pc[p], y_pc[p])
            rows.append({"pilot": a.pilot, "seed": seed, "encoder": e, "S": s, "cc1_raw": float(cc[0]),
                         "cc2sum_raw": float((cc ** 2).sum()), "cka_lin": linear_cka(xn[p], yn[p]),
                         "mknn10": mutual_knn(knn(xn[p]), knn(yn[p])), "auc": float(auc[e])})
        print(e, json.dumps(rows[-1]), flush=True)
    df = pd.DataFrame(rows)
    ref = pd.read_csv(f"{a.pilot_curve_dir}/pilot_curve_m{a.pilot}.csv").set_index(["seed", "encoder"]).S
    got = df.set_index(["seed", "encoder"]).S
    if set(ref.index) != set(got.index):
        raise ValueError(f"pilot_curve_m{a.pilot}.csv has {len(ref)} (seed, encoder) rows, recomputed {len(got)}; key sets differ")
    np.testing.assert_allclose(got.loc[ref.index].to_numpy(), ref.to_numpy(), rtol=1e-12, atol=0)
    print(f"GATE pilot_curve S reproduced m={a.pilot}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/pilot_baselines_m{a.pilot}.csv", index=False)


def seed_gaps(g, score):
    """Gap and Spearman rho of one score for one (m, seed): g has one row per encoder."""
    g = g.set_index("encoder").loc[list(ENCODERS)]
    picked = g[score].idxmax()
    return {"rho": float(spearmanr(g[score], g.auc).statistic), "picked": picked,
            "gap": float(g.auc.max() - g.auc[picked]), "picked_is_best": bool(picked == g.auc.idxmax())}


def summary(a):
    files = {int(os.path.basename(f)[len("pilot_baselines_m"):-len(".csv")]): f
             for f in glob.glob(f"{a.out}/pilot_baselines_m*.csv")}
    missing = sorted(set(PILOTS) - set(files))
    if missing:
        raise SystemExit(f"missing pilot sizes {missing}")
    df = pd.concat([pd.read_csv(files[m]) for m in PILOTS])
    seeds = pd.DataFrame([{"pilot": m, "seed": s, "score": sc, **seed_gaps(g, sc)}
                          for (m, s), g in df.groupby(["pilot", "seed"]) for sc in SCORES])
    rows = []
    for m in PILOTS:
        for sc in SCORES:
            g = seeds[(seeds.pilot == m) & (seeds.score == sc)]
            rows.append({"pilot": m, "score": sc, "seeds": len(g), "rho_median": float(g.rho.median()),
                         "rho_q25": float(g.rho.quantile(0.25)), "rho_q75": float(g.rho.quantile(0.75)),
                         "p_best": float(g.picked_is_best.mean()), "p_gap_le_0.005": float((g.gap <= GAP_TOL).mean()),
                         "gap_mean": float(g.gap.mean()), "gap_max": float(g.gap.max())})
    out = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    paired = []
    for m in PILOTS:
        gs = seeds[(seeds.pilot == m) & (seeds.score == "S")].set_index("seed").gap
        for sc in BASELINES:
            gb = seeds[(seeds.pilot == m) & (seeds.score == sc)].set_index("seed").gap.loc[gs.index]
            d = (gb - gs).to_numpy()
            boot = d[rng.integers(0, len(d), (BOOT, len(d)))].mean(1)
            paired.append({"pilot": m, "baseline": sc, "seeds": len(d), "frac_S_better": float((gs.to_numpy() < gb.to_numpy()).mean()),
                           "frac_equal": float((gs.to_numpy() == gb.to_numpy()).mean()),
                           "frac_S_worse": float((gs.to_numpy() > gb.to_numpy()).mean()),
                           "gap_diff_mean": float(d.mean()), "gap_diff_ci_lo": float(np.quantile(boot, 0.025)),
                           "gap_diff_ci_hi": float(np.quantile(boot, 0.975))})
    pr = pd.DataFrame(paired)
    out.to_csv(f"{a.out}/pilot_baselines_summary.csv", index=False)
    pr.to_csv(f"{a.out}/pilot_baselines_paired.csv", index=False)
    json.dump({"summary": rows, "paired": paired}, open(f"{a.out}/pilot_baselines_summary.json", "w"), indent=1)
    print(out.round(4).to_string(), flush=True)
    print(pr.round(4).to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "summary"])
    ap.add_argument("--emb")
    ap.add_argument("--downstream")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pilot", type=int)
    ap.add_argument("--pilot-curve-dir")
    ap.add_argument("--seeds", type=int, default=50)
    a = ap.parse_args()
    if a.step == "run":
        for f in ("emb", "downstream", "pilot", "pilot_curve_dir"):
            if vars(a)[f] is None:
                ap.error(f"run needs --{f.replace('_', '-')}")
        run(a)
    else:
        summary(a)
    print("PILOT_BASELINES_EXIT rc=0", flush=True)


if __name__ == "__main__":
    main()
