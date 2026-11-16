"""Pilot-size curve of the pair-budget planner: how many pilot pairs m are needed for the
pilot score S to pick the best of the 15 CT encoders?

Same machinery as align.planner: top-64 principal coordinates fitted on all train rows, the first m of the
np.random.default_rng(seed).permutation(len(train)) as pilot, r_hat = align.spiked.estimate(...) with
default_rng(seed + 4000), r_hat > 0 kept, S = the budget law at n = 1600. Target per encoder = mean_auc at n = 1600 of
align.downstream curves.csv. Per (m, seed): Spearman rho(S, auc) over the encoders, the picked encoder (argmax S), the
AUC gap between the best encoder and the picked one.

run:     one pilot size m (array task), seeds 0 .. N-1. Output DIR/pilot_curve_m<M>.csv, one row per (seed, encoder).
         With m = 200 the S of seeds 0-4 is checked against the rows (pilot = 200, n = 1600) of planner.csv.
summary: all pilot sizes -> DIR/pilot_curve_seeds.csv, pilot_curve_summary.csv, pilot_curve_summary.json.

    python -m align.pilot_curve run --emb DIR --downstream DIR --out DIR --pilot M --planner-csv CSV [--seeds N]
    python -m align.pilot_curve summary --out DIR
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from align.downstream import MODEL
from align.planner import D, law
from align.spiked import ENCODERS, estimate, pcs

N_TARGET = 1600
PILOTS = (150, 200, 300, 400, 600, 800, 1600)
GAP_TOL = 0.005


def run(a):
    if a.pilot <= 2 * D:
        raise ValueError(f"pilot {a.pilot} must exceed 2 * D = {2 * D}")
    if a.pilot == 200 and a.seeds < 5:
        raise ValueError("the planner.csv check of m = 200 needs seeds >= 5")
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    train = np.flatnonzero(cases.split == "train")
    if a.pilot > len(train):
        raise ValueError(f"pilot {a.pilot} > {len(train)} train rows")
    y = np.load(f"{a.emb}/textmid_{MODEL}.npy").astype(np.float64)
    y_pc = pcs(y[train], D)
    curves = pd.read_csv(f"{a.downstream}/curves.csv")
    auc = curves[curves.n == N_TARGET].set_index("encoder").mean_auc
    rows = []
    for e in ENCODERS:
        x = np.stack([np.load(f"{a.emb}/{e}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64)
        x_pc = pcs(x[train], D)
        for seed in range(a.seeds):
            perm = np.random.default_rng(seed).permutation(len(train))
            r_hat = estimate(x_pc[perm[:a.pilot]], y_pc[perm[:a.pilot]], np.random.default_rng(seed + 4000))[0]
            r_hat = r_hat[r_hat > 0]
            k, s = law(r_hat, N_TARGET)
            rows.append({"pilot": a.pilot, "seed": seed, "encoder": e, "k": k, "S": s,
                         "r1": float(r_hat[0]) if len(r_hat) else 0.0, "auc": float(auc[e])})
        print(e, json.dumps(rows[-1]), flush=True)
    df = pd.DataFrame(rows)
    if a.pilot == 200:
        ref = pd.read_csv(a.planner_csv)
        ref = ref[(ref.pilot == 200) & (ref.n == N_TARGET) & (ref.seed < 5)].set_index(["encoder", "seed"]).S_pilot
        got = df[df.seed < 5].set_index(["encoder", "seed"]).S
        if len(ref) != len(ENCODERS) * 5 or set(ref.index) != set(got.index):
            raise ValueError(f"planner.csv has {len(ref)} rows for (pilot 200, n {N_TARGET}, seeds 0-4); need {len(ENCODERS) * 5}")
        np.testing.assert_allclose(got.loc[ref.index].to_numpy(), ref.to_numpy(), rtol=1e-12, atol=0)
        print("GATE planner.csv S_pilot reproduced (m=200, seeds 0-4, rtol 1e-12)", flush=True)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/pilot_curve_m{a.pilot}.csv", index=False)


def seed_stats(g):
    g = g.set_index("encoder").loc[list(ENCODERS)]
    picked = g.S.idxmax()
    return {"rho": float(spearmanr(g.S, g.auc).statistic), "picked": picked,
            "gap": float(g.auc.max() - g.auc[picked]), "picked_is_best": bool(picked == g.auc.idxmax())}


def summary(a):
    files = {int(os.path.basename(f)[len("pilot_curve_m"):-len(".csv")]): f for f in glob.glob(f"{a.out}/pilot_curve_m*.csv")}
    missing = sorted(set(PILOTS) - set(files))
    if missing:
        raise SystemExit(f"missing pilot sizes {missing}")
    df = pd.concat([pd.read_csv(files[m]) for m in PILOTS])
    seeds = pd.DataFrame([{"pilot": m, "seed": s, **seed_stats(g)} for (m, s), g in df.groupby(["pilot", "seed"])])
    seeds.to_csv(f"{a.out}/pilot_curve_seeds.csv", index=False)
    rows, counts = [], {}
    for m, g in seeds.groupby("pilot"):
        counts[int(m)] = {e: int(c) for e, c in g.picked.value_counts().items()}
        rows.append({"pilot": int(m), "seeds": len(g), "p_best": float(g.picked_is_best.mean()),
                     "p_gap_le_0.005": float((g.gap <= GAP_TOL).mean()), "rho_median": float(g.rho.median()),
                     "rho_q25": float(g.rho.quantile(0.25)), "rho_q75": float(g.rho.quantile(0.75)),
                     "gap_mean": float(g.gap.mean()), "gap_max": float(g.gap.max()),
                     "k_median": float(df[df.pilot == m].k.median())})
    out = pd.DataFrame(rows)
    out.to_csv(f"{a.out}/pilot_curve_summary.csv", index=False)
    json.dump({"summary": rows, "picked_counts": counts}, open(f"{a.out}/pilot_curve_summary.json", "w"), indent=1)
    print(out.round(4).to_string(), flush=True)
    print(json.dumps(counts, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "summary"])
    ap.add_argument("--emb")
    ap.add_argument("--downstream")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pilot", type=int)
    ap.add_argument("--planner-csv")
    ap.add_argument("--seeds", type=int, default=50)
    a = ap.parse_args()
    if a.step == "run":
        for f in ("emb", "downstream", "pilot", "planner_csv"):
            if vars(a)[f] is None:
                ap.error(f"run needs --{f.replace('_', '-')}")
        run(a)
    else:
        summary(a)
    print("PILOT_CURVE_EXIT rc=0", flush=True)


if __name__ == "__main__":
    main()
