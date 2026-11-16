"""Pair-budget planner: can a small pilot of m true pairs, through
the budget law alone, (i) rank the encoders by their downstream result at a larger n and (ii) predict how many
shared directions become detectable at that n?

For each encoder and seed, m pilot pairs are drawn from the train patients (the first m of the align.downstream
permutation, so the pilot is a subset of the pairs behind the downstream curve). r_hat comes from align.spiked.estimate
on the pilot (top-d principal coordinates fitted on all train rows, no pairing involved); the law then gives
k(n) = #{r_hat > rc(d/n, d/n)} and S(n) = sum of those r_hat. Targets: align.downstream curves (mean AUC, FOSCTTM)
and the sel-patient estimates of align.spiked (2,024 pairs) as the reference r.

    python -m align.planner --emb DIR --downstream DIR --spiked DIR --out DIR
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from align.align import SEEDS
from align.downstream import MODEL
from align.spiked import ENCODERS, estimate, pcs, rc

D = 64
PILOTS = (200, 400)
TARGETS = (400, 800, 1600)


def law(r, n, d=D):
    vis = r > rc(d / n, d / n)
    return int(vis.sum()), float(r[vis].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--downstream", required=True)
    ap.add_argument("--spiked", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    train = np.flatnonzero(cases.split == "train")
    y = np.load(f"{a.emb}/textmid_{MODEL}.npy").astype(np.float64)
    y_pc = pcs(y[train], D)
    rows = []
    for e in ENCODERS:
        x = np.stack([np.load(f"{a.emb}/{e}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64)
        x_pc = pcs(x[train], D)
        ref = np.array(json.load(open(f"{a.spiked}/{e}_estimates.json"))[f"{MODEL}/mid/{D}"]["r_hat"])
        for seed in SEEDS:
            perm = np.random.default_rng(seed).permutation(len(train))     # positions in train, as align.downstream
            for m in PILOTS:
                r_hat = estimate(x_pc[perm[:m]], y_pc[perm[:m]], np.random.default_rng(seed + 4000))[0]
                r_hat = r_hat[r_hat > 0]
                for n in TARGETS:
                    k, s = law(r_hat, n)
                    k_ref, s_ref = law(ref, n)
                    rows.append({"encoder": e, "seed": seed, "pilot": m, "n": n, "k_pilot": k, "S_pilot": s,
                                 "k_ref": k_ref, "S_ref": s_ref, "r1_pilot": float(r_hat[0]) if len(r_hat) else 0.0})
        print(e, json.dumps(rows[-1]), flush=True)
    df = pd.DataFrame(rows)
    curves = pd.read_csv(f"{a.downstream}/curves.csv")[["encoder", "n", "mean_auc", "foscttm"]]
    mean = df.groupby(["encoder", "pilot", "n"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    mean = mean.merge(curves, on=["encoder", "n"])
    res = {}
    for (m, n), g in mean.groupby(["pilot", "n"]):
        res[f"pilot{m}_n{n}"] = {
            "rho_S_auc": float(spearmanr(g.S_pilot, g.mean_auc).statistic),
            "rho_S_foscttm": float(-spearmanr(g.S_pilot, g.foscttm).statistic),
            "rho_r1_auc": float(spearmanr(g.r1_pilot, g.mean_auc).statistic),
            "k_abs_err_median": float((g.k_pilot - g.k_ref).abs().median()),
            "best_encoder_pilot": g.loc[g.S_pilot.idxmax(), "encoder"],
            "best_encoder_auc": g.loc[g.mean_auc.idxmax(), "encoder"],
            "auc_loss_of_pick": float(g.mean_auc.max() - g.loc[g.S_pilot.idxmax(), "mean_auc"])}
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/planner.csv", index=False)
    mean.to_csv(f"{a.out}/planner_mean.csv", index=False)
    json.dump(res, open(f"{a.out}/planner.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)
    print("PLANNER_EXIT rc=0", flush=True)


if __name__ == "__main__":
    main()
