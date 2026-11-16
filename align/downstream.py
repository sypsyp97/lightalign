"""Downstream tasks against the number of true pairs.

run:     one encoder, Qwen3.5 4B middle layer. n = 0: zero-pair Gromov-Wasserstein on two disjoint halves of the train
         patients (as align.budget.amplify at factor 1). n > 0: ridge map from n random train pairs (as the random_alone
         arm of align.anchor_pairsel, but drawn from all train patients so that n reaches 1,600). 5 seeds. On the test
         patients: zero-shot classification of the 18 CT-RATE findings (mean AUC) and image -> report retrieval
         (FOSCTTM, recall at 1 / 5 / 10).
summary: the curves of all encoders next to the budget law of align.spiked (d = 64): predicted number of detectable
         shared components k(n) and their summed strength S(n), and the criteria D1 / D2.

    python -m align.downstream run --emb DIR --out DIR --encoder E
    python -m align.downstream summary --out DIR --spiked DIR
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from align.align import SEEDS, foscttm, gromov_wasserstein, label_names, normalize
from align.anchor_followup import ridge
from align.baselines import auc_matrix, recall_at, zero_shot_scores
from align.spiked import ENCODERS, rc

NS = (0, 25, 50, 100, 200, 400, 800, 1600)
MODEL, D = "4B", 64


def residual(scores):
    """Zero-shot scores minus their common axis: each finding's score regressed on the mean score over findings
    (no labels used). What is left separates a finding only if the map carries finding-specific information."""
    m = scores.mean(1, keepdims=True)
    mc = m - m.mean()
    beta = ((scores - scores.mean(0)) * mc).sum(0) / (mc ** 2).sum()
    return scores - mc * beta


def metrics(w, xt, yt, prompts, lab):
    u = normalize(xt @ w)
    sim = u @ yt.T
    s = zero_shot_scores(u, prompts)
    return {"mean_auc": float(np.nanmean(auc_matrix(s, lab))), "auc_resid": float(np.nanmean(auc_matrix(residual(s), lab))),
            "foscttm": foscttm(sim), **recall_at(sim)}


def run(a, layer="mid", model=MODEL):
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    names = label_names(a.emb)
    train, test = (np.flatnonzero(cases.split == s) for s in ("train", "test"))
    if len(train) < max(NS):
        raise ValueError(f"{len(train)} train patients < n = {max(NS)}")
    x = normalize(np.stack([np.load(f"{a.emb}/{a.encoder}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64))
    if layer in ("mid", "last"):
        text, prompt = {"mid": ("textmid", "promptsmid"), "last": ("text", "prompts")}[layer]
    elif re.fullmatch(r"L\d+", layer):
        text, prompt = f"text{layer}", f"prompts{layer}"
    else:
        raise ValueError(f"layer {layer}: mid, last or L<k>")
    y = normalize(np.load(f"{a.emb}/{text}_{model}.npy").astype(np.float64))
    prompts = normalize(np.load(f"{a.emb}/{prompt}_{model}.npy").astype(np.float64))
    lab = cases[names].to_numpy(float)[test]
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(train)
        for n in NS:
            if n == 0:      # unpaired: the image half and the report half share no patient
                half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]
                mx, my = x[half_a].mean(0), y[half_b].mean(0)
                fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
                w, gx, gy = gromov_wasserstein(fx(x[half_a]), fy(y[half_b]), seed)
                fx, fy = (lambda v, f=fx, g=gx: g(f(v))), (lambda v, f=fy, g=gy: g(f(v)))   # the aligner's own centering on top
            else:           # the first n of one permutation per seed: nested pair sets, so the curve is paired in n
                s = perm[:n]
                mx, my = x[train].mean(0), y[train].mean(0)
                fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
                w = ridge(fx(x[s]), fy(y[s])).coef_.T
            rows.append({"encoder": a.encoder, "layer": layer, "model": model, "seed": seed, "n": n,
                         **metrics(w, fx(x[test]), fy(y[test]), fy(prompts), lab)})
            print(json.dumps(rows[-1]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    tag = ("" if layer == "mid" else f"_{layer}") + ("" if model == MODEL else f"_{model}")
    pd.DataFrame(rows).to_csv(f"{a.out}/{a.encoder}_curve{tag}.csv", index=False)


def summary(a):
    df = pd.concat([pd.read_csv(f"{a.out}/{e}_curve.csv") for e in ENCODERS])
    mean = df.groupby(["encoder", "n"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    law = []
    for e in ENCODERS:
        r = np.array(json.load(open(f"{a.spiked}/{e}_estimates.json"))[f"{MODEL}/mid/{D}"]["r_hat"])
        for n in NS:
            vis = r > rc(D / n, D / n) if n > 2 * D else np.zeros(len(r), bool)
            law.append({"encoder": e, "n": n, "k_pred": int(vis.sum()), "S": float(r[vis].sum()), "r1": float(r[0]) if len(r) else 0.0})
    mean = mean.merge(pd.DataFrame(law), on=["encoder", "n"])
    rho = lambda g, u, v: float(spearmanr(g[u], g[v]).statistic)
    # D1: within each encoder, both tasks improve with n among n > 0 (Spearman over the 7 pair counts)
    d1 = {e: {"auc": rho(g[g.n > 0], "n", "mean_auc"), "foscttm": -rho(g[g.n > 0], "n", "foscttm")}
          for e, g in mean.groupby("encoder")}
    # D2: across encoders at each n > 2d, the law's S(n) ranks the downstream results
    d2 = {int(n): {"auc": rho(g, "S", "mean_auc"), "foscttm": -rho(g, "S", "foscttm")}
          for n, g in mean[mean.n > 2 * D].groupby("n")}
    # D3: finding-specific information (common axis removed) grows with pairs and follows the law
    d3_n = {e: rho(g[g.n > 0], "n", "auc_resid") for e, g in mean.groupby("encoder")}
    d3_s = {int(n): rho(g, "S", "auc_resid") for n, g in mean[mean.n > 2 * D].groupby("n")}
    res = {"D1": d1, "D1_pass": bool(all(v["auc"] >= 0.9 and v["foscttm"] >= 0.9 for v in d1.values())),
           "D2": d2, "D2_pass": bool(np.median([v["auc"] for v in d2.values()]) >= 0.7
                                     and np.median([v["foscttm"] for v in d2.values()]) >= 0.7),
           "D3_n": d3_n, "D3_S": d3_s,
           "D3_pass": bool(np.median(list(d3_n.values())) >= 0.7 and np.median(list(d3_s.values())) >= 0.5)}
    mean.to_csv(f"{a.out}/curves.csv", index=False)
    json.dump(res, open(f"{a.out}/summary.json", "w"), indent=1)
    print(mean.round(3).to_string())
    print(json.dumps(res, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "summary"])
    ap.add_argument("--emb")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder")
    ap.add_argument("--spiked")
    ap.add_argument("--model", default=MODEL, help="text model name of the embedding files (text<layer>_<model>.npy)")
    ap.add_argument("--layer", default="mid", help="LLM layer of the report embeddings: mid, last or L<k> (extract depth)")
    a = ap.parse_args()
    run(a, a.layer, a.model) if a.step == "run" else summary(a)
    print("DOWNSTREAM_EXIT rc=0", flush=True)


if __name__ == "__main__":
    main()
