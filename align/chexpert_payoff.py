"""Chest X-ray payoff (second modality): CheXpert Plus image-report pairs, scored on the official CheXpert test set.

For one image encoder E and one text model M, n pairs from the Plus "train" split fit the map image -> report (ridge;
n = 0: the official zero-pair aligner on two disjoint halves, as align.official_ablation), five seeds. Two readers:
P, the zero-shot prompts "<finding> is present." minus "No <finding>."; TC, one logistic classifier per finding trained
on every train-split report with its CheXbert label (report text only). Scores are AUROCs of the five competition
findings against radiologist labels, mean over the findings, with a study-level bootstrap 95% interval (1,000
resamples, the same for every cell).

    python -m align.chexpert_payoff run --train DIR --test DIR --out DIR --encoder E --model M --n N --eval test
        [--save-scores] [--base DIR --reader-pool base|all]
        --eval test | valid: the official sets (radiologist labels, emb_chexpert_test)
        --eval sel: the Plus "sel" split (CheXbert labels of the report), for the pre-registered choice of E x M
        --base DIR: --train is a scale-up of the cases of DIR. The pairs of DIR's train split come first in every seed's
        order (the same order as without --base), then the new ones, so smaller pair sets are nested in larger ones;
        n <= the base pool gives the same numbers as DIR. --reader-pool base (default): centring and the TC classifiers
        use the base pool only, as in DIR; all: the whole train split.
    python -m align.chexpert_payoff select --out DIR      pick of E x M: highest TC mean AUROC on "sel" at n = 200
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from align.align import SEEDS, gromov_wasserstein, normalize
from align.anchor_followup import ridge
from align.baselines import auc_matrix
from align.domains import COMPETITION
from align.extract import kept_cases

NS = (0, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 40000, 80000, -1)       # -1: every train-split pair
BOOT = 1000


def text(emb, model):
    """Report and prompt features: the middle layer of a Qwen3.5 model, the sentence embedding of mpnet."""
    tag = "" if model == "mpnet" else "mid"
    return (normalize(np.load(f"{emb}/text{tag}_{model}.npy").astype(np.float64)),
            normalize(np.load(f"{emb}/prompts{tag}_{model}.npy").astype(np.float64)))


def images(emb, cases, encoder):
    return normalize(np.stack([np.load(f"{emb}/{encoder}/{k}.npy") for k in cases["key"]]).astype(np.float64))


def run(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tc = pd.read_json(f"{a.train}/cases.jsonl", lines=True)
    labels = json.load(open(f"{a.train}/labels.json"))["labels"]
    missing = [t for t in COMPETITION if t not in labels]
    if missing:
        raise ValueError(f"competition findings without enough positives in {a.train}: {missing}")
    y, prompts = text(a.train, a.model)
    if len(y) != len(tc):
        raise ValueError(f"{len(y)} report features for {len(tc)} cases")
    keep = kept_cases(tc, a.train)                   # studies with an image that never decodes: no feature, no pair
    print(f"[payoff] {int((~keep).sum())} of {len(tc)} train cases excluded (undecodable image)", flush=True)
    y, tc = y[keep], tc[keep].reset_index(drop=True)
    xt = images(a.train, tc, a.encoder)
    pool = np.flatnonzero(tc["split"] == "train")
    n = len(pool) if a.n == -1 else a.n
    if a.base:
        old = pd.read_json(f"{a.base}/cases.jsonl", lines=True)
        old_train = old.loc[old["split"] == "train", "key"]
        base = np.flatnonzero((tc["split"] == "train").to_numpy() & tc["key"].isin(old_train).to_numpy())
        if len(base) != len(old_train):
            raise ValueError(f"{len(old_train) - len(base)} train cases of {a.base} are not in {a.train}")
        if n == 0:
            raise ValueError("--base does not support --n 0 (the zero-pair split would mix in the new pairs)")
    else:
        base = pool
    extra = np.setdiff1d(pool, base)
    rpool = base if a.reader_pool == "base" else pool
    if n > len(pool):
        raise ValueError(f"n = {n} > {len(pool)} train-split pairs")
    if a.eval == "sel":
        ec = tc[tc["split"] == "sel"].reset_index(drop=True)
        xe = xt[tc["split"].to_numpy() == "sel"]
    else:
        ec = pd.read_json(f"{a.test}/cases.jsonl", lines=True)
        keep = (ec["split"] == a.eval).to_numpy()
        ec = ec[keep].reset_index(drop=True)
        xe = images(a.test, ec, a.encoder)
    le = ec[list(COMPETITION)].to_numpy(float)
    pi = np.array([labels.index(t) for t in COMPETITION])
    seeds = SEEDS if n < len(pool) else [0]
    clfs = None
    per_seed = {"P": [], "TC": []}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = np.concatenate([rng.permutation(base), rng.permutation(extra)])
        if n == 0:                                   # unpaired: image half and report half share no study
            half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]
            mx, my = xt[half_a].mean(0), y[half_b].mean(0)
            fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
            w, gx, gy = gromov_wasserstein(fx(xt[half_a]), fy(y[half_b]), seed)
            fx, fy = (lambda v, f=fx, g=gx: g(f(v))), (lambda v, f=fy, g=gy: g(f(v)))
            u = normalize(fx(xe) @ w)
        else:
            mx, my = xt[rpool].mean(0), y[rpool].mean(0)
            fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
            u = fx(xe) @ ridge(fx(xt[perm[:n]]), fy(y[perm[:n]])).coef_.T
        pr = fy(prompts)
        if clfs is None or n == 0:                   # the report classifier depends on fy only through its centring
            clfs = [make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(
                fy(y[rpool]), tc.loc[rpool, t].to_numpy(float)) for t in COMPETITION]
        per_seed["P"].append(normalize(u) @ (pr[2 * pi] - pr[2 * pi + 1]).T)     # cosine, as CheXzero
        per_seed["TC"].append(np.stack([c.decision_function(u) for c in clfs], 1))
    if a.save_scores:                            # per-seed scores (seeds x studies x findings), for align.chexpert_thresh
        os.makedirs(a.out, exist_ok=True)
        np.savez(f"{a.out}/{a.encoder}_{a.model}_n{a.n}_{a.eval}_scores.npz", keys=ec["key"].to_numpy(str),
                 labels=le, **{r: np.stack(v) for r, v in per_seed.items()})
    rng = np.random.default_rng(0)
    res = [rng.integers(0, len(ec), len(ec)) for _ in range(BOOT)]
    rows = []
    for r, sc in per_seed.items():
        per_f = np.mean([auc_matrix(s, le) for s in sc], 0)
        draws = np.array([np.mean([np.nanmean(auc_matrix(s[i], le[i])) for s in sc]) for i in res])
        rows.append({"encoder": a.encoder, "model": a.model, "n": n, "eval": a.eval, "studies": len(ec), "reader": r,
                     "seeds": len(sc), "auc": float(np.nanmean(per_f)), "lo": float(np.percentile(draws, 2.5)),
                     "hi": float(np.percentile(draws, 97.5)), **{t: float(v) for t, v in zip(COMPETITION, per_f)}})
        print(json.dumps(rows[-1]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/{a.encoder}_{a.model}_n{a.n}_{a.eval}.csv", index=False)


def select(a):
    df = pd.concat([pd.read_csv(f) for f in glob.glob(f"{a.out}/*_n200_sel.csv")])
    tc = df[df.reader == "TC"].sort_values("auc", ascending=False)
    print(tc[["encoder", "model", "auc", "lo", "hi"]].to_string(index=False))
    best = tc.iloc[0]
    json.dump({"encoder": best.encoder, "model": str(best.model), "sel_auc": float(best.auc)},
              open(f"{a.out}/selected.json", "w"), indent=1)
    print(f"selected {best.encoder} x {best.model}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "select"])
    ap.add_argument("--train")
    ap.add_argument("--test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder")
    ap.add_argument("--model")
    ap.add_argument("--n", type=int, choices=NS)
    ap.add_argument("--eval", choices=["sel", "test", "valid"])
    ap.add_argument("--base", help="run: earlier cases directory the --train cases extend (see the module docstring)")
    ap.add_argument("--reader-pool", choices=["base", "all"], default="base")
    ap.add_argument("--save-scores", action="store_true", help="run: also save the per-seed scores (.npz)")
    a = ap.parse_args()
    {"run": run, "select": select}[a.step](a)


if __name__ == "__main__":
    main()
