"""Pair ablation on the official CT-RATE validation units for one cell: readers P and TC
at n = 0 (zero-pair Gromov-Wasserstein map, as align.decisive readout-zero) and n = 25 ... 1,600 and 5,994 pairs (ridge
map, as align.official_matrix pool "all"), each against the released CT-CLIP on the same volumes with a paired
patient-clustered bootstrap (the resamples of align.official_matrix).

    python -m align.official_ablation run --train DIR --valid DIR --ctclip CSV --out DIR --encoder E --model M --n N
    python -m align.official_ablation summary --out DIR --encoder E --model M
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from align.align import SEEDS, gromov_wasserstein, label_names, normalize
from align.anchor_followup import ridge
from align.baselines import auc_matrix
from align.official import units
from align.official_matrix import BOOT, DROP16, patient_resamples

NS = (0, 25, 50, 100, 200, 400, 800, 1600, 5994)


def fit_map(xt, y, n, seed, pool):
    """The map of one seed on the CT-RATE train patients `pool`: n = 0 the zero-pair aligner on the unpaired halves of
    the permutation, n > 0 the ridge map on its first n patients. Returns (apply, fy): apply(xv, centre=None) maps
    unit-norm image features xv into the report space, centring them on `centre` (default: the centring of the fit, the
    pool mean for n > 0, the first half for n = 0); fy centres report-side vectors (prompts, report classifier input)."""
    perm = np.random.default_rng(seed).permutation(pool)
    if n == 0:                                           # unpaired: image half and report half share no patient
        half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]
        mx, my = xt[half_a].mean(0), y[half_b].mean(0)
        fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
        w, gx, gy = gromov_wasserstein(fx(xt[half_a]), fy(y[half_b]), seed)
        fy = (lambda v, f=fy, g=gy: g(f(v)))             # the aligner's own centering on top
        return (lambda xv, centre=None: normalize(gx(normalize(xv - (mx if centre is None else centre))) @ w)), fy
    mx, my = xt[pool].mean(0), y[pool].mean(0)           # as official_matrix: centring from the pool
    fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
    coef = ridge(fx(xt[perm[:n]]), fy(y[perm[:n]])).coef_.T
    return (lambda xv, centre=None: normalize(xv - (mx if centre is None else centre)) @ coef), fy


def report_classifiers(fy, y, lt, pool):
    """One report classifier per label (TC reader), trained on the centred report embeddings of the pool."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return [make_pipeline(StandardScaler(), LogisticRegression()).fit(fy(y[pool]), lt[pool, j]) for j in range(lt.shape[1])]


def pair_report_classifiers(fy, y, lt, rows):
    """TCn reader: as report_classifiers, fitted on the reports of the n pairs `rows` only. A finding with a single
    class among these reports cannot be fitted and gets None."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return [make_pipeline(StandardScaler(), LogisticRegression()).fit(fy(y[rows]), lt[rows, j])
            if 0 < lt[rows, j].sum() < len(rows) else None for j in range(lt.shape[1])]


def run(a):
    if a.pair_reports and a.n == 0:
        raise ValueError("--pair-reports needs n > 0: n = 0 has no pairs")
    if a.report_count is not None and not 0 < a.n <= a.report_count:
        raise ValueError(f"--report-count {a.report_count} needs 0 < n <= report count (n = {a.n})")
    names = label_names(a.train)
    assert names == label_names(a.valid)
    tc = pd.read_json(f"{a.train}/cases.jsonl", lines=True)
    vc = pd.read_json(f"{a.valid}/cases.jsonl", lines=True)
    load = lambda emb, cases: np.stack([np.load(f"{emb}/{a.encoder}/{os.path.basename(k)}.npy")
                                        for k in cases["key"]]).astype(np.float64)
    xt, xv = normalize(load(a.train, tc)), normalize(load(a.valid, vc))
    mid = "mid" if a.text_layer == "mid" else ""         # text_<M>.npy = last layer, textmid_<M>.npy = middle layer
    y = normalize(np.load(f"{a.train}/text{mid}_{a.model}.npy").astype(np.float64))
    prompts = normalize(np.load(f"{a.train}/prompts{mid}_{a.model}.npy").astype(np.float64))
    pool = np.arange(len(tc))                            # pool "all": every train-split patient
    if max(a.n, a.report_count or 0) > len(pool):
        raise ValueError(f"n = {a.n} or report count {a.report_count} > {len(pool)} pool patients")
    tag = (f"{a.encoder}_{a.model}" + ("" if a.text_layer == "mid" else "_last") + f"_n{a.n}"
           + ("" if a.report_count is None else f"_m{a.report_count}"))
    lt, lv = tc[names].to_numpy(float), vc[names].to_numpy(float)
    keep16 = np.array([l not in DROP16 for l in names])
    ct = pd.read_csv(a.ctclip, index_col="VolumeName").loc[[f"{i}.nii.gz" for i in vc["id"]], names].to_numpy(float)
    seeds = SEEDS if a.n < len(pool) else [0]           # n = 5,994 uses every patient: one fit
    per_seed = {"P": [], "TC": []}
    for seed in seeds:
        apply, fy = fit_map(xt, y, a.n, seed, pool)
        u = apply(xv)
        pr = fy(prompts)
        clfs = report_classifiers(fy, y, lt, pool)
        per_seed["P"].append(normalize(u) @ (pr[0::2] - pr[1::2]).T)     # cosine: ridge rows vary in length
        per_seed["TC"].append(np.stack([c.decision_function(u) for c in clfs], 1))
        # the permutation of fit_map: its first n patients are the pairs, so TCm holds their reports plus m - n more
        for reader, m in (("TCn", a.n),) * a.pair_reports + (("TCm", a.report_count),) * (a.report_count is not None):
            cm = pair_report_classifiers(fy, y, lt, np.random.default_rng(seed).permutation(pool)[:m])
            # an unfitted finding scores a constant: AUROC 0.5
            per_seed.setdefault(reader, []).append(
                np.stack([np.zeros(len(u)) if c is None else c.decision_function(u) for c in cm], 1))
            print(json.dumps({"n": a.n, "seed": int(seed), "reports": int(m),
                              f"{reader.lower()}_unfitted_findings": sum(c is None for c in cm)}), flush=True)
    if a.save_scores:                                    # per-seed scores (seeds x volumes x findings), for align.thresh_metrics
        os.makedirs(a.out, exist_ok=True)
        np.savez(f"{a.out}/{tag}_scores.npz", ids=vc["id"].to_numpy(),
                 **{r: np.stack(v) for r, v in per_seed.items()})
    patients = vc["id"].str.split("_").str[1].to_numpy()
    rows = []
    for un, mk in units(vc).items():
        idx = np.flatnonzero(mk)
        res = patient_resamples(patients[idx])
        for lname, cols in (("mean_auc", np.ones(len(names), bool)), ("mean_auc16", keep16)):
            draws = lambda sc: np.array([np.mean([np.nanmean(auc_matrix(s[idx][j], lv[idx][j])[cols]) for s in sc])
                                         for j in res])
            d_ct = draws([ct])
            for r in per_seed:
                point = float(np.mean([np.nanmean(auc_matrix(s[idx], lv[idx])[cols]) for s in per_seed[r]]))
                diff = draws(per_seed[r]) - d_ct
                rows.append({"encoder": a.encoder, "model": a.model, "n": a.n, "unit": un, "volumes": int(mk.sum()),
                             "metric": lname, "reader": r, "auc": point,
                             "ctclip": float(np.nanmean(auc_matrix(ct[idx], lv[idx])[cols])),
                             "minus_ctclip_lo": float(np.percentile(diff, 2.5)),
                             "minus_ctclip_hi": float(np.percentile(diff, 97.5)), "boot": BOOT})
                print(json.dumps(rows[-1]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/{tag}.csv", index=False)


def summary(a):
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{a.out}/{a.encoder}_{a.model}_n*.csv"))])
    missing = sorted(set(NS) - set(df.n))
    if missing:
        raise SystemExit(f"missing n = {missing}")
    df = df.sort_values(["unit", "metric", "reader", "n"])
    df.to_csv(f"{a.out}/{a.encoder}_{a.model}_ablation.csv", index=False)
    res = {}
    for (un, lname, r), g in df.groupby(["unit", "metric", "reader"]):
        better = g[g.minus_ctclip_lo > 0].n          # whole interval above zero
        no_diff = g[(g.minus_ctclip_lo <= 0) & (g.minus_ctclip_hi >= 0)].n   # interval includes zero: no significant difference
        res[f"{un}/{lname}/{r}"] = {"first_n_better": int(better.min()) if len(better) else None,
                                    "first_n_no_significant_difference": int(no_diff.min()) if len(no_diff) else None}
    json.dump(res, open(f"{a.out}/{a.encoder}_{a.model}_ablation_summary.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "summary"])
    ap.add_argument("--train")
    ap.add_argument("--valid")
    ap.add_argument("--ctclip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int)
    ap.add_argument("--save-scores", action="store_true", help="run: also save the per-seed scores (.npz)")
    ap.add_argument("--text-layer", choices=["mid", "last"], default="mid",
                    help="run: report and prompt embeddings from the middle (default) or last layer of the text model; "
                         "last adds _last to the output name")
    ap.add_argument("--pair-reports", action="store_true",
                    help="run: add reader TCn, the report classifier fitted on the reports of the n pairs only")
    ap.add_argument("--report-count", type=int, metavar="M",
                    help="run: add reader TCm, the report classifier fitted on M reports (those of the n pairs and "
                         "M - n further training reports); the output is named ..._n<N>_m<M>")
    a = ap.parse_args()
    {"run": run, "summary": summary}[a.step](a)


if __name__ == "__main__":
    main()
