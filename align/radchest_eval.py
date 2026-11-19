"""RAD-ChestCT as an external cohort of the CT-RATE pipeline. Every map and report classifier is fitted on the CT-RATE
train patients exactly as for the CT-RATE rows (align.official_ablation.fit_map / report_classifiers; align.external
for TRANSFER) and applied unchanged to the 3,630 RAD-ChestCT volumes (no split, no report). The 18 CT-RATE findings map
to the 16 RAD labels of align.radchest: "Mosaic attenuation pattern" is dropped and the score of "Arterial wall
calcification" is the maximum of the CT-RATE scores of "Arterial wall calcification" and "Coronary artery wall
calcification" (the CT-CLIP convention).

main: final cell (--encoder voxelfm_b11 --model 9B), reader TC, n = 0 (zero-pair aligner on the unpaired CT-RATE halves)
and every pair count of align.official_ablation.NS, the same seeds (n = 5,994: seed 0). Centring of the RAD image
features: "ctrate" (primary, the paper protocol: no RAD data enter the pipeline) keeps the centring of the CT-RATE
fit, "own" subtracts the mean over all RAD volumes.
transfer: the four maps of align.external.METHODS for one CT-RATE encoder (align.anchor_zero.fit_seed, same seeds,
halves and pairs), prompt reader, read on the RAD labels with at least align.external.MIN_POS positives and, as the
in-domain reference, on the CT-RATE test patients (merged labels). RAD has no report, so these BIMCV columns of
align.external do not exist: the report-side own centring (the prompts keep the CT-RATE half-B centring in both
centrings), FOSCTTM on the external cohort (needs paired reports), WITHIN (needs RAD anchors from segmentation
measures and reports to fit on), the bimcv_all zero-pair fit (unpaired RAD images and reports), and the E1-E3 criteria.

    python -m align.radchest_eval main --ctrate DIR --rad DIR --out DIR --encoder voxelfm_b11 --model 9B [--save-scores]
    python -m align.radchest_eval transfer --ctrate DIR --rad DIR --out DIR --encoder E --model 4B
    python -m align.radchest_eval summary --ctrate DIR --out DIR --encoder voxelfm_b11 --model 9B [--results results/radchest]

transfer reads the RAD feature directory voxelfm_b12 for the CT-RATE encoder "voxelfm" (align.radchest split-depth:
voxelfm_b12 equals the voxelfm feature).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from align.align import SEEDS, label_names, normalize
from align.anchor_zero import ANCHORS, fit_seed, image_anchors
from align.baselines import auc_matrix, zero_shot_scores
from align.external import ENCODERS, METHODS, MIN_POS, anchor_zero_draws, boot_mean_auc, images, interval, reports
from align.official_ablation import NS, fit_map, report_classifiers

MERGED = "Arterial wall calcification"
MERGED_FROM = ("Arterial wall calcification", "Coronary artery wall calcification")
DROPPED = "Mosaic attenuation pattern"
N_RAD = 3630
CENTRINGS = ("own", "ctrate")


def merge_columns(names, rnames):
    """Per RAD label, the CT-RATE column indices whose maximum is its score; every CT-RATE label but DROPPED and the
    merged pair must appear exactly once."""
    cols = [[names.index(n) for n in (MERGED_FROM if r == MERGED else (r,))] for r in rnames]
    used = sorted(j for c in cols for j in c)
    if used != sorted(set(range(len(names))) - {names.index(DROPPED)}):
        raise ValueError(f"RAD labels {rnames} do not cover the CT-RATE labels {names} without {DROPPED}")
    return cols


def merge_scores(sc, cols):
    return np.stack([sc[:, c].max(1) for c in cols], 1)


def rad_cases(rad, ctrate):
    rc = pd.read_json(f"{rad}/cases.jsonl", lines=True)
    if len(rc) != N_RAD or rc["key"].duplicated().any():
        raise ValueError(f"{rad}: {len(rc)} cases, expected {N_RAD} distinct volumes")
    rnames = label_names(rad)
    merge_columns(label_names(ctrate), rnames)
    return rc, rnames


def main_run(a):
    names = label_names(a.ctrate)
    tc = pd.read_json(f"{a.ctrate}/cases.jsonl", lines=True)
    rc, rnames = rad_cases(a.rad, a.ctrate)
    cols = merge_columns(names, rnames)
    xt, y = images(a.ctrate, a.encoder, tc["key"]), reports(a.ctrate, a.model, len(tc))
    xr = images(a.rad, a.encoder, rc["key"])
    lt, lr = tc[names].to_numpy(float), rc[rnames].to_numpy(float)
    pool = np.arange(len(tc))
    centre = {"own": xr.mean(0), "ctrate": None}
    os.makedirs(a.out, exist_ok=True)
    ridge_clfs = None                                    # fy of the ridge maps is the pool mean: one set serves every n > 0
    for n in NS:
        path = f"{a.out}/{a.encoder}_{a.model}_n{n}.csv"
        npz = f"{a.out}/{a.encoder}_{a.model}_n{n}_scores.npz"
        have_csv = os.path.exists(path)
        if have_csv and not (a.save_scores and not os.path.exists(npz)):
            continue
        rows, kept = [], {c: [] for c in CENTRINGS}
        for seed in (SEEDS if n < len(pool) else [0]):
            apply, fy = fit_map(xt, y, n, seed, pool)
            if n == 0:
                clfs = report_classifiers(fy, y, lt, pool)
            else:
                if ridge_clfs is None:
                    ridge_clfs = report_classifiers(fy, y, lt, pool)
                clfs = ridge_clfs
            for c in CENTRINGS:
                u = apply(xr, centre[c])
                sc = merge_scores(np.stack([m.decision_function(u) for m in clfs], 1), cols)
                kept[c].append(sc)
                auc = auc_matrix(sc, lr)
                rows.append({"encoder": a.encoder, "model": a.model, "n": n, "seed": seed, "unit": "radchest",
                             "volumes": len(rc), "metric": "mean_auc", "reader": "TC", "centring": c,
                             "auc": float(np.nanmean(auc)), **{f"auc|{l}": float(v) for l, v in zip(rnames, auc)}})
                print(json.dumps({k: v for k, v in rows[-1].items() if "|" not in k}), flush=True)
        if a.save_scores:                                # per-volume scores for align.rad_ci boot
            np.savez(f"{npz[:-4]}.tmp.npz", keys=rc["key"].to_numpy().astype(str),
                     **{f"TC_{c}": np.stack(v) for c, v in kept.items()})
            os.replace(f"{npz[:-4]}.tmp.npz", npz)
        if not have_csv:                                 # an existing csv is kept as it is
            pd.DataFrame(rows).to_csv(f"{path}.tmp", index=False)
            os.replace(f"{path}.tmp", path)


def rad_encoder(encoder):
    return "voxelfm_b12" if encoder == "voxelfm" else encoder


def transfer_run(a):
    names = label_names(a.ctrate)
    ct = pd.read_json(f"{a.ctrate}/cases.jsonl", lines=True)
    rc, rnames = rad_cases(a.rad, a.ctrate)
    x, y = images(a.ctrate, a.encoder, ct["key"]), reports(a.ctrate, a.model, len(ct))
    prompts = normalize(np.load(f"{a.ctrate}/promptsmid_{a.model}.npy").astype(np.float64))
    img_anchor = image_anchors(f"{a.ctrate}/measures.csv", ct["key"])
    idx = [names.index(n) for n in ANCHORS]
    train, test = (np.flatnonzero(ct.split == s) for s in ("train", "test"))
    rx = images(a.rad, rad_encoder(a.encoder), rc["key"])
    evals = [n for n in rnames if rc[n].sum() >= MIN_POS]
    cols = merge_columns(names, rnames)
    ev = [rnames.index(n) for n in evals]
    rlab = rc[evals].to_numpy(float)
    clab = np.stack([ct.loc[test, list(MERGED_FROM)].max(1) if n == MERGED else ct.loc[test, n] for n in evals], 1).astype(float)
    g_own = lambda v: normalize(v - rx.mean(0))

    sets = ("own", "ctrate", "ctrate_test")
    sc = {(c, m): [] for c in sets for m in METHODS}
    for seed in SEEDS:
        maps, fx, fy, pr = fit_seed(x, y, prompts, img_anchor, idx, train, seed, anchor_zero_draws(seed))
        for m, w in maps.items():
            for c, (g, xx) in {"own": (g_own, rx), "ctrate": (fx, rx), "ctrate_test": (fx, x[test])}.items():
                s = merge_scores(zero_shot_scores(normalize(g(xx) @ w), pr), cols)
                sc[(c, m)].append(s[:, ev])
        print(f"[radchest_eval] {a.encoder} seed={seed} own: " + " ".join(
            f"{m}={np.nanmean(auc_matrix(sc[('own', m)][-1], rlab)):.3f}" for m in METHODS), flush=True)

    out = {"encoder": a.encoder, "rad_encoder": rad_encoder(a.encoder), "model": a.model, "eval_labels": evals,
           "min_pos": MIN_POS, "n_rad": len(rc), "n_ctrate_test": len(test),
           "prevalence_rad": {n: float(rc[n].mean()) for n in evals}, "transfer": {}}
    for c in sets:
        lab = clab if c == "ctrate_test" else rlab
        boots = {}
        for m in METHODS:
            point, boots[m] = boot_mean_auc(sc[(c, m)], lab, np.random.default_rng(12345))   # same resamples for every method
            out["transfer"].setdefault(c, {})[m] = {
                "mean_auc": interval(point, boots[m]),
                "per_label_auc": dict(zip(evals, np.nanmean([auc_matrix(s, lab) for s in sc[(c, m)]], 0).tolist()))}
        diff = out["transfer"][c]["anchor_ridge"]["mean_auc"]["point"] - out["transfer"][c]["true100_ridge"]["mean_auc"]["point"]
        out["transfer"][c]["anchor_minus_true100"] = interval(diff, boots["anchor_ridge"] - boots["true100_ridge"])
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/{a.encoder}.json", "w"), indent=1)
    for c in sets:
        print(f"[radchest_eval] {a.encoder} {c}: " + " ".join(
            f"{m}={out['transfer'][c][m]['mean_auc']['point']:.3f}" for m in METHODS), flush=True)


def summary(a):
    """Main-table numbers (mean over seeds per n and centring) and the transfer table over align.external.ENCODERS."""
    n_pool = len(pd.read_json(f"{a.ctrate}/cases.jsonl", lines=True))
    files = [f"{a.out}/{a.encoder}_{a.model}_n{n}.csv" for n in NS]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"missing {missing}")
    df = pd.concat([pd.read_csv(f) for f in files]).sort_values(["centring", "n", "seed"])
    want = {n: len(SEEDS) if n < n_pool else 1 for n in NS}
    seeds = df.groupby(["centring", "n"]).seed.nunique()
    if any(seeds[c, n] != k for c in CENTRINGS for n, k in want.items()):
        raise ValueError(f"seed counts differ from {want}: {seeds.to_dict()}")
    os.makedirs(a.results, exist_ok=True)
    df.to_csv(f"{a.results}/{a.encoder}_{a.model}_rows.csv", index=False)
    labels = [c for c in df.columns if c.startswith("auc|")]
    agg = df.groupby(["centring", "n"])[["auc"] + labels].mean()
    agg["std"] = df.groupby(["centring", "n"]).auc.std(ddof=1)
    agg["seeds"] = seeds
    agg.reset_index().to_csv(f"{a.results}/{a.encoder}_{a.model}_summary.csv", index=False)
    print(agg[["auc", "std", "seeds"]].round(4).to_string(), flush=True)

    rs = [json.load(open(f"{a.out}/{e}.json")) for e in ENCODERS]
    if len({(r["model"], r["n_rad"], tuple(r["eval_labels"])) for r in rs}) != 1:
        raise ValueError("encoder results differ in model, case count or labels")
    rows = [{"encoder": r["encoder"], "centring": c, "method": m, **r["transfer"][c][m]["mean_auc"]}
            for r in rs for c in ("own", "ctrate", "ctrate_test") for m in METHODS]
    t = pd.DataFrame(rows)
    t.to_csv(f"{a.results}/transfer.csv", index=False)
    print(t.pivot_table(index="encoder", columns=["centring", "method"], values="point").round(3).to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["main", "transfer", "summary"])
    ap.add_argument("--ctrate", required=True)
    ap.add_argument("--rad")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--results", default="results/radchest")
    ap.add_argument("--save-scores", action="store_true",
                    help="main: also write {encoder}_{model}_n{n}_scores.npz (keys, TC_<centring>: seeds x volumes x labels)")
    a = ap.parse_args()
    if a.step != "summary" and a.rad is None:
        ap.error(f"{a.step} needs --rad")
    if a.save_scores and a.step != "main":
        ap.error("--save-scores needs the main step")
    {"main": main_run, "transfer": transfer_run, "summary": summary}[a.step](a)
    print(f"RADCHEST_EVAL_EXIT rc=0 step={a.step}", flush=True)


if __name__ == "__main__":
    main()
