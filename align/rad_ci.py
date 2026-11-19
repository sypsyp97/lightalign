"""RAD-ChestCT intervals and label subsets for the main-table rows of align.radchest_eval (reader TC, 16 RAD labels).

boot: bootstrap interval (align.official_matrix.BOOT resamples, rng seed 12345) of the seed-mean mean AUROC over the
16 RAD labels, from the per-volume scores that `align.radchest_eval main --save-scores` writes, and the fraction of
resamples above --ref. RAD-ChestCT has one volume per case and its cases.jsonl has no patient field, so the resample
unit is the volume.
subset: seed-mean and seed std of the mean AUROC over the labels in --labels, from the per-label csv of
align.radchest_eval main.

    python -m align.rad_ci boot --ctrate DIR --rad DIR --dir DIR --encoder voxelfm_b11 --model 9B --csv CSV
        [--ns 100] [--centring ctrate] [--ref 0.629] [--expect N:AUC ...]
    python -m align.rad_ci subset --dir DIR --encoder voxelfm_b11 --model 9B --labels L1 L2 ... --csv CSV
        [--ns 100 ...] [--centring ctrate]

--dir is the --out directory of align.radchest_eval main. --expect checks the seed-mean point estimate of the listed n
at 3 decimals before the bootstrap and raises on a mismatch.
"""
import argparse

import numpy as np
import pandas as pd

from align.baselines import auc_matrix
from align.ci_vs_hadgi import scores
from align.official_matrix import BOOT, patient_resamples
from align.radchest_eval import CENTRINGS, rad_cases

RNG_SEED = 12345


def boot(a):
    expect = {}
    for e in a.expect:
        n, auc = e.split(":")
        expect[int(n)] = float(auc)
    if set(expect) - set(a.ns):
        raise ValueError(f"--expect n {sorted(set(expect) - set(a.ns))} not in --ns {a.ns}")
    rc, rnames = rad_cases(a.rad, a.ctrate)
    keys = rc["key"].astype(str).to_numpy()
    lab = rc[rnames].to_numpy(float)
    res = patient_resamples(keys, seed=RNG_SEED)
    rows = []
    for n in a.ns:
        sc = scores(f"{a.dir}/{a.encoder}_{a.model}_n{n}_scores.npz", keys, f"TC_{a.centring}", "keys")
        if sc.shape[2] != len(rnames):
            raise ValueError(f"n={n}: {sc.shape[2]} score columns, {len(rnames)} RAD labels")
        point = float(np.mean([np.nanmean(auc_matrix(s, lab)) for s in sc]))
        print(f"n={n} centring={a.centring} seeds={len(sc)} auc={point:.4f}", flush=True)
        if n in expect and round(point, 3) != round(expect[n], 3):
            raise AssertionError(f"n={n} does not reproduce {expect[n]}: {point:.4f}")
        d = np.array([np.mean([np.nanmean(auc_matrix(s[j], lab[j])) for s in sc]) for j in res])
        rows.append({"n": n, "centring": a.centring, "unit": "volume", "volumes": len(keys), "labels": len(rnames),
                     "seeds": len(sc), "boot": BOOT, "rng_seed": RNG_SEED, "auc": point,
                     "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)), "ref": a.ref,
                     "frac_above_ref": float(np.mean(d > a.ref))})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(a.csv, index=False)


def subset(a):
    rows = []
    for n in a.ns:
        path = f"{a.dir}/{a.encoder}_{a.model}_n{n}.csv"
        df = pd.read_csv(path)
        df = df[df.centring == a.centring]
        if df.empty or df.seed.duplicated().any():
            raise ValueError(f"{path}: centring {a.centring} has seeds {df.seed.tolist()}")
        missing = [l for l in a.labels if f"auc|{l}" not in df.columns]
        if missing:
            raise KeyError(f"{path}: no labels {missing}, has {[c[4:] for c in df.columns if c.startswith('auc|')]}")
        per_seed = np.nanmean(df[[f"auc|{l}" for l in a.labels]].to_numpy(float), 1)
        rows.append({"n": n, "centring": a.centring, "labels": len(a.labels), "seeds": len(df),
                     "auc": float(per_seed.mean()), "std": float(pd.Series(per_seed).std(ddof=1))})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(a.csv, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["boot", "subset"])
    ap.add_argument("--dir", required=True, help="--out directory of align.radchest_eval main")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ns", type=int, nargs="+", default=[100])
    ap.add_argument("--centring", choices=CENTRINGS, default="ctrate")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ctrate", help="boot")
    ap.add_argument("--rad", help="boot")
    ap.add_argument("--ref", type=float, help="boot")
    ap.add_argument("--expect", nargs="+", default=[], help="boot: n:auc")
    ap.add_argument("--labels", nargs="+", help="subset: RAD label names")
    a = ap.parse_args()
    if a.step == "boot" and None in (a.ctrate, a.rad, a.ref):
        ap.error("boot needs --ctrate, --rad and --ref")
    if a.step == "subset" and not a.labels:
        ap.error("subset needs --labels")
    {"boot": boot, "subset": subset}[a.step](a)


if __name__ == "__main__":
    main()
