"""Paired patient-bootstrap CI of mean AUROC, LightAlign (reader TC of the ridge map) minus the tuned image classifier
(reader IL of align.fewpair image_probe_cv), on one unit of the CT-RATE validation set. Both sides are averaged over
their seeds before the difference, on the patient resamples of align.official_matrix.patient_resamples.

    python -m align.ci_vs_probe --valid DIR --ridge PATH_WITH_{n} --probe PATH_WITH_{n} --out CSV
        [--ns 100 200 400 1600 5994] [--unit chest] [--expect N:OURS:PROBE ...]

--expect checks the point estimates of the listed n at 3 decimals before the bootstrap and raises on a mismatch.
"""
import argparse

import numpy as np
import pandas as pd

from align.align import label_names
from align.baselines import auc_matrix
from align.ci_vs_hadgi import scores
from align.official import units
from align.official_matrix import BOOT, patient_resamples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", required=True)
    ap.add_argument("--ridge", required=True, help="format string with {n}; npz with ids and TC")
    ap.add_argument("--probe", required=True, help="format string with {n}; npz with ids and IL")
    ap.add_argument("--ns", type=int, nargs="+", default=[100, 200, 400, 1600, 5994])
    ap.add_argument("--unit", default="chest")
    ap.add_argument("--expect", nargs="+", default=[], help="n:ours:probe")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    expect = {}
    for e in a.expect:
        n, ours, probe = e.split(":")
        expect[int(n)] = (float(ours), float(probe))
    if set(expect) - set(a.ns):
        raise ValueError(f"--expect n {sorted(set(expect) - set(a.ns))} not in --ns {a.ns}")
    names = label_names(a.valid)
    vc = pd.read_json(f"{a.valid}/cases.jsonl", lines=True)
    ids = vc["id"].astype(str).to_numpy()
    lv = vc[names].to_numpy(float)
    idx = np.flatnonzero(units(vc)[a.unit])
    res = patient_resamples(vc["id"].str.split("_").str[1].to_numpy()[idx])
    lab = lv[idx]

    def draws(sc):
        sc = sc[:, idx]
        return np.array([np.mean([np.nanmean(auc_matrix(s[j], lab[j])) for s in sc]) for j in res])

    def point(sc):
        return float(np.mean([np.nanmean(auc_matrix(s[idx], lab)) for s in sc]))

    rows = []
    for n in a.ns:
        r, p = scores(a.ridge.format(n=n), ids, "TC"), scores(a.probe.format(n=n), ids, "IL")
        pr, pp = point(r), point(p)
        print(f"n={n} seeds ours={len(r)} probe={len(p)} ours={pr:.4f} probe={pp:.4f}", flush=True)
        if n in expect and (round(pr, 3), round(pp, 3)) != tuple(round(v, 3) for v in expect[n]):
            raise AssertionError(f"n={n} does not reproduce {expect[n][0]} / {expect[n][1]}: {pr:.4f} / {pp:.4f}")
        d = draws(r) - draws(p)
        rows.append({"n": n, "unit": a.unit, "findings": len(names), "boot": BOOT, "ours": pr, "probe": pp,
                     "diff": pr - pp, "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
