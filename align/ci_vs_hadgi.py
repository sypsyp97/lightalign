"""Paired patient-bootstrap CI of mean AUROC, ridge minus Hadgi, on one unit of the CT-RATE validation set."""
import argparse

import numpy as np
import pandas as pd

from align.align import label_names
from align.baselines import auc_matrix
from align.official import units
from align.official_matrix import BOOT, patient_resamples


def scores(path, ids, key="TC", id_key="ids"):
    """Per-seed scores (seeds x cases x findings) of reader `key`, rows reordered to `ids`."""
    z = np.load(path, allow_pickle=True)
    for k in (id_key, key):
        if k not in z.files:
            raise KeyError(f"{path}: no {k}, keys {z.files}")
    zi = z[id_key].astype(str)
    if len(set(zi)) != len(zi) or set(zi) != set(ids):
        raise ValueError(f"{path}: ids differ from the evaluated cases")
    tc = z[key]
    if tc.ndim != 3 or tc.shape[1] != len(ids):
        raise ValueError(f"{path}: {key} shape {tc.shape}")
    pos = pd.Series(np.arange(len(zi)), index=zi)[ids].to_numpy()
    return tc[:, pos, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", required=True)
    ap.add_argument("--ridge", required=True, help="format string with {n}")
    ap.add_argument("--hadgi", required=True, help="format string with {n}")
    ap.add_argument("--ns", type=int, nargs="+", default=[100, 400, 1600])
    ap.add_argument("--unit", default="chest")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
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
        r, h = scores(a.ridge.format(n=n), ids), scores(a.hadgi.format(n=n), ids)
        pr, ph = point(r), point(h)
        print(f"n={n} seeds ridge={len(r)} hadgi={len(h)} ridge={pr:.4f} hadgi={ph:.4f}", flush=True)
        if n == 1600 and (round(pr, 3), round(ph, 3)) != (0.809, 0.790):
            raise AssertionError(f"n=1600 does not reproduce published 0.809 / 0.790: {pr:.4f} / {ph:.4f}")
        d = draws(r) - draws(h)
        rows.append({"n": n, "unit": a.unit, "reader": "TC", "findings": len(names), "boot": BOOT, "ridge": pr,
                     "hadgi": ph, "diff": pr - ph, "lo": float(np.percentile(d, 2.5)),
                     "hi": float(np.percentile(d, 97.5))})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
