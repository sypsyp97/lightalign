"""Image-side measurements of CT-RATE volumes from the TotalSegmentator masks released with the dataset (no report).

Per case: volume in ml of every label of ts_total (anatomy), ts_lung_nodules and ts_pleural_pericard_effusion, and
the number of connected lung-nodule components. Written as one CSV row per case; the alignability predictor in
align.t2 reads it.

    python -m align.measures --emb DIR --rank R --world N --workers 32
    python -m align.measures --emb DIR --merge
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from align.ctrate import CTRATE

TASKS = ("ts_total", "ts_lung_nodules", "ts_pleural_pericard_effusion")


def measure(key):
    import nibabel as nib

    return measure_masks(key, {task: nib.load(f"{CTRATE}/ts_seg/{task}/{key}.nii.gz") for task in TASKS})


def measure_masks(key, masks):
    """masks: TASKS -> TotalSegmentator label image of one case."""
    from scipy import ndimage

    row = {"key": key}
    for task in TASKS:
        img = masks[task]
        a = np.asanyarray(img.dataobj)
        if not np.issubdtype(a.dtype, np.integer):
            raise TypeError(f"{task} mask of {key} is {a.dtype}, expected integer labels")
        ml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
        counts = np.bincount(a.ravel())
        for label in np.flatnonzero(counts[1:]) + 1:
            row[f"{task}_{label}"] = counts[label] * ml
        if task == "ts_lung_nodules":       # labels of the TotalSegmentator lung_nodules task: 1 lung, 2 nodule
            if counts.size > 3:
                raise ValueError(f"unexpected label {counts.size - 1} in the lung-nodule mask of {key}")
            row["ts_lung_nodules_components"] = ndimage.label(a == 2)[1]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        parts = sorted(glob.glob(f"{a.emb}/measures/part_*.csv"))
        df = pd.concat([pd.read_csv(p) for p in parts]).fillna(0.0)
        cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
        missing = set(cases["key"]) - set(df["key"])
        if missing:
            raise RuntimeError(f"{len(missing)} cases have no measurements, e.g. {sorted(missing)[:3]}")
        df.set_index("key").loc[cases["key"]].to_csv(f"{a.emb}/measures.csv")
        print(f"[measures] merged {len(parts)} parts: {len(df)} cases x {df.shape[1] - 1} measurements", flush=True)
        return
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    keys = cases["key"].tolist()[a.rank::a.world]
    os.makedirs(f"{a.emb}/measures", exist_ok=True)
    out = f"{a.emb}/measures/part_{a.rank:03d}.csv"
    if os.path.exists(out):
        print(f"[measures] rank={a.rank} done before", flush=True)
        return
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(measure, keys, chunksize=4))
    tmp = f"{out}.{os.uname().nodename}.{os.getpid()}.tmp"   # two jobs may finish the same part
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, out)
    print(f"[measures] rank={a.rank} cases={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
