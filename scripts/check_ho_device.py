"""Equivalence checks for `align.rosetta ours --sign-ho K --ho-device {cpu,cuda}`.

    python scripts/check_ho_device.py csv CPU.csv CUDA.csv
        both files are $DATA/rosetta/<tag>__s8_signfix__ho10.csv of the same cell (one row per seed). Integer, flip-count
        and rank columns must be equal; *_foscttm, *_auc and ho_oracle_* columns agree to 1e-6 absolute (a tie in the
        test similarity can move one count); every other float column agrees to 1e-9 relative. Exit 1 on FAIL.
    python scripts/check_ho_device.py assign [--m 1500] [--n 8]
        needs CUDA and torch-linear-assignment: n random float64 m x m cost matrices, the GPU assignment must equal
        scipy.optimize.linear_sum_assignment index for index.
"""
import argparse
import sys

import numpy as np
import pandas as pd

ABS = ("_foscttm", "_auc", "ho_oracle_best_foscttm", "ho_oracle_worst_foscttm")


def check_csv(cpu_path, cuda_path):
    a, b = pd.read_csv(cpu_path), pd.read_csv(cuda_path)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        print(f"FAIL: columns or row count differ: {len(a)} x {list(a.columns)} vs {len(b)} x {list(b.columns)}")
        return False
    ok = True
    for col in a.columns:
        x, y = a[col], b[col]
        if not pd.api.types.is_float_dtype(x):
            diff, tol, kind = float((x != y).sum()), 0.0, "exact"
            good = diff == 0
        elif col.endswith(ABS[:2]) or col in ABS[2:]:
            d = np.abs(x - y)
            diff = float(np.nanmax(d)) if d.notna().any() else 0.0   # all-NaN columns (label-free domains): compare NaN masks only
            tol, kind = 1e-6, "abs"
            good = bool(np.array_equal(x.isna(), y.isna())) and diff <= tol
        else:
            d = np.abs(x - y) / np.maximum(np.abs(x), np.abs(y)).clip(lower=1e-300)
            diff = float(np.nanmax(d)) if d.notna().any() else 0.0
            tol, kind = 1e-9, "rel"
            good = bool(np.array_equal(x.isna(), y.isna())) and diff <= tol
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {col:32s} {kind:5s} max diff {diff:.3e}  tol {tol:.0e}")
    print("PASS" if ok else "FAIL")
    return ok


def check_assign(m, n):
    import torch
    from scipy.optimize import linear_sum_assignment
    from torch_linear_assignment import batch_linear_assignment

    if not torch.cuda.is_available():
        raise RuntimeError("assign check needs CUDA")
    rng = np.random.default_rng(0)
    cost = rng.standard_normal((n, m, m))
    got = batch_linear_assignment(torch.as_tensor(cost, device="cuda")).cpu().numpy()
    ok = True
    for i in range(n):
        want = linear_sum_assignment(cost[i])[1]
        same = np.array_equal(got[i], want)
        ok &= same
        print(f"{'PASS' if same else 'FAIL'}  matrix {i}: assignment equal={same}, "
              f"cost gpu {cost[i][np.arange(m), got[i]].sum():.12g} scipy {cost[i][np.arange(m), want].sum():.12g}")
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("csv")
    c.add_argument("cpu")
    c.add_argument("cuda")
    s = sub.add_parser("assign")
    s.add_argument("--m", type=int, default=1500)
    s.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    sys.exit(0 if (check_csv(a.cpu, a.cuda) if a.mode == "csv" else check_assign(a.m, a.n)) else 1)
