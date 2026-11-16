"""Shared-variance share f of every zero-pair cell, without running the aligner: the shared canonical directions and
f_image / f_report exactly as align.rosetta ours --scale 1 records them (rosetta._scale_shared, factor 1).

    python -m align.fshare --emb DIR --image E [E ...] --text T [T ...] --out CSV
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--image", nargs="+", required=True)
    ap.add_argument("--text", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from align.rosetta import _scale_shared

    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    train = np.flatnonzero(cases.split == "train")
    rows = []
    for img in a.image:
        xr = np.stack([np.load(f"{a.emb}/{img}/{os.path.basename(k)}.npy") for k in cases["key"]])
        for txt in a.text:
            yr = np.load(f"{a.emb}/{txt}.npy")
            pr = np.load(f"{a.emb}/{txt.replace('text', 'prompts', 1)}.npy")
            try:
                info = _scale_shared(cases, train, xr, yr, pr, 1.0)[3]
            except ValueError as e:
                if "no significant shared component" not in str(e):
                    raise
                info = {"k": 0, "f_image": 0.0, "f_report": 0.0}
            rows.append({"data": os.path.basename(a.emb.rstrip("/")), "image": img, "text": txt, **info})
            print(json.dumps(rows[-1]), flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
