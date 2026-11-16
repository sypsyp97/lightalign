"""Threshold metrics as CT-CLIP reports them (CT-CLIP repo scripts/bootstrap_values.py @ a2a155c): per finding, the
threshold among 100 points of np.linspace(0, 1, 100) closest to the top-left corner of the ROC (minimum of
sqrt((1 - TPR)^2 + FPR^2), ties to the larger threshold), chosen on the evaluated volumes themselves; then weighted F1,
accuracy, precision and recall of the positive class; means over the 18 findings, per evaluation unit.
Scores that are not probabilities (ours: decision functions) are min-max scaled to [0, 1] per finding first, which
keeps their order; CT-CLIP's rerun probabilities are used as they are. Ours: metrics per seed, then the seed mean.

    python -m align.thresh_metrics --valid DIR --ctclip CSV --scores NPZ [NPZ ...] --out CSV
"""
import argparse
import os

import numpy as np
import pandas as pd

from align.align import label_names
from align.official import units

GRID = np.linspace(0, 1, 100)


def threshold(prob, lab):
    best, best_t = np.inf, 0.0
    pos, neg = lab == 1, lab == 0
    for t in GRID:
        pred = prob > t
        tpr, fpr = (pred & pos).sum() / pos.sum(), (pred & neg).sum() / neg.sum()
        d = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
        if d <= best:
            best, best_t = d, t
    return best_t


def metrics(prob, lab):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    rows = []
    for j in range(lab.shape[1]):
        p, l = prob[:, j], lab[:, j]
        pred = (p > threshold(p, l)).astype(int)
        rows.append({"f1_weighted": f1_score(l, pred, average="weighted"), "accuracy": accuracy_score(l, pred),
                     "precision": precision_score(l, pred, zero_division=0), "recall": recall_score(l, pred)})
    return pd.DataFrame(rows).mean().to_dict()


def minmax(s):
    lo, hi = s.min(0), s.max(0)
    return (s - lo) / np.where(hi > lo, hi - lo, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid", required=True)
    ap.add_argument("--ctclip", required=True)
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    vc = pd.read_json(f"{a.valid}/cases.jsonl", lines=True)
    names = label_names(a.valid)
    lab = vc[names].to_numpy(int)
    ct = pd.read_csv(a.ctclip, index_col="VolumeName").loc[[f"{i}.nii.gz" for i in vc["id"]], names].to_numpy(float)
    rows = []
    for un, mk in units(vc).items():
        rows.append({"model": "CT-CLIP (prompt, released)", "unit": un, "volumes": int(mk.sum()),
                     **metrics(ct[mk], lab[mk])})
        for path in a.scores:
            z = np.load(path, allow_pickle=True)
            assert (z["ids"] == vc["id"].to_numpy()).all()
            tag = os.path.basename(path).replace("_scores.npz", "")
            for r in ("P", "TC"):
                per = [metrics(minmax(s[mk]), lab[mk]) for s in z[r]]
                rows.append({"model": f"{tag} {r}", "unit": un, "volumes": int(mk.sum()),
                             **pd.DataFrame(per).mean().to_dict()})
        print(pd.DataFrame([r for r in rows if r["unit"] == un]).round(3).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
