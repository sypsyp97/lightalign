"""CXR threshold metrics as CheXzero and KAD report them: per finding, the threshold that maximises MCC on the official
CheXpert valid set (radiologist labels; no overlap with the test set), applied to the test set; MCC, F1 of the positive
class and accuracy with that threshold, mean over the five competition findings. Ours: per seed, then the seed mean,
with a study-level bootstrap 95% interval (1,000 resamples, as align.chexpert_payoff).

Input: the *_valid_scores.npz / *_test_scores.npz pairs of align.chexpert_payoff run --save-scores.

    python -m align.chexpert_thresh --scores DIR --out CSV
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from align.domains import COMPETITION

BOOT = 1000


def counts(pred, lab):
    """Confusion counts per column of 0/1 arrays (studies x findings): tp, fp, fn, tn."""
    tp = (pred & lab).sum(0)
    fp = (pred & (1 - lab)).sum(0)
    fn = ((1 - pred) & lab).sum(0)
    return tp, fp, fn, len(lab) - tp - fp - fn


def mcc_f1_acc(tp, fp, fn, tn):
    """MCC (0 when undefined, as sklearn), positive-class F1 (0 when undefined) and accuracy."""
    tp, fp, fn, tn = (np.asarray(v, float) for v in (tp, fp, fn, tn))
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros_like(den), where=den > 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    return mcc, f1, (tp + tn) / (tp + fp + fn + tn)


def best_threshold(s, l):
    """As CheXzero metrics.py: candidates are the valid scores themselves, prediction is score >= threshold; the
    candidate with the highest MCC wins, ties to the lowest threshold."""
    cand = np.unique(s)
    pred = (s[:, None] >= cand[None, :]).astype(int)                    # studies x candidates
    mcc = mcc_f1_acc(*counts(pred, np.repeat(l[:, None], len(cand), 1).astype(int)))[0]
    return cand[int(np.argmax(mcc))]


def tag_of(path):
    return os.path.basename(path).replace("_test_scores.npz", "")


def scores(pred, lab):
    return np.stack(mcc_f1_acc(*counts(pred, lab)), 1)                    # findings x (mcc, f1, acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for tp in sorted(glob.glob(f"{a.scores}/*_test_scores.npz")):
        vp = tp.replace("_test_scores.npz", "_valid_scores.npz")
        if not os.path.exists(vp):
            raise FileNotFoundError(f"{vp} missing for {tp}")
        zt, zv = np.load(tp), np.load(vp)
        if np.isnan(zt["labels"]).any() or np.isnan(zv["labels"]).any():
            raise ValueError(f"{tag_of(tp)}: unlabelled findings in the radiologist labels")
        lt, lv = zt["labels"].astype(int), zv["labels"].astype(int)
        if lt.shape[1] != len(COMPETITION) or lv.shape[1] != len(COMPETITION):
            raise ValueError(f"{tp}: expected {len(COMPETITION)} findings")
        rng = np.random.default_rng(0)
        res = [rng.integers(0, len(lt), len(lt)) for _ in range(BOOT)]
        tag = tag_of(tp)
        for r in ("P", "TC"):
            st, sv = zt[r], zv[r]
            if len(st) != len(sv):
                raise ValueError(f"{tag} {r}: {len(st)} test seeds vs {len(sv)} valid seeds")
            if not (np.isfinite(st).all() and np.isfinite(sv).all()):
                raise ValueError(f"{tag} {r}: non-finite scores")
            preds = []
            for s_t, s_v in zip(st, sv):
                thr = np.array([best_threshold(s_v[:, j], lv[:, j]) for j in range(lv.shape[1])])
                preds.append((s_t >= thr).astype(int))
            point = np.mean([scores(p, lt) for p in preds], 0)                     # findings x 3, seed mean
            draws = np.array([np.mean([scores(p[i], lt[i]).mean(0) for p in preds], 0) for i in res])   # BOOT x 3
            row = {"cell": tag, "reader": r, "seeds": len(preds), "test_studies": len(lt), "valid_studies": len(lv)}
            for k, name in enumerate(("mcc", "f1", "acc")):
                row[name] = float(point[:, k].mean())
                row[f"{name}_lo"], row[f"{name}_hi"] = (float(v) for v in np.percentile(draws[:, k], [2.5, 97.5]))
            row.update({f"mcc_{t}": float(v) for t, v in zip(COMPETITION, point[:, 0])})
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if "_" not in k or k == "test_studies"},
                  flush=True)
    if not rows:
        raise FileNotFoundError(f"no *_test_scores.npz in {a.scores}")
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
