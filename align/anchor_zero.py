"""Zero-pair gate: anatomy anchors read independently on each side, matched across unpaired sets, then an image -> report
map fitted on the matched (pseudo) pairs. No true image-report pair enters the fit.

Anchors (findings that both a segmentation and a report can state):
  pleural effusion      image: ml of ts_pleural_pericard_effusion label 2   report: zero-shot prompt score
  pericardial effusion  image: ml of ts_pleural_pericard_effusion label 3   report: zero-shot prompt score
  lung nodule           image: lung-nodule component count                report: zero-shot prompt score
  cardiomegaly          image: heart ml (ts_total label 51) / lung ml      report: zero-shot prompt score
The report score is cos(report, "X is present.") - cos(report, "No x.") in the frozen LLM space (no image involved).
Each side is rank-normalised within its own unpaired set; images (half A of "train") and reports (half B) are matched
one-to-one by the Hungarian algorithm on the squared distance of the anchor ranks. Maps: ridge and Procrustes on the
pseudo pairs. Controls: random pseudo pairs; 100 true pairs (reference). Evaluation on the "test" patients.

    python -m align.anchor_zero --emb DIR --out DIR [--encoder voxelfm] [--model 4B]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from sklearn.linear_model import RidgeCV

from align.align import SEEDS, foscttm, label_names, normalize, polar
from align.baselines import ALPHAS, auc_matrix, zero_shot_scores
from align.t2 import K

ANCHORS = ("Pleural effusion", "Pericardial effusion", "Lung nodule", "Cardiomegaly")
LUNG = [f"ts_total_{i}" for i in range(10, 15)]


def image_anchors(path, keys):
    m = pd.read_csv(path).set_index("key").loc[keys]
    lung = m[LUNG].sum(1)
    if (lung <= 0).any():
        raise ValueError(f"{int((lung <= 0).sum())} cases have no lung volume")
    return np.c_[m["ts_pleural_pericard_effusion_2"], m["ts_pleural_pericard_effusion_3"],
                 m["ts_lung_nodules_components"], m["ts_total_51"] / lung]


def ranks(a):
    return np.stack([rankdata(c) / len(c) for c in a.T], 1)


def evaluate(w, xt, yt, pr, lab):
    u = normalize(xt @ w)
    return auc_matrix(zero_shot_scores(u, pr), lab), foscttm(u @ yt.T)


def fit_seed(x, y, prompts, img_anchor, idx, train, seed, on_pairs=None):
    """The four maps of one seed: halves, anchor matching, fits. on_pairs(rng, half_a, half_b, txt_anchor, img_p, txt_p)
    runs right after the matching and may draw from rng. Returns maps and the centring of half A / half B."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train)
    half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]
    pairs = rng.choice(half_a, K, replace=False)
    mx, my = x[half_a].mean(0), y[half_b].mean(0)
    fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
    pr = fy(prompts)
    txt_anchor = zero_shot_scores(fy(y[half_b]), pr)[:, idx]
    ra, rb = ranks(img_anchor[half_a]), ranks(txt_anchor)
    cost = ((ra[:, None, :] - rb[None, :, :]) ** 2).sum(-1)
    i, j = linear_sum_assignment(cost)
    img_p, txt_p = half_a[i], half_b[j]
    if on_pairs is not None:
        on_pairs(rng, half_a, half_b, txt_anchor, img_p, txt_p)
    xs, ys = fx(x[img_p]), fy(y[txt_p])
    rand = rng.permutation(len(txt_p))
    maps = {"anchor_ridge": RidgeCV(alphas=ALPHAS, fit_intercept=False).fit(xs, ys).coef_.T,
            "anchor_procrustes": polar(xs.T @ ys),
            "random_ridge": RidgeCV(alphas=ALPHAS, fit_intercept=False).fit(xs, ys[rand]).coef_.T,
            "true100_ridge": RidgeCV(alphas=ALPHAS, fit_intercept=False).fit(fx(x[pairs]), fy(y[pairs])).coef_.T}
    return maps, fx, fy, pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="voxelfm")
    ap.add_argument("--model", default="4B")
    a = ap.parse_args()
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    names = label_names(a.emb)
    lab = cases[names].to_numpy(float)
    x = normalize(np.stack([np.load(f"{a.emb}/{a.encoder}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64))
    y = normalize(np.load(f"{a.emb}/textmid_{a.model}.npy").astype(np.float64))
    prompts = normalize(np.load(f"{a.emb}/promptsmid_{a.model}.npy").astype(np.float64))
    img_anchor = image_anchors(f"{a.emb}/measures.csv", cases["key"])
    idx = [names.index(n) for n in ANCHORS]
    train, test = (np.flatnonzero(cases.split == s) for s in ("train", "test"))
    rows, diag = [], {}
    def anchor_quality(rng, half_a, half_b, txt_anchor, img_p, txt_p):
        """Read on the halves' own labels (diagnostic only, never used in the fit)."""
        diag["image_anchor_auc_vs_label"] = auc_matrix(img_anchor[half_a], lab[half_a][:, idx]).tolist()
        diag["text_anchor_auc_vs_label"] = auc_matrix(txt_anchor, lab[half_b][:, idx]).tolist()
        diag["pseudo_pair_label_agreement"] = (lab[img_p][:, idx] == lab[txt_p][:, idx]).mean(0).tolist()
        diag["random_pair_label_agreement"] = (lab[img_p][:, idx] == lab[rng.permutation(txt_p)][:, idx]).mean(0).tolist()

    for seed in SEEDS:
        maps, fx, fy, pr = fit_seed(x, y, prompts, img_anchor, idx, train, seed,
                                    anchor_quality if seed == 0 else None)
        other = [k for k in range(len(names)) if k not in idx]
        for name, w in maps.items():
            auc, fos = evaluate(w, fx(x[test]), fy(y[test]), pr, lab[test])
            rows.append({"seed": seed, "method": name, "mean_auc": float(np.nanmean(auc)), "foscttm": fos,
                         "auc_anchor_labels": float(np.nanmean(auc[idx])), "auc_other_labels": float(np.nanmean(auc[other])),
                         **{f"auc_{n}": float(v) for n, v in zip(names, auc)}})
        print(f"[anchor-zero] seed={seed} " + " ".join(f"{r['method']}={r['mean_auc']:.3f}/{r['foscttm']:.3f}" for r in rows[-4:]), flush=True)
    df = pd.DataFrame(rows)
    summ = df.groupby("method")[["mean_auc", "auc_anchor_labels", "auc_other_labels", "foscttm"]].agg(["mean", "std"])
    best = df[df.method.str.startswith("anchor")].groupby("method")[["mean_auc", "foscttm"]].mean()
    out = {"encoder": a.encoder, "model": a.model, "anchors": ANCHORS, "diagnostics": diag,
           "summary": {m: {"mean_auc": float(g.mean_auc.mean()), "auc_anchor_labels": float(g.auc_anchor_labels.mean()),
                           "auc_other_labels": float(g.auc_other_labels.mean()), "foscttm": float(g.foscttm.mean())}
                       for m, g in df.groupby("method")},
           "gate_pass": bool(((best.foscttm <= 0.40) & (best.mean_auc >= 0.60)).any())}
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/{a.encoder}_cells.csv", index=False)
    json.dump(out, open(f"{a.out}/{a.encoder}.json", "w"), indent=1)
    print(summ.to_string())
    print(json.dumps(out, indent=1), flush=True)
    print("ANCHOR_ZERO_EXIT rc=0", flush=True)


if __name__ == "__main__":
    main()
