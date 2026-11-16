"""Budget law -> alignment outcomes.

curve:   B1 / B2. Predictor S(m) from the sel-patient CCA of align.spiked; targets from align.anchor_pairsel
         (random_alone learning curve) and align.anchor_zero (anchor_ridge, true100_ridge).
amplify: B3. Zero-pair Gromov-Wasserstein after amplifying each side along its shared canonical directions (oracle
         directions from the sel patients), as a function of the shared-variance fraction.

    python -m align.budget curve --spiked DIR --pairsel DIR --anchor DIR --out DIR
    python -m align.budget amplify --emb DIR --out DIR --encoder E [--factors 1,0.5,0.25]
    python -m align.budget fraction --emb DIR --out DIR        (shared-variance fraction of all encoders, factor 1)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from align.align import SEEDS, foscttm, gromov_wasserstein, label_names, normalize, svd
from align.baselines import auc_matrix, zero_shot_scores
from align.spiked import ENCODERS, PERMS, edge, n_significant, qr_centred, rc, top_null

CURVE_D, CURVE_MODEL = 16, "4B"
MS = (50, 100, 200, 400)
AMP_D = 64
FACTORS = (1, 2, 4, 8, 16, 32)


def budget_signal(r_hat, m, d=CURVE_D):
    t = rc(d / m, d / m)
    return float(sum(r for r in r_hat if r > t))


def curve(a):
    rows = []
    for e in ENCODERS:
        r_hat = json.load(open(f"{a.spiked}/{e}_estimates.json"))[f"{CURVE_MODEL}/mid/{CURVE_D}"]["r_hat"]
        cells = pd.read_csv(f"{a.pairsel}/{e}_cells.csv")
        alone = cells[cells.method == "random_alone"].groupby("m").mean_auc.mean()
        az = json.load(open(f"{a.anchor}/{e}.json"))["summary"]
        rows.append({"encoder": e, "r1": r_hat[0] if r_hat else 0.0, "k": len(r_hat),
                     **{f"S{m}": budget_signal(r_hat, m) for m in MS}, **{f"auc{m}": float(alone.loc[m]) for m in MS},
                     "anchor_minus_true100": az["anchor_ridge"]["mean_auc"] - az["true100_ridge"]["mean_auc"]})
    df = pd.DataFrame(rows)
    rho = lambda u, v: float(spearmanr(df[u], df[v]).statistic)
    res = {"B1_spearman": {m: rho(f"S{m}", f"auc{m}") for m in MS}, "B1_r1_spearman": {m: rho("r1", f"auc{m}") for m in MS},
           "B2_spearman": rho("S100", "anchor_minus_true100")}
    res["B1_pass"] = bool(res["B1_spearman"][100] >= 0.7)
    res["B2_pass"] = bool(res["B2_spearman"] <= -0.6)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/curve.csv", index=False)
    json.dump(res, open(f"{a.out}/curve.json", "w"), indent=1)
    print(df.round(3).to_string())
    print(json.dumps(res, indent=1), flush=True)


def shared_directions(x, y, rng):
    """Orthonormal bases (in the original feature spaces) of the canonical variates of the top-AMP_D principal
    coordinates that are significant against the permutation null and above the null edge."""
    xc, yc = x - x.mean(0), y - y.mean(0)
    vx = svd(xc, full_matrices=False)[2][:AMP_D].T
    vy = svd(yc, full_matrices=False)[2][:AMP_D].T
    (qx, rx), (qy, ry) = qr_centred(xc @ vx), qr_centred(yc @ vy)
    u, sv, vt = svd(qx.T @ qy)
    n = len(x)
    k = n_significant(sv ** 2, top_null(qx, qy, rng, PERMS))
    k = int(sum(sv[:k] ** 2 > edge(AMP_D / n, AMP_D / n)))
    if k == 0:
        raise ValueError("no significant shared component")
    ux = np.linalg.qr(vx @ np.linalg.solve(rx, u[:, :k]))[0]
    uy = np.linalg.qr(vy @ np.linalg.solve(ry, vt[:k].T))[0]
    return ux, uy, k


def amplify_map(u, mean, factor):
    return lambda z: z + (factor - 1) * ((z - mean) @ u) @ u.T


def shared_fraction(z, u):
    zc = z - z.mean(0)
    return float(((zc @ u) ** 2).sum() / (zc ** 2).sum())


def amplify(a):
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    names = label_names(a.emb)
    sel, train, test = (np.flatnonzero(cases.split == s) for s in ("sel", "train", "test"))
    x = np.stack([np.load(f"{a.emb}/{a.encoder}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64)
    y = np.load(f"{a.emb}/textmid_{CURVE_MODEL}.npy").astype(np.float64)
    prompts = np.load(f"{a.emb}/promptsmid_{CURVE_MODEL}.npy").astype(np.float64)
    lab = cases[names].to_numpy(float)[test]
    ux, uy, k = shared_directions(x[sel], y[sel], np.random.default_rng(0))
    amp_x, amp_y = (ux[:, :a.top], uy[:, :a.top]) if a.top else (ux, uy)   # --top K: amplify the K strongest only
    rows = []
    factors = [float(v) for v in a.factors.split(",")] if a.factors else FACTORS
    for factor in factors:
        gx, gy = amplify_map(amp_x, x[sel].mean(0), factor), amplify_map(amp_y, y[sel].mean(0), factor)
        xa, ya, pa = normalize(gx(x)), normalize(gy(y)), normalize(gy(prompts))
        f = {"f_image": shared_fraction(xa[train], ux), "f_report": shared_fraction(ya[train], uy)}
        if a.axis:
            f["f_image_top1"] = shared_fraction(xa[train], ux[:, :1])
        for seed in SEEDS:
            perm = np.random.default_rng(seed).permutation(train)
            half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]
            mx, my = xa[half_a].mean(0), ya[half_b].mean(0)
            fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
            w, gx, gy = gromov_wasserstein(fx(xa[half_a]), fy(ya[half_b]), seed)
            fx, fy = (lambda v, f=fx, g=gx: g(f(v))), (lambda v, f=fy, g=gy: g(f(v)))   # the aligner's own centering on top
            u = normalize(fx(xa[test]) @ w)
            rows.append({"encoder": a.encoder, "factor": factor, "seed": seed, "k": k, **f,
                         "mean_auc": float(np.nanmean(auc_matrix(zero_shot_scores(u, fy(pa)), lab))),
                         "foscttm": foscttm(u @ fy(ya[test]).T)})
            if a.axis:      # does the zero-pair map recover the strongest shared axis? |corr| of test coordinates
                rows[-1]["axis_corr"] = float(abs(np.corrcoef(u @ uy[:, 0], fy(ya[test]) @ uy[:, 0])[0, 1]))
            print(json.dumps(rows[-1]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    df = pd.DataFrame(rows)
    tag = f"{a.encoder}_top{a.top}" if a.top else a.encoder
    df.to_csv(f"{a.out}/{tag}_amplify.csv", index=False)
    s = df.groupby("factor")[["f_image", "f_report", "mean_auc", "foscttm"] + (["f_image_top1", "axis_corr"] if a.axis else [])].mean()
    print(s.round(3).to_string())
    json.dump({"encoder": a.encoder, "k": k, "by_factor": s.reset_index().to_dict("records"),
               "leaves_chance": {f"{fa:g}": bool(r.mean_auc >= 0.60 and r.foscttm <= 0.40) for fa, r in s.iterrows()}},
              open(f"{a.out}/{tag}_amplify.json", "w"), indent=1)


def fraction(a):
    """amplify at factor 1 for every encoder: share of each side's train variance in its shared directions."""
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    sel, train = (np.flatnonzero(cases.split == s) for s in ("sel", "train"))
    y = np.load(f"{a.emb}/textmid_{CURVE_MODEL}.npy").astype(np.float64)
    out = {}
    for e in ENCODERS:
        x = np.stack([np.load(f"{a.emb}/{e}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64)
        ux, uy, k = shared_directions(x[sel], y[sel], np.random.default_rng(0))
        out[e] = {"k": k, "f_image": shared_fraction(normalize(x)[train], ux),
                  "f_report": shared_fraction(normalize(y)[train], uy)}
        print(e, json.dumps(out[e]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/fractions.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["curve", "amplify", "fraction"])
    ap.add_argument("--spiked")
    ap.add_argument("--pairsel")
    ap.add_argument("--anchor")
    ap.add_argument("--emb")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder")
    ap.add_argument("--top", type=int, help="amplify: amplify only the K strongest shared directions")
    ap.add_argument("--axis", action="store_true", help="amplify: also report recovery of the strongest shared axis")
    ap.add_argument("--factors", help="amplify: comma-separated factors (default 1,2,4,8,16,32; < 1 attenuates)")
    a = ap.parse_args()
    {"curve": curve, "amplify": amplify, "fraction": fraction}[a.step](a)
    print("BUDGET_EXIT rc=0", flush=True)


if __name__ == "__main__":
    main()
