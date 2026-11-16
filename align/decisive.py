"""Decisive checks of the two-condition account.

spectrum:    S. Canonical spectrum (align.spiked.estimate) of the B3-amplified features, per factor.
prospective: P. A 200-pair BIMCV pilot predicts the number of significant shared components at larger n.
strata:      M. Burden- and length-matched finding AUC of full vs rank-one ridge maps.
probes:      Q. Finding probes on canonical variate 1 vs 1..k, same strata.
probes-control: R. Same probes on k canonical variates vs top-k principal coordinates vs k random directions.
readout:     RO. Report-based readers (report prototype, text classifier) vs the prompt readout, full vs rank-one map.
reader:      RD. Label-free canonical reader (k from the budget law) vs the ridge map with prompt dot product.

    python -m align.decisive spectrum --emb DIR --out DIR --encoder E
    python -m align.decisive prospective --emb DIR --spiked DIR --out DIR --encoder E
    python -m align.decisive prospective-summary --out DIR
    python -m align.decisive strata --emb DIR --out DIR --encoder E
    python -m align.decisive strata-summary --out DIR
    python -m align.decisive probes --emb DIR --spiked DIR --out DIR --encoder E
    python -m align.decisive probes-summary --out DIR
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from align.align import SEEDS, label_names, normalize, svd
from align.anchor_followup import ridge
from align.baselines import auc_matrix
from align.budget import AMP_D, CURVE_MODEL, FACTORS, amplify_map, shared_directions, shared_fraction
from align.spiked import ENCODERS, draw_stats, estimate, pcs, rc

TOP = 10
PILOT, TARGETS, DRAWS, PB_D = 200, (400, 800), 10, 16
STRATA_NS = (100, 1600)
MIN_CLASS = 5
PROBE_PAIRS = 1600
RANDOM_DRAWS = 5
CKA_PAIRS, CKA_ENCODERS, SCORE_N = 2000, ("voxelfm_patch", "dinov2_2d", "ltx_vae"), 1600
READER_NS = (400, 1600)


def spectrum(a):
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    sel, train = (np.flatnonzero(cases.split == s) for s in ("sel", "train"))
    x = np.stack([np.load(f"{a.emb}/{a.encoder}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64)
    y = np.load(f"{a.emb}/textmid_{CURVE_MODEL}.npy").astype(np.float64)
    ux, uy, k = shared_directions(x[sel], y[sel], np.random.default_rng(0))
    rows, base = [], {}
    factors = [float(v) for v in a.factors.split(",")] if a.factors else FACTORS
    for factor in factors:
        gx, gy = amplify_map(ux, x[sel].mean(0), factor), amplify_map(uy, y[sel].mean(0), factor)
        xa, ya = normalize(gx(x)), normalize(gy(y))
        r_hat, lam, _ = estimate(pcs(xa, AMP_D)[train], pcs(ya, AMP_D)[train], np.random.default_rng(0))
        base[factor] = r_hat[:k]
        rows.append({"encoder": a.encoder, "factor": factor, "k_oracle": k,
                     "f_image": shared_fraction(xa[train], ux), "f_report": shared_fraction(ya[train], uy),
                     "r_hat": r_hat[:TOP].tolist(), "lam": lam[:TOP].tolist()})
        print(json.dumps(rows[-1]), flush=True)
    if a.factors:                                    # NAT: largest change of r_hat_j (j <= k) against factor 1
        res = {"encoder": a.encoder, "k_oracle": k, "by_factor": rows,
               "max_abs_dr_vs_1": {f"{fa:g}": float(np.abs(base[fa] - base[1.0]).max()) for fa in factors}}
    else:
        res = {"encoder": a.encoder, "k_oracle": k, "by_factor": rows,
               "max_abs_dr_8_vs_1": float(np.abs(base[8] - base[1]).max()),
               "S1_encoder": bool(np.abs(base[8] - base[1]).max() <= 0.05)}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/{a.encoder}_spectrum.json", "w"), indent=1)
    print(json.dumps({k_: v for k_, v in res.items() if k_ != "by_factor"}), flush=True)


def bimcv_kept(emb):
    """Cases minus the volumes listed in <emb>/rejected, as align.spiked.run."""
    cases = pd.read_json(f"{emb}/cases.jsonl", lines=True)
    rejected = {os.path.basename(p)[:-5] for p in glob.glob(f"{emb}/rejected/*.json")}
    keep = ~cases.key.isin(rejected).to_numpy()
    return cases[keep].reset_index(drop=True), keep


def r1_calibration(spiked, d, ns):
    """Per n: least-squares line of CT-RATE k_obs(n) on CT-RATE r1 across the encoders (4B, mid, d); an n off the
    CT-RATE grid takes each encoder's k_obs linearly interpolated in n."""
    r1, kobs = [], {n: [] for n in ns}
    for e in ENCODERS:
        r = json.load(open(f"{spiked}/{e}_estimates.json"))[f"{CURVE_MODEL}/mid/{d}"]["r_hat"]
        r1.append(r[0] if r else 0.0)
        c = pd.read_csv(f"{spiked}/{e}_cells.csv")
        c = c[(c.model == CURVE_MODEL) & (c.layer == "mid") & (c.d == d)].sort_values("n")
        for n in ns:
            if not c.n.min() <= n <= c.n.max():
                raise ValueError(f"n = {n} outside the CT-RATE grid {c.n.tolist()}")
            kobs[n].append(float(np.interp(n, c.n, c.k_obs)))
    return {n: np.polyfit(r1, kobs[n], 1) for n in ns}


def prospective(a):
    cases, keep = bimcv_kept(a.emb)
    x = pcs(np.stack([np.load(f"{a.emb}/{a.encoder}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64), PB_D)
    txt = np.load(f"{a.emb}/textmid_{CURVE_MODEL}.npy").astype(np.float64)
    if len(txt) != len(keep):
        raise ValueError(f"textmid_{CURVE_MODEL}: {len(txt)} rows for {len(keep)} cases")
    y = pcs(txt[keep], PB_D)
    n_all = len(cases)
    targets = (TARGETS[0], min(TARGETS[1], (n_all - PILOT) // 100 * 100))
    if targets[1] <= targets[0] or targets[0] > n_all - PILOT:
        raise ValueError(f"{n_all} kept patients: too few for a {PILOT}-pair pilot and targets {TARGETS}")
    calib = r1_calibration(a.spiked, PB_D, targets)
    ref = np.array(json.load(open(f"{a.spiked}/{a.encoder}_estimates.json"))[f"{CURVE_MODEL}/mid/{PB_D}"]["r_hat"])
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_all)
        pilot, rest = perm[:PILOT], perm[PILOT:]
        r_hat = estimate(x[pilot], y[pilot], rng)[0]
        r_hat = r_hat[r_hat > 0]
        k_pilot = draw_stats(x[pilot], y[pilot], rng)[1]
        r1 = float(r_hat[0]) if len(r_hat) else 0.0
        for n in targets:
            t = rc(PB_D / n, PB_D / n)
            k_obs = np.mean([draw_stats(x[idx], y[idx], rng)[1]
                             for idx in (rng.choice(rest, n, replace=False) for _ in range(DRAWS))])
            rows.append({"encoder": a.encoder, "seed": seed, "n": n, "n_patients": n_all, "k_obs": float(k_obs),
                         "L": int((r_hat > t).sum()), "C": max(0.0, float(np.polyval(calib[n], r1))),
                         "P": int(k_pilot), "X": int((ref > t).sum()), "r1_pilot": r1})
            print(json.dumps(rows[-1]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/{a.encoder}_prospective.csv", index=False)


def prospective_summary(a):
    df = pd.concat([pd.read_csv(f"{a.out}/{e}_prospective.csv") for e in ENCODERS])
    res = {}
    for n, g in df.groupby("n"):
        res[int(n)] = {p: float((g[p] - g.k_obs).abs().median()) for p in ("L", "C", "P", "X")}
    per_n = [res[int(n)] for n in df.n.unique()]
    res["PB_pass"] = bool(all(v["L"] < v["C"] and v["L"] < v["P"] and v["L"] <= 1.5 for v in per_n))   # no claim
    res["PX_pass"] = bool(all(v["X"] < v["C"] and v["X"] < v["L"] and v["X"] <= 1.5 for v in per_n))
    json.dump(res, open(f"{a.out}/prospective_summary.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


def stratified_auc(scores, lab, strata):
    """Per finding: within-stratum AUCs (strata with >= MIN_CLASS of each class) pooled with weights pos x neg;
    nan if no stratum qualifies."""
    num, den = np.zeros(lab.shape[1]), np.zeros(lab.shape[1])
    for s in np.unique(strata):
        m = strata == s
        pos = lab[m].sum(0)
        neg = m.sum() - pos
        ok = (pos >= MIN_CLASS) & (neg >= MIN_CLASS)
        if not ok.any():
            continue
        auc = auc_matrix(scores[m], lab[m])
        w = np.where(ok, pos * neg, 0.0)
        num += np.where(ok, auc, 0.0) * w
        den += w
    with np.errstate(invalid="ignore"):
        return num / den


def test_strata(cases, lab, test):
    """Stratum of each test patient: finding count (0..5, >= 6) x report-length tercile of the test patients."""
    burden = np.minimum(lab.sum(1), 6).astype(int)
    length = cases["report"].str.split().str.len().to_numpy(float)[test]
    tercile = np.searchsorted(np.quantile(length, [1 / 3, 2 / 3]), length, side="right")
    return burden * 3 + tercile


def rank_one(w):
    u, s, vt = svd(w, full_matrices=False)
    return s[0] * np.outer(u[:, 0], vt[0])


def strata(a):
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    names = label_names(a.emb)
    train, test = (np.flatnonzero(cases.split == s) for s in ("train", "test"))
    x = normalize(np.stack([np.load(f"{a.emb}/{a.encoder}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64))
    y = normalize(np.load(f"{a.emb}/textmid_{CURVE_MODEL}.npy").astype(np.float64))
    prompts = normalize(np.load(f"{a.emb}/promptsmid_{CURVE_MODEL}.npy").astype(np.float64))
    lab = cases[names].to_numpy(float)[test]
    group = test_strata(cases, lab, test)
    mx, my = x[train].mean(0), y[train].mean(0)
    fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
    pr = fy(prompts)
    diff = (pr[0::2] - pr[1::2]).T
    rows = []
    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(train)
        for n in STRATA_NS:
            w = ridge(fx(x[perm[:n]]), fy(y[perm[:n]])).coef_.T
            for name, m in (("full", w), ("rank_one", rank_one(w))):
                auc = stratified_auc(fx(x[test]) @ m @ diff, lab, group)
                rows.append({"encoder": a.encoder, "seed": seed, "n": n, "map": name,
                             "strat_auc": float(np.nanmean(auc)), **{f"auc|{l}": float(v) for l, v in zip(names, auc)}})
                print(json.dumps({k: v for k, v in rows[-1].items() if "|" not in k}), flush=True)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/{a.encoder}_strata.csv", index=False)


def readout(a, ns=STRATA_NS, name="readout"):
    ns = tuple(int(n) for n in a.ns.split(",")) if a.ns else ns
    tag = name if a.model == CURVE_MODEL else f"{a.model}_{name}"          # default model keeps the RO file names
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    names = label_names(a.emb)
    train, test = (np.flatnonzero(cases.split == s) for s in ("train", "test"))
    x = normalize(np.stack([np.load(f"{a.emb}/{a.encoder}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64))
    y = normalize(np.load(f"{a.emb}/textmid_{a.model}.npy").astype(np.float64))
    prompts = normalize(np.load(f"{a.emb}/promptsmid_{a.model}.npy").astype(np.float64))
    lab_all = cases[names].to_numpy(float)
    lab = lab_all[test]
    group = test_strata(cases, lab, test)
    mx, my = x[train].mean(0), y[train].mean(0)
    fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
    pr = fy(prompts)
    yt, lt = fy(y[train]), lab_all[train]
    directions = {"P": (pr[0::2] - pr[1::2]).T,
                  "RP": np.stack([yt[lt[:, j] == 1].mean(0) - yt[lt[:, j] == 0].mean(0) for j in range(len(names))], 1)}
    clfs = [make_pipeline(StandardScaler(), LogisticRegression()).fit(yt, lt[:, j]) for j in range(len(names))]
    rows = []
    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(train)      # as align.downstream / M
        for n in ns:
            w = ridge(fx(x[perm[:n]]), fy(y[perm[:n]])).coef_.T
            for mname, m in (("full", w), ("rank_one", rank_one(w))):
                u = fx(x[test]) @ m
                scores = {r: u @ d for r, d in directions.items()}
                scores["TC"] = np.stack([c.decision_function(u) for c in clfs], 1)
                for r, sc in scores.items():
                    rows.append({"encoder": a.encoder, "model": a.model, "seed": seed, "n": n, "map": mname, "reader": r,
                                 "strat_auc": float(np.nanmean(stratified_auc(sc, lab, group))),
                                 "mean_auc": float(np.nanmean(auc_matrix(sc, lab)))})
                    print(json.dumps(rows[-1]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/{a.encoder}_{tag}.csv", index=False)


def readout_zero(a):
    """Readers on the zero-pair map: Gromov-Wasserstein on two disjoint halves of the train patients, exactly as
    align.downstream at n = 0 (same seeds, halves and centring), read by P and by TC; TC is fitted on all train
    reports in the same centred report space (report labels only, no pair)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from align.align import gromov_wasserstein, mini_vec2vec_official
    zero_map = {"gw": gromov_wasserstein, "minivec2vec": mini_vec2vec_official}[a.map]
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    names = label_names(a.emb)
    train, test = (np.flatnonzero(cases.split == s) for s in ("train", "test"))
    x = normalize(np.stack([np.load(f"{a.emb}/{a.encoder}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64))
    y = normalize(np.load(f"{a.emb}/textmid_{a.model}.npy").astype(np.float64))
    prompts = normalize(np.load(f"{a.emb}/promptsmid_{a.model}.npy").astype(np.float64))
    lab_all = cases[names].to_numpy(float)
    lab = lab_all[test]
    group = test_strata(cases, lab, test)
    rows = []
    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(train)
        half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]
        mx, my = x[half_a].mean(0), y[half_b].mean(0)
        fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
        w, gx, gy = zero_map(fx(x[half_a]), fy(y[half_b]), seed)
        fx, fy = (lambda v, f=fx, g=gx: g(f(v))), (lambda v, f=fy, g=gy: g(f(v)))   # the aligner's own centering on top
        u = normalize(fx(x[test]) @ w)
        pr = fy(prompts)
        yt, lt = fy(y[train]), lab_all[train]
        clfs = [make_pipeline(StandardScaler(), LogisticRegression()).fit(yt, lt[:, j]) for j in range(len(names))]
        scores = {"P": u @ (pr[0::2] - pr[1::2]).T, "TC": np.stack([c.decision_function(u) for c in clfs], 1)}
        for r, sc in scores.items():
            rows.append({"encoder": a.encoder, "model": a.model, "seed": seed, "n": 0, "map": a.map, "reader": r,
                         "strat_auc": float(np.nanmean(stratified_auc(sc, lab, group))),
                         "mean_auc": float(np.nanmean(auc_matrix(sc, lab)))})
            print(json.dumps(rows[-1]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    tag = "" if a.map == "gw" else f"_{a.map}"
    pd.DataFrame(rows).to_csv(f"{a.out}/{a.encoder}_readout_zero{tag}.csv", index=False)


def readout_summary(a):
    df = pd.concat([pd.read_csv(f"{a.out}/{e}_readout.csv") for e in ENCODERS])
    m = df[df.n == 1600].groupby(["encoder", "reader", "map"])[["strat_auc", "mean_auc"]].mean()
    res = {"per_encoder": {f"{e}/{r}/{mp}": v.to_dict() for (e, r, mp), v in m.iterrows()}}
    for r in ("RP", "TC"):
        g = lambda rd, mp, col: m.xs((rd, mp), level=("reader", "map"))[col]
        ro1 = float((g(r, "full", "strat_auc") - g("P", "full", "strat_auc")).median())
        ro2 = float((g(r, "full", "strat_auc") - g(r, "rank_one", "strat_auc")).median())
        ro3 = float((g(r, "full", "mean_auc") - g("P", "full", "mean_auc")).median())
        res[r] = {"RO1": ro1, "RO2": ro2, "RO3": ro3, "pass": bool(ro1 >= 0.02 and ro2 >= 0.02 and ro3 >= -0.01)}
    res["P_full_minus_rank_one"] = float((m.xs(("P", "full"), level=("reader", "map")).strat_auc
                                          - m.xs(("P", "rank_one"), level=("reader", "map")).strat_auc).median())
    json.dump(res, open(f"{a.out}/readout_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_encoder"}, indent=1), flush=True)


def strata_summary(a):
    df = pd.concat([pd.read_csv(f"{a.out}/{e}_strata.csv") for e in ENCODERS])
    m = df.groupby(["encoder", "n", "map"]).strat_auc.mean().unstack(["n", "map"])
    full16, one16, full1 = m[(1600, "full")], m[(1600, "rank_one")], m[(100, "full")]
    res = {"per_encoder": {e: {"full100": float(full1[e]), "full1600": float(full16[e]), "rank_one1600": float(one16[e])}
                           for e in m.index},
           "M1_median_full1600": float(full16.median()), "M2_median_full_minus_one": float((full16 - one16).median()),
           "M3_median_growth": float((full16 - full1).median())}
    res["M1_pass"] = bool(res["M1_median_full1600"] >= 0.55)
    res["M2_pass"] = bool(res["M2_median_full_minus_one"] >= 0.02)
    res["M3_pass"] = bool(res["M3_median_growth"] > 0)
    json.dump(res, open(f"{a.out}/strata_summary.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


def pca_basis(z, d):
    m = z.mean(0)
    return m, svd(z - m, full_matrices=False)[2][:d].T


def train_pcs(z, train, d=AMP_D):
    """Top-d principal coordinates of every row, with mean and axes fitted on the train rows only."""
    m, v = pca_basis(z[train], d)
    return (z - m) @ v


def canonical_variates(xp, yp, pairs):
    """CCA between principal coordinates on the given pairs; returns the variate maps (image, report) as functions
    of principal coordinates, all components, ordered by canonical correlation."""
    mx, my = xp[pairs].mean(0), yp[pairs].mean(0)
    (qx, rx), (qy, ry) = np.linalg.qr(xp[pairs] - mx), np.linalg.qr(yp[pairs] - my)
    u, _, vt = svd(qx.T @ qy)
    a, b = np.linalg.solve(rx, u), np.linalg.solve(ry, vt.T)
    return (lambda z: (z - mx) @ a), (lambda z: (z - my) @ b)


def probe_auc(v, lab, fit, test, group):
    """Per finding: standardised logistic regression on v[fit], stratified AUC on v[test]; mean over findings."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    scores = np.empty((len(test), lab.shape[1]))
    for j in range(lab.shape[1]):
        if len(np.unique(lab[fit, j])) < 2:
            raise ValueError(f"finding {j}: one class among the {len(fit)} probe-fit patients")
        clf = make_pipeline(StandardScaler(), LogisticRegression()).fit(v[fit], lab[fit, j])
        scores[:, j] = clf.decision_function(v[test])
    return float(np.nanmean(stratified_auc(scores, lab[test], group)))


def probes(a):
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    names = label_names(a.emb)
    train, test = (np.flatnonzero(cases.split == s) for s in ("train", "test"))
    xp = train_pcs(np.stack([np.load(f"{a.emb}/{a.encoder}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64), train)
    yp = train_pcs(np.load(f"{a.emb}/textmid_{CURVE_MODEL}.npy").astype(np.float64), train)
    lab = cases[names].to_numpy(float)
    group = test_strata(cases, lab[test], test)
    k = len(json.load(open(f"{a.spiked}/{a.encoder}_estimates.json"))[f"{CURVE_MODEL}/mid/{AMP_D}"]["r_hat"])
    if k == 0:
        raise ValueError(f"{a.encoder}: no shared component in align.spiked")
    rows = []
    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(train)      # as align.downstream
        pairs, fit = perm[:PROBE_PAIRS], perm[PROBE_PAIRS:]
        fx, fy = canonical_variates(xp, yp, pairs)
        for side, f, z in (("image", fx, xp), ("report", fy, yp)):
            v = f(z)
            row = {"encoder": a.encoder, "seed": seed, "side": side, "k": k,
                   "auc_1": probe_auc(v[:, :1], lab, fit, test, group),
                   "auc_1k": probe_auc(v[:, :k], lab, fit, test, group)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/{a.encoder}_probes.csv", index=False)


def probes_control(a):
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    names = label_names(a.emb)
    train, test = (np.flatnonzero(cases.split == s) for s in ("train", "test"))
    xp = train_pcs(np.stack([np.load(f"{a.emb}/{a.encoder}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64), train)
    yp = train_pcs(np.load(f"{a.emb}/textmid_{CURVE_MODEL}.npy").astype(np.float64), train)
    lab = cases[names].to_numpy(float)
    group = test_strata(cases, lab[test], test)
    k = len(json.load(open(f"{a.spiked}/{a.encoder}_estimates.json"))[f"{CURVE_MODEL}/mid/{AMP_D}"]["r_hat"])
    if k == 0:
        raise ValueError(f"{a.encoder}: no shared component in align.spiked")
    rows = []
    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(train)      # as probes
        pairs, fit = perm[:PROBE_PAIRS], perm[PROBE_PAIRS:]
        fx, fy = canonical_variates(xp, yp, pairs)
        rng = np.random.default_rng(seed + 7000)
        for side, f, z in (("image", fx, xp), ("report", fy, yp)):
            rand = [probe_auc(z @ np.linalg.qr(rng.standard_normal((AMP_D, k)))[0], lab, fit, test, group)
                    for _ in range(RANDOM_DRAWS)]
            row = {"encoder": a.encoder, "seed": seed, "side": side, "k": k,
                   "auc_cca": probe_auc(f(z)[:, :k], lab, fit, test, group),
                   "auc_pcs": probe_auc(z[:, :k], lab, fit, test, group), "auc_random": float(np.mean(rand))}
            rows.append(row)
            print(json.dumps(row), flush=True)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/{a.encoder}_probes_control.csv", index=False)


def probes_control_summary(a):
    df = pd.concat([pd.read_csv(f"{a.out}/{e}_probes_control.csv") for e in ENCODERS])
    m = df.groupby(["side", "encoder"])[["auc_cca", "auc_pcs", "auc_random"]].mean()
    res = {"per_encoder": {f"{s}/{e}": r.to_dict() for (s, e), r in m.iterrows()}}
    for side in ("image", "report"):
        g = m.loc[side]
        res[f"{side}_median_cca_minus_pcs"] = float((g.auc_cca - g.auc_pcs).median())
        res[f"{side}_median_cca_minus_random"] = float((g.auc_cca - g.auc_random).median())
    res["R1_pass"] = bool(res["image_median_cca_minus_pcs"] >= 0.02 and res["image_median_cca_minus_random"] >= 0.02)
    json.dump(res, open(f"{a.out}/probes_control_summary.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


def reader(a):
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    names = label_names(a.emb)
    train, test = (np.flatnonzero(cases.split == s) for s in ("train", "test"))
    x = np.stack([np.load(f"{a.emb}/{a.encoder}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64)
    y = np.load(f"{a.emb}/textmid_{CURVE_MODEL}.npy").astype(np.float64)
    prompts = np.load(f"{a.emb}/promptsmid_{CURVE_MODEL}.npy").astype(np.float64)
    (mxa, vx), (mya, vy) = pca_basis(x[train], AMP_D), pca_basis(y[train], AMP_D)
    xp, pp = (x - mxa) @ vx, (prompts - mya) @ vy
    yp = (y - mya) @ vy
    lab = cases[names].to_numpy(float)[test]
    group = test_strata(cases, lab, test)
    xn, yn, prn = normalize(x), normalize(y), normalize(prompts)     # ridge baseline exactly as M (strata)
    mxn, myn = xn[train].mean(0), yn[train].mean(0)
    fx, fy = (lambda v: normalize(v - mxn)), (lambda v: normalize(v - myn))
    pr = fy(prn)
    diff = (pr[0::2] - pr[1::2]).T
    r_hat = np.array(json.load(open(f"{a.spiked}/{a.encoder}_estimates.json"))[f"{CURVE_MODEL}/mid/{AMP_D}"]["r_hat"])
    rows = []
    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(train)      # as align.downstream
        for n in READER_NS:
            k = int((r_hat > rc(AMP_D / n, AMP_D / n)).sum())
            if k == 0:
                rows.append({"encoder": a.encoder, "seed": seed, "n": n, "k": 0, "strat_auc": np.nan, "mean_auc": np.nan,
                             "ridge_strat_auc": np.nan})
                continue
            pairs = perm[:n]
            mx, my = xp[pairs].mean(0), yp[pairs].mean(0)
            (qx, rx), (qy, ry) = np.linalg.qr(xp[pairs] - mx), np.linalg.qr(yp[pairs] - my)
            u, sv, vt = svd(qx.T @ qy)
            img = (xp[test] - mx) @ np.linalg.solve(rx, u[:, :k])
            pc = (pp - my) @ np.linalg.solve(ry, vt[:k].T)
            scores = (img * sv[:k]) @ (pc[0::2] - pc[1::2]).T
            w = ridge(fx(xn[pairs]), fy(yn[pairs])).coef_.T
            rows.append({"encoder": a.encoder, "seed": seed, "n": n, "k": k,
                         "strat_auc": float(np.nanmean(stratified_auc(scores, lab, group))),
                         "mean_auc": float(np.nanmean(auc_matrix(scores, lab))),
                         "ridge_strat_auc": float(np.nanmean(stratified_auc(fx(xn[test]) @ w @ diff, lab, group)))})
            print(json.dumps(rows[-1]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/{a.encoder}_reader.csv", index=False)


def reader_summary(a):
    rd = pd.concat([pd.read_csv(f"{a.out}/{e}_reader.csv") for e in ENCODERS]).groupby(["encoder", "n"])[
        ["k", "strat_auc", "mean_auc", "ridge_strat_auc"]].mean()
    dn = pd.read_csv(f"{a.downstream}/curves.csv").set_index(["encoder", "n"]).mean_auc
    res = {"per_encoder": {}}
    for e in ENCODERS:
        res["per_encoder"][e] = {int(n): {"k": float(rd.loc[(e, n), "k"]), "reader_strat": float(rd.loc[(e, n), "strat_auc"]),
                                          "reader_mean": float(rd.loc[(e, n), "mean_auc"]),
                                          "ridge_strat": float(rd.loc[(e, n), "ridge_strat_auc"]),
                                          "ridge_mean": float(dn.loc[(e, n)])} for n in READER_NS}
    d_strat = [v[1600]["reader_strat"] - v[1600]["ridge_strat"] for v in res["per_encoder"].values()]
    d_mean = [v[1600]["reader_mean"] - v[1600]["ridge_mean"] for v in res["per_encoder"].values()]
    res["RD1_median_strat_gain"], res["RD2_median_mean_gain"] = float(np.median(d_strat)), float(np.median(d_mean))
    res["RD1_pass"] = bool(res["RD1_median_strat_gain"] >= 0.02)
    res["RD2_pass"] = bool(res["RD2_median_mean_gain"] >= -0.01)
    json.dump(res, open(f"{a.out}/reader_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_encoder"}, indent=1), flush=True)


def probes_summary(a):
    df = pd.concat([pd.read_csv(f"{a.out}/{e}_probes.csv") for e in ENCODERS])
    m = df.groupby(["side", "encoder"])[["auc_1", "auc_1k"]].mean()
    gain = (m.auc_1k - m.auc_1)
    res = {"per_encoder": {f"{s}/{e}": {"auc_1": float(r.auc_1), "auc_1k": float(r.auc_1k)} for (s, e), r in m.iterrows()},
           "Q1_median_gain_image": float(gain.loc["image"].median()), "Q2_median_gain_report": float(gain.loc["report"].median())}
    res["Q1_pass"] = bool(res["Q1_median_gain_image"] >= 0.02)
    json.dump(res, open(f"{a.out}/probes_summary.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


CURVE_NS = (100, 200, 400, 800, 1600)
USE_GAIN, NO_USE = 0.02, 3200


def readout_curve_summary(a):
    from scipy.stats import spearmanr

    ns = tuple(int(n) for n in a.ns.split(",")) if a.ns else CURVE_NS
    df = pd.concat([pd.read_csv(f"{a.out}/{e}_readout_curve.csv") for e in ENCODERS])
    m = df[df["map"] == "full"].groupby(["encoder", "reader", "n"]).strat_auc.mean().unstack("reader")
    rows = []
    for e in ENCODERS:
        g = (m.loc[e, "TC"] - m.loc[e, "P"]).reindex(ns)
        n_use = NO_USE
        for (n0, g0), (n1, g1) in zip(g.items(), list(g.items())[1:]):
            if g0 >= USE_GAIN:
                n_use = n0
                break
            if g1 >= USE_GAIN:
                n_use = float(np.exp(np.log(n0) + (USE_GAIN - g0) / (g1 - g0) * (np.log(n1) - np.log(n0))))
                break
        r = json.load(open(f"{a.spiked}/{e}_estimates.json"))[f"{CURVE_MODEL}/mid/{AMP_D}"]["r_hat"]
        r1 = r[0] if r else 0.0
        n_law = AMP_D * (1 + 1 / r[1]) if len(r) > 1 and r[1] > 0 else np.inf
        rows.append({"encoder": e, "n_use": n_use, "n_law": n_law, "inv_r1": 1 / r1 if r1 > 0 else np.inf,
                     **{f"gain_{n}": float(g[n]) for n in ns}})
    t = pd.DataFrame(rows)
    rho_law = float(spearmanr(t.n_law, t.n_use).statistic)
    rho_r1 = float(spearmanr(t.inv_r1, t.n_use).statistic)
    res = {"table": t.replace({np.inf: None}).to_dict("records"), "rho_law": rho_law, "rho_r1": rho_r1,
           "U1_pass": bool(rho_law >= 0.7), "U2_pass": bool(abs(rho_law) - abs(rho_r1) >= 0.1)}
    json.dump(res, open(f"{a.out}/readout_curve_summary.json", "w"), indent=1)
    print(t.round(3).to_string())
    print(json.dumps({k: v for k, v in res.items() if k != "table"}, indent=1), flush=True)


def cka(a):
    from align.align import gromov_wasserstein, linear_cka
    from align.baselines import zero_shot_scores

    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    names = label_names(a.emb)
    sel, train, test = (np.flatnonzero(cases.split == s) for s in ("sel", "train", "test"))
    x = np.stack([np.load(f"{a.emb}/{a.encoder}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64)
    y = np.load(f"{a.emb}/textmid_{CURVE_MODEL}.npy").astype(np.float64)
    prompts = np.load(f"{a.emb}/promptsmid_{CURVE_MODEL}.npy").astype(np.float64)
    lab = cases[names].to_numpy(float)[test]
    ox, oy, k = shared_directions(x[sel], y[sel], np.random.default_rng(0))
    rng = np.random.default_rng(9000)
    rx, ry = (np.linalg.qr(rng.standard_normal((x.shape[1], 10)))[0], np.linalg.qr(rng.standard_normal((y.shape[1], 10)))[0])
    pairs = np.random.default_rng(1).choice(train, min(CKA_PAIRS, len(train)), replace=False)
    rows = []
    arms = {"oracle": (ox, oy), "random": (rx, ry)}
    factors = [float(v) for v in a.factors.split(",")] if a.factors else FACTORS
    for arm in [a.arm] if a.arm else arms:
        ux, uy = arms[arm]
        for factor in factors:
            gx, gy = amplify_map(ux, x[sel].mean(0), factor), amplify_map(uy, y[sel].mean(0), factor)
            xa, ya, pa = normalize(gx(x)), normalize(gy(y)), normalize(gy(prompts))
            c = linear_cka(xa[pairs] - xa[pairs].mean(0), ya[pairs] - ya[pairs].mean(0))
            aucs = []
            for seed in SEEDS:
                perm = np.random.default_rng(seed).permutation(train)
                half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]
                mx, my = xa[half_a].mean(0), ya[half_b].mean(0)
                fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
                w, gx, gy = gromov_wasserstein(fx(xa[half_a]), fy(ya[half_b]), seed)
                fx, fy = (lambda v, f=fx, g=gx: g(f(v))), (lambda v, f=fy, g=gy: g(f(v)))   # the aligner's own centering on top
                aucs.append(float(np.nanmean(auc_matrix(zero_shot_scores(normalize(fx(xa[test]) @ w), fy(pa)), lab))))
            rows.append({"encoder": a.encoder, "arm": arm, "factor": factor, "cka": c,
                         "f_image": shared_fraction(xa[train], ox), "f_report": shared_fraction(ya[train], oy),
                         "gw_auc": float(np.mean(aucs))})
            print(json.dumps(rows[-1]), flush=True)
    os.makedirs(a.out, exist_ok=True)
    part = f"_{a.arm or 'all'}_{(a.factors or 'all').replace(',', '-')}" if a.arm or a.factors else ""   # one array task's share
    pd.DataFrame(rows).to_csv(f"{a.out}/{a.encoder}_cka{part}.csv", index=False)


def cka_cells(out, e):
    """<e>_cka.csv, or, when the cells ran as array shares (cka --arm/--factors), their union: every arm x factor once."""
    whole = f"{out}/{e}_cka.csv"
    if os.path.exists(whole):
        return pd.read_csv(whole)
    parts = sorted(glob.glob(f"{out}/{e}_cka_*.csv"))
    if not parts:
        raise FileNotFoundError(whole)
    df = pd.concat([pd.read_csv(p) for p in parts])
    cells = set(zip(df.arm, df.factor.astype(float)))
    if len(df) != len(cells) or cells != {(arm, float(f)) for arm in ("oracle", "random") for f in FACTORS}:
        raise ValueError(f"{e}: cka shares are incomplete or duplicated: {sorted(cells)}")
    return df


def cka_summary(a):
    from scipy.stats import spearmanr

    df = pd.concat([cka_cells(a.out, e) for e in CKA_ENCODERS])
    df["f_min"] = df[["f_image", "f_report"]].min(1)
    rho_f = float(spearmanr(df.f_min, df.gw_auc).statistic)
    rho_c = float(spearmanr(df.cka, df.gw_auc).statistic)
    disc = []
    v = df.to_dict("records")
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            if abs(v[i]["cka"] - v[j]["cka"]) <= 0.02 and abs(v[i]["gw_auc"] - v[j]["gw_auc"]) >= 0.1:
                disc.append([{k: v[i][k] for k in ("encoder", "arm", "factor", "cka", "f_min", "gw_auc")},
                             {k: v[j][k] for k in ("encoder", "arm", "factor", "cka", "f_min", "gw_auc")}])
    res = {"rho_f_auc": rho_f, "rho_cka_auc": rho_c, "K1_pass": bool(rho_f - rho_c >= 0.1), "discordant": disc,
           "cells": df.to_dict("records")}
    json.dump(res, open(f"{a.out}/cka_summary.json", "w"), indent=1)
    print(df.round(3).to_string())
    print(json.dumps({k: v for k, v in res.items() if k not in ("cells",)}, indent=1), flush=True)


def scorerank(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    names = label_names(a.emb)
    train, test = (np.flatnonzero(cases.split == s) for s in ("train", "test"))
    x = normalize(np.stack([np.load(f"{a.emb}/{a.encoder}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64))
    y = normalize(np.load(f"{a.emb}/textmid_{a.model}.npy").astype(np.float64))
    prompts = normalize(np.load(f"{a.emb}/promptsmid_{a.model}.npy").astype(np.float64))
    lab_all = cases[names].to_numpy(float)
    lab = lab_all[test]
    group = test_strata(cases, lab, test)
    mx, my = x[train].mean(0), y[train].mean(0)
    fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
    pr = fy(prompts)
    yt, lt = fy(y[train]), lab_all[train]
    clfs = [make_pipeline(StandardScaler(), LogisticRegression()).fit(yt, lt[:, j]) for j in range(len(names))]
    dirs = {"P": (pr[0::2] - pr[1::2]).T, "TC": np.stack([c[-1].coef_[0] / c[0].scale_ for c in clfs], 1)}
    rows = []
    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(train)
        w = ridge(fx(x[perm[:SCORE_N]]), fy(y[perm[:SCORE_N]])).coef_.T
        _, s, vt = svd(w, full_matrices=False)
        u = fx(x[test]) @ w
        for r, d in dirs.items():
            e = (s[:, None] * (vt @ d)) ** 2
            sc = u @ d if r == "P" else np.stack([c.decision_function(u) for c in clfs], 1)
            scc = sc - sc.mean(0)
            sv = svd(scc, compute_uv=False) ** 2
            p = sv / sv.sum()
            row = {"encoder": a.encoder, "seed": seed, "reader": r, "axis1_share": float(np.median(e[0] / e.sum(0))),
                   "eff_rank": float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))}
            if r == "P":
                uu, ss, vv = svd(scc, full_matrices=False)
                deflated = scc - ss[0] * np.outer(uu[:, 0], vv[0])
                row["strat_auc"] = float(np.nanmean(stratified_auc(scc, lab, group)))
                row["strat_auc_deflated"] = float(np.nanmean(stratified_auc(deflated, lab, group)))
            rows.append(row)
            print(json.dumps(row), flush=True)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/{a.encoder}_scorerank.csv", index=False)


def scorerank_summary(a):
    df = pd.concat([pd.read_csv(f"{a.out}/{e}_scorerank.csv") for e in ENCODERS])
    m = df.groupby(["encoder", "reader"])[["axis1_share", "eff_rank"]].mean().unstack("reader")
    pd_ = df[df.reader == "P"].groupby("encoder")[["strat_auc", "strat_auc_deflated"]].mean()
    res = {"per_encoder": {e: {"axis1_P": float(m.loc[e, ("axis1_share", "P")]), "axis1_TC": float(m.loc[e, ("axis1_share", "TC")]),
                               "effrank_P": float(m.loc[e, ("eff_rank", "P")]), "effrank_TC": float(m.loc[e, ("eff_rank", "TC")]),
                               "P_strat": float(pd_.loc[e, "strat_auc"]), "P_strat_deflated": float(pd_.loc[e, "strat_auc_deflated"])}
                           for e in ENCODERS}}
    res["median_axis1_P"] = float(m[("axis1_share", "P")].median())
    res["median_axis1_TC"] = float(m[("axis1_share", "TC")].median())
    res["median_effrank_P"] = float(m[("eff_rank", "P")].median())
    res["median_effrank_TC"] = float(m[("eff_rank", "TC")].median())
    res["E1_pass"] = bool(res["median_axis1_P"] >= 0.8 and res["median_axis1_TC"] <= 0.5)
    res["E2_pass"] = bool(res["median_effrank_P"] < res["median_effrank_TC"])
    json.dump(res, open(f"{a.out}/scorerank_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_encoder"}, indent=1), flush=True)


def solver_diag(a):
    """Why does Test K amplification leave chance with POT's Gromov-Wasserstein (B3, before f804b70) but not with the
    official aligner? Same encoder, text model, halves and oracle amplification as B3; per seed, both zero-pair maps and
    a paired-Procrustes reference on all train pairs. Per map: AUC, FOSCTTM, the share of the mapped image shared
    subspace that lands in the report shared subspace, the test correlation along each report shared direction, and
    the GW loss (square loss, cosine distances, 1,500-point subsample) of its coupling."""
    import ot
    from scipy.optimize import linear_sum_assignment

    from align.align import foscttm, gromov_wasserstein, polar
    from align.baselines import zero_shot_scores

    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    names = label_names(a.emb)
    sel, train, test = (np.flatnonzero(cases.split == s) for s in ("sel", "train", "test"))
    x = np.stack([np.load(f"{a.emb}/{a.encoder}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(np.float64)
    y = np.load(f"{a.emb}/textmid_{CURVE_MODEL}.npy").astype(np.float64)
    prompts = np.load(f"{a.emb}/promptsmid_{CURVE_MODEL}.npy").astype(np.float64)
    lab = cases[names].to_numpy(float)[test]
    ux, uy, k = shared_directions(x[sel], y[sel], np.random.default_rng(0))
    rows = []
    for factor in [float(v) for v in a.factors.split(",")]:
        ax, ay = amplify_map(ux, x[sel].mean(0), factor), amplify_map(uy, y[sel].mean(0), factor)
        xa, ya, pa = normalize(ax(x)), normalize(ay(y)), normalize(ay(prompts))
        for seed in SEEDS:
            perm = np.random.default_rng(seed).permutation(train)
            half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]
            mx, my = xa[half_a].mean(0), ya[half_b].mean(0)
            fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
            xs_all, ys_all = fx(xa[half_a]), fy(ya[half_b])
            rng = np.random.default_rng(seed)       # the B3 subsample: same draws as the pre-f804b70 solver
            xi = rng.choice(len(xs_all), min(1500, len(xs_all)), replace=False)
            yi = rng.choice(len(ys_all), min(1500, len(ys_all)), replace=False)
            xs, ys = xs_all[xi], ys_all[yi]
            c1, c2 = 1 - xs @ xs.T, 1 - ys @ ys.T
            p, q = ot.unif(len(xs)), ot.unif(len(ys))
            const, h1, h2 = ot.gromov.init_matrix(c1, c2, p, q, "square_loss")
            gw_loss = lambda t: float(ot.gromov.gwloss(const, h1, h2, t))

            def hard(sim):
                r, c = linear_sum_assignment(-sim)
                t = np.zeros_like(sim)
                t[r, c] = 1 / len(r)
                return t

            t_pot = ot.gromov.gromov_wasserstein(c1, c2, p, q, loss_fun="square_loss", max_iter=500)
            maps = {"pot": (polar(xs.T @ t_pot @ ys), fx, fy)}
            w, gx, gy = gromov_wasserstein(xs_all, ys_all, seed)
            maps["official"] = (w, (lambda v, f=fx, g=gx: g(f(v))), (lambda v, f=fy, g=gy: g(f(v))))
            maps["paired"] = (polar(fx(xa[train]).T @ fy(ya[train])), fx, fy)
            for name, (w, px, py) in maps.items():
                u, yt = normalize(px(xa[test]) @ w), py(ya[test])
                m = ux.T @ w
                corr = [float(np.corrcoef(u @ uy[:, j], yt @ uy[:, j])[0, 1]) for j in range(k)]
                row = {"encoder": a.encoder, "factor": factor, "seed": seed, "map": name, "k": k,
                       "mean_auc": float(np.nanmean(auc_matrix(zero_shot_scores(u, py(pa)), lab))),
                       "foscttm": foscttm(u @ yt.T),
                       "subspace_overlap": float(((m @ uy) ** 2).sum() / (m ** 2).sum()),
                       "dir_corr_mean": float(np.mean(corr)), "dir_corr": corr,
                       "gw_loss_hard": gw_loss(hard(px(xa[half_a][xi]) @ w @ py(ya[half_b][yi]).T))}
                if name == "pot":
                    row["gw_loss_soft"] = gw_loss(t_pot)
                rows.append(row)
                print(json.dumps(row), flush=True)
    os.makedirs(a.out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_json(f"{a.out}/{a.encoder}_solver_diag_{a.factors.replace(',', '-')}.jsonl", orient="records", lines=True)
    print(df.groupby(["factor", "map"])[["mean_auc", "foscttm", "subspace_overlap", "dir_corr_mean", "gw_loss_hard"]]
          .mean().round(4).to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["spectrum", "prospective", "prospective-summary", "strata", "strata-summary",
                                       "probes", "probes-summary", "probes-control", "probes-control-summary",
                                       "reader", "reader-summary", "readout", "readout-summary", "readout-curve",
                                       "readout-curve-summary", "readout-zero", "cka", "cka-summary", "scorerank", "scorerank-summary", "solver-diag"])
    ap.add_argument("--emb")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder")
    ap.add_argument("--factors", help="spectrum, cka, solver-diag: comma-separated factors (default 1..32 as align.budget)")
    ap.add_argument("--map", choices=["gw", "minivec2vec"], default="gw",
                    help="readout-zero: zero-pair map (official aligner, or official mini-vec2vec)")
    ap.add_argument("--arm", choices=["oracle", "random"], help="cka: run one arm only")
    ap.add_argument("--spiked")
    ap.add_argument("--downstream")
    ap.add_argument("--model", default=CURVE_MODEL, help="text model of the readout steps (embedding file suffix)")
    ap.add_argument("--ns", help="readout steps: comma-separated pair counts replacing the step default")
    a = ap.parse_args()
    {"spectrum": spectrum, "prospective": prospective, "prospective-summary": prospective_summary,
     "strata": strata, "strata-summary": strata_summary, "probes": probes, "probes-summary": probes_summary,
     "probes-control": probes_control, "probes-control-summary": probes_control_summary,
     "reader": reader, "reader-summary": reader_summary, "readout": readout, "readout-summary": readout_summary,
     "readout-curve": lambda a: readout(a, CURVE_NS, "readout_curve"), "readout-curve-summary": readout_curve_summary, "readout-zero": readout_zero,
     "cka": cka, "cka-summary": cka_summary, "scorerank": scorerank, "scorerank-summary": scorerank_summary,
     "solver-diag": solver_diag}[a.step](a)
    print("DECISIVE_EXIT rc=0", flush=True)


if __name__ == "__main__":
    main()
