"""External cohort: do the CT-RATE findings hold on BIMCV-R?

TRANSFER: the four maps of align.anchor_zero (anchor ridge, anchor Procrustes, random pseudo pairs, 100 true pairs),
fitted on CT-RATE exactly as there (same seeds, halves and pairs), applied unchanged to every BIMCV case. Centring:
"own" (primary) subtracts the BIMCV image mean and the BIMCV report mean, each over the unpaired BIMCV set; "ctrate"
keeps the CT-RATE half means. Read-out on the CT-RATE labels that have a PadChest counterpart and at least MIN_POS
positives among the BIMCV cases; the same labels are read on the CT-RATE test patients for the in-domain reference.
WITHIN: align.anchor_zero.fit_seed run on the BIMCV train patients (its own anchors from TotalSegmentator), evaluated
on the BIMCV test patients.
Intervals: patient bootstrap (N_BOOT) of the seed-mean of the mean AUC.

    python -m align.external --ctrate DIR --bimcv DIR --out DIR --encoder E [--model 4B]
    python -m align.external --out DIR --summary
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from align.align import SEEDS, foscttm, gromov_wasserstein, label_names, normalize
from align.anchor_zero import ANCHORS, fit_seed, image_anchors
from align.baselines import N_BOOT, auc_matrix, zero_shot_scores
from align.bimcv import LABEL_MAP

MIN_POS = 20
EQUIV = (0.45, 0.55)        # E1: zero-pair GW mean AUC interval inside this band = chance
E2_MIN = 0.55               # E2: anchor-ridge non-anchor mean AUC lower bound above this
CONTROL_GAP = -0.05         # positive control: median over encoders of BIMCV within true100 - CT-RATE true100
METHODS = ("anchor_ridge", "anchor_procrustes", "random_ridge", "true100_ridge")
ENCODERS = ("voxelfm", "voxelfm_patch", "ctfm", "vjepa2_l", "vjepa2_h", "vjepa2_g", "wan_vae", "wan_dit", "ltx_vae",
            "dinov3_2d", "dinov3_25d", "dalect0", "dalect1s", "videomae_b", "dinov2_2d")


def anchor_zero_draws(seed):
    """align.anchor_zero draws one extra permutation at seed 0 (its diagnostic) before the random pseudo pairs; the
    same draw here keeps random_ridge identical to the CT-RATE run."""
    return (lambda rng, half_a, half_b, txt_anchor, img_p, txt_p: rng.permutation(txt_p)) if seed == 0 else None


def images(emb, encoder, keys):
    return normalize(np.stack([np.load(f"{emb}/{encoder}/{os.path.basename(k)}.npy") for k in keys]).astype(np.float64))


def reports(emb, model, n):
    y = normalize(np.load(f"{emb}/textmid_{model}.npy").astype(np.float64))
    if len(y) != n:
        raise ValueError(f"{emb}: {len(y)} report embeddings for {n} cases")
    return y


def bimcv_cases(emb):
    """Cases in the order of the report embeddings (align.extract.run_texts embeds every case), rejected volumes
    marked so that they can be dropped after the embeddings are aligned."""
    cases = pd.read_json(f"{emb}/cases.jsonl", lines=True)
    rejected = {os.path.basename(p)[:-5] for p in glob.glob(f"{emb}/rejected/*.json")}
    return cases.assign(rejected=cases.key.isin(rejected))


def scores(w, gx, gy, x, y, pr):
    u = normalize(gx(x) @ w)
    return zero_shot_scores(u, pr), foscttm(u @ gy(y).T)


def boot_mean_auc(score_list, lab, rng):
    """score_list: one (n, L) score matrix per seed. Point estimate and patient-bootstrap 95 % interval of the mean over
    seeds of the mean AUC over labels; the same resamples serve every method (paired differences)."""
    idx = [rng.integers(0, len(lab), len(lab)) for _ in range(N_BOOT)]
    point = float(np.mean([np.nanmean(auc_matrix(s, lab)) for s in score_list]))
    boot = np.array([np.mean([np.nanmean(auc_matrix(s[i], lab[i])) for s in score_list]) for i in idx])
    return point, boot


def interval(point, boot):
    return {"point": point, "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5))}


def run(a):
    names = label_names(a.ctrate)
    ct = pd.read_json(f"{a.ctrate}/cases.jsonl", lines=True)
    x, y = images(a.ctrate, a.encoder, ct["key"]), reports(a.ctrate, a.model, len(ct))
    prompts = normalize(np.load(f"{a.ctrate}/promptsmid_{a.model}.npy").astype(np.float64))
    img_anchor = image_anchors(f"{a.ctrate}/measures.csv", ct["key"])
    idx = [names.index(n) for n in ANCHORS]
    train, test = (np.flatnonzero(ct.split == s) for s in ("train", "test"))

    bc = bimcv_cases(a.bimcv)
    by = reports(a.bimcv, a.model, len(bc))[~bc.rejected.to_numpy()]
    bc = bc[~bc.rejected].reset_index(drop=True)
    bx = images(a.bimcv, a.encoder, bc["key"])
    evals = [n for n in LABEL_MAP if bc[n].sum() >= MIN_POS]
    ev = [names.index(n) for n in evals]
    blab, clab = bc[evals].to_numpy(float), ct[evals].to_numpy(float)
    gx_own, gy_own = (lambda v: normalize(v - bx.mean(0))), (lambda v: normalize(v - by.mean(0)))

    sc = {(c, m): [] for c in ("own", "ctrate", "ctrate_test") for m in METHODS}
    fos = {(c, m): [] for c in ("own", "ctrate", "ctrate_test") for m in METHODS}
    for seed in SEEDS:
        maps, fx, fy, pr = fit_seed(x, y, prompts, img_anchor, idx, train, seed, anchor_zero_draws(seed))
        pr_own = gy_own(prompts)
        for m, w in maps.items():
            for c, (gx, gy, xx, yy, p) in {"own": (gx_own, gy_own, bx, by, pr_own), "ctrate": (fx, fy, bx, by, pr),
                                           "ctrate_test": (fx, fy, x[test], y[test], pr)}.items():
                s, f = scores(w, gx, gy, xx, yy, p)
                sc[(c, m)].append(s[:, ev])
                fos[(c, m)].append(f)
        print(f"[external] {a.encoder} seed={seed} own: " + " ".join(
            f"{m}={np.nanmean(auc_matrix(sc[('own', m)][-1], blab)):.3f}" for m in METHODS), flush=True)

    out = {"encoder": a.encoder, "model": a.model, "eval_labels": evals, "min_pos": MIN_POS, "n_bimcv": len(bc),
           "n_ctrate_test": len(test), "prevalence_bimcv": {n: float(bc[n].mean()) for n in evals},
           "prevalence_ctrate_test": {n: float(ct.loc[test, n].mean()) for n in evals}, "transfer": {}}
    for c in ("own", "ctrate", "ctrate_test"):
        lab = clab[test] if c == "ctrate_test" else blab
        boots = {}
        for m in METHODS:
            rng = np.random.default_rng(12345)        # identical resamples for every method of one evaluation set
            point, boots[m] = boot_mean_auc(sc[(c, m)], lab, rng)
            out["transfer"].setdefault(c, {})[m] = {
                "mean_auc": interval(point, boots[m]), "foscttm": float(np.mean(fos[(c, m)])),
                "per_label_auc": dict(zip(evals, np.nanmean([auc_matrix(s, lab) for s in sc[(c, m)]], 0).tolist()))}
        diff = out["transfer"][c]["anchor_ridge"]["mean_auc"]["point"] - out["transfer"][c]["true100_ridge"]["mean_auc"]["point"]
        out["transfer"][c]["anchor_minus_true100"] = interval(diff, boots["anchor_ridge"] - boots["true100_ridge"])

    within = []
    btrain, btest = (np.flatnonzero(bc.split == s) for s in ("train", "test"))
    b_anchor = image_anchors(f"{a.bimcv}/measures.csv", bc["key"])
    blab_test = blab[btest]
    for seed in SEEDS:
        maps, fx, fy, pr = fit_seed(bx, by, prompts, b_anchor, idx, btrain, seed, anchor_zero_draws(seed))
        for m, w in maps.items():
            s, f = scores(w, fx, fy, bx[btest], by[btest], pr)
            within.append({"seed": seed, "method": m, "mean_auc": float(np.nanmean(auc_matrix(s[:, ev], blab_test))), "foscttm": f})
    wd = pd.DataFrame(within).groupby("method")[["mean_auc", "foscttm"]].mean()
    out["within"] = {m: {"mean_auc": float(r.mean_auc), "foscttm": float(r.foscttm)} for m, r in wd.iterrows()}
    out["within_true100_minus_ctrate_true100"] = float(
        wd.loc["true100_ridge", "mean_auc"] - out["transfer"]["ctrate_test"]["true100_ridge"]["mean_auc"]["point"])

    # Zero-pair Gromov-Wasserstein: fitted on the unpaired train halves (no pair, no label); CT-RATE read on its test
    # patients, BIMCV read on all its cases (labels enter only the read-out).
    gw = {"ctrate_test": [], "bimcv_all": []}
    for seed in SEEDS:
        for name, xx, yy, tr, ex, ey in (("ctrate_test", x, y, train, x[test], y[test]),
                                         ("bimcv_all", bx, by, btrain, bx, by)):
            perm = np.random.default_rng(seed).permutation(tr)
            half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]
            mx, my = xx[half_a].mean(0), yy[half_b].mean(0)
            fx, fy = (lambda v, m=mx: normalize(v - m)), (lambda v, m=my: normalize(v - m))
            w, gx, gy = gromov_wasserstein(fx(xx[half_a]), fy(yy[half_b]), seed)
            fx, fy = (lambda v, f=fx, g=gx: g(f(v))), (lambda v, f=fy, g=gy: g(f(v)))   # the aligner's own centering on top
            s, _ = scores(w, fx, fy, ex, ey, fy(prompts))
            gw[name].append(s[:, ev])
    for name, lab in (("ctrate_test", clab[test]), ("bimcv_all", blab)):
        point, boot = boot_mean_auc(gw[name], lab, np.random.default_rng(12345))
        out.setdefault("gw", {})[name] = interval(point, boot)
    non_anchor = [n for n in evals if n not in ANCHORS]
    na = [evals.index(n) for n in non_anchor]
    point, boot = boot_mean_auc([s[:, na] for s in sc[("own", "anchor_ridge")]], blab[:, na], np.random.default_rng(12345))
    out["transfer"]["own"]["anchor_ridge"]["non_anchor_mean_auc"] = interval(point, boot)
    out["non_anchor_labels"] = non_anchor
    out["within_gate_pass"] = bool(((wd.loc[["anchor_ridge", "anchor_procrustes"], "foscttm"] <= 0.40)
                                    & (wd.loc[["anchor_ridge", "anchor_procrustes"], "mean_auc"] >= 0.60)).any())
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/{a.encoder}.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("encoder", "eval_labels", "n_bimcv", "within_gate_pass")}), flush=True)
    for c in ("own", "ctrate", "ctrate_test"):
        print(f"[external] {a.encoder} {c}: " + " ".join(
            f"{m}={out['transfer'][c][m]['mean_auc']['point']:.3f}" for m in METHODS)
            + f" anchor-true100={out['transfer'][c]['anchor_minus_true100']}", flush=True)


def summary(out_dir):
    """Pre-registered criteria over all encoders."""
    rs = [json.load(open(f"{out_dir}/{e}.json")) for e in ENCODERS]
    if len({(r["model"], r["n_bimcv"], tuple(r["eval_labels"])) for r in rs}) != 1:
        raise ValueError("encoder results differ in model, case count or labels")
    t = lambda r, c, m: r["transfer"][c][m]["mean_auc"]
    n = len(rs)
    inside = lambda iv: EQUIV[0] <= iv["lo"] and iv["hi"] <= EQUIV[1]
    e1_ct = sum(inside(r["gw"]["ctrate_test"]) for r in rs)
    e1 = sum(inside(r["gw"]["bimcv_all"]) for r in rs)
    e2 = sum(r["transfer"]["own"]["anchor_ridge"]["non_anchor_mean_auc"]["lo"] > E2_MIN for r in rs)
    d_b = [r["transfer"]["own"]["anchor_minus_true100"]["point"] for r in rs]
    d_c = [r["transfer"]["ctrate_test"]["anchor_minus_true100"]["point"] for r in rs]
    e3_sign = sum(np.sign(b) == np.sign(c) for b, c in zip(d_b, d_c))
    rho = float(spearmanr(d_b, d_c).statistic)
    control = float(np.median([r["within_true100_minus_ctrate_true100"] for r in rs]))
    res = {"n_encoders": n, "positive_control_median_true100_gap": control,
           "positive_control_pass": bool(control >= CONTROL_GAP),
           "E1_ctrate_in_domain_inside_band": int(e1_ct), "E1_null_informative": bool(e1_ct >= n - 2),
           "E1_bimcv_inside_band": int(e1), "E1_pass": bool(e1_ct >= n - 2 and e1 >= n - 2),
           "E2_non_anchor_lo_above_min": int(e2), "E2_pass": bool(e2 >= n - 3),
           "E3_sign_agree": int(e3_sign), "E3_spearman": rho, "E3_pass": bool(e3_sign >= n - 5 and rho >= 0.5),
           "within_gate_pass": int(sum(r["within_gate_pass"] for r in rs)),
           "per_encoder": {r["encoder"]: {"own_anchor": t(r, "own", "anchor_ridge")["point"],
                                          "own_true100": t(r, "own", "true100_ridge")["point"],
                                          "own_random": t(r, "own", "random_ridge")["point"],
                                          "ctrate_anchor_minus_true100": r["transfer"]["ctrate_test"]["anchor_minus_true100"]["point"],
                                          "bimcv_anchor_minus_true100": r["transfer"]["own"]["anchor_minus_true100"]["point"]}
                           for r in rs}}
    json.dump(res, open(f"{out_dir}/summary.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrate")
    ap.add_argument("--bimcv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder")
    ap.add_argument("--model", default="4B")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    summary(a.out) if a.summary else run(a)
    print("EXTERNAL_EXIT rc=0", flush=True)


if __name__ == "__main__":
    main()
