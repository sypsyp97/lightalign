"""V2 summary: headline cell chosen on held-out patients, criteria C1-C4 on the official
validation units, and the paired bootstrap interval of headline minus VoxelFM-patch.

    python -m align.official_matrix_summary --heldout DIR --official DIR --valid DIR --out FILE.json [--ctclip CSV]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from align.align import label_names
from align.baselines import auc_matrix
from align.budget import CURVE_MODEL
from align.official import units
from align.official_matrix import DROP16, patient_resamples
from align.spiked import ENCODERS

MODELS = (CURVE_MODEL, "gemma4_31b", "ministral3_14b", "olmo31_32b", "granite42_8b", "medgemma15_4b", "qwen3vlemb_8b",
          "cxrbert")
N = 1600
CTCLIP_REF, CTCLIP_TOL = 0.7328, 0.01   # NV-Reason-CT re-run of CT-CLIP on the 3,002 chest reconstructions


def heldout_table(d):
    """Seed-mean held-out stratified AUC of TC (full map, n = 1,600) per (encoder, model)."""
    out = {}
    for e in ENCODERS:
        for m in MODELS:
            f = f"{d}/{e}_readout.csv" if m == CURVE_MODEL else f"{d}/{e}_{m}_readout.csv"
            df = pd.read_csv(f)
            sel = df[(df.reader == "TC") & (df["map"] == "full") & (df.n == N)]
            out[(e, m)] = float(sel.strat_auc.mean())
    return out


def headline(table):
    best = max(table.values())
    tied = [k for k, v in table.items() if v == best]
    for pref in ((CURVE_MODEL, "voxelfm"),):
        tied.sort(key=lambda k: (k[1] != pref[0], k[0] != pref[1]))
    return tied[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--official", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctclip", help="align.ctclip_rerun scores.csv; adds CT-CLIP AUC and headline minus CT-CLIP")
    a = ap.parse_args()
    hold = heldout_table(a.heldout)
    head = headline(hold)
    off = pd.concat([pd.read_csv(f"{a.official}/{e}_{m}_ro.csv") for e in ENCODERS for m in MODELS])
    full = off[(off.n == N)].groupby(["unit", "encoder", "model", "reader", "map"])[["strat_auc", "mean_auc"]].mean()
    res = {"headline": {"encoder": head[0], "model": head[1], "heldout_strat_auc": hold[head]},
           "heldout_tc": {f"{e}/{m}": v for (e, m), v in hold.items()}, "criteria": {}}
    e_off = pd.concat([pd.read_csv(f"{a.official}/{e}_{m}_ro_e.csv") for e in ENCODERS for m in MODELS])
    for un in full.index.get_level_values("unit").unique():
        g = full.xs(un, level="unit").strat_auc
        crit = {}
        for m in MODELS:
            tc, p = (g.xs((m, r, "full"), level=("model", "reader", "map")) for r in ("TC", "P"))
            one = g.xs((m, "TC", "rank_one"), level=("model", "reader", "map"))
            crit[m] = {"C1_tc_beats_p": int((tc > p).sum()), "C1_pass": bool((tc > p).all()),
                       "C2_median_full_minus_rank_one": float((tc - one).median()),
                       "C2_pass": bool((tc - one).median() >= 0.02)}
        for m in MODELS:
            tcm = g.xs((m, "TC", "full"), level=("model", "reader", "map"))
            rho = float(spearmanr([hold[(e, m)] for e in ENCODERS], tcm.reindex(ENCODERS)).statistic)
            crit[m]["C3_spearman_heldout_vs_official"] = rho
            crit[m]["C3_pass"] = bool(rho >= 0.7)
        res["criteria"][un] = crit
    res["C4_patient_a1"] = {}                        # E metrics exist on the a_1 unit only (known limitation)
    for m in MODELS:
        em = e_off[e_off.model == m].groupby(["encoder", "reader"])[["axis1_share", "eff_rank"]].mean()
        share_p, share_tc = (em.xs(r, level="reader").axis1_share.median() for r in ("P", "TC"))
        rank_p, rank_tc = (em.xs(r, level="reader").eff_rank.median() for r in ("P", "TC"))
        res["C4_patient_a1"][m] = {"E1_share_P": float(share_p), "E1_share_TC": float(share_tc),
                                   "E1_pass": bool(share_p >= 0.8 and share_tc <= 0.5), "E2_rank_P": float(rank_p),
                                   "E2_rank_TC": float(rank_tc), "E2_pass": bool(rank_p < rank_tc)}

    # headline minus VoxelFM-patch (TC, full map, n = 1,600), paired over the same patient resamples
    vc = pd.read_json(f"{a.valid}/cases.jsonl", lines=True)
    names = label_names(a.valid)
    lv = vc[names].to_numpy(float)
    keep16 = np.array([l not in DROP16 for l in names])
    patients = vc["id"].str.split("_").str[1].to_numpy()
    load = lambda e, m: np.load(f"{a.official}/{e}_{m}_ro_scores.npz", allow_pickle=True)   # ids: our own object array
    sh, sp = load(*head), load("voxelfm_patch", head[1])
    assert (sh["ids"] == vc["id"].to_numpy()).all() and (sp["ids"] == vc["id"].to_numpy()).all()
    # CT-CLIP released checkpoint (align.ctclip_rerun scores.csv): one score set, columns matched by finding name
    if a.ctclip:
        ct = pd.read_csv(a.ctclip, index_col="VolumeName")
        ct = ct.loc[[f"{i}.nii.gz" for i in vc["id"]], names].to_numpy(float)[None]
        chest = units(vc)["chest"]
        rep = float(np.nanmean(auc_matrix(ct[0][chest], lv[chest])))
        if abs(rep - CTCLIP_REF) > CTCLIP_TOL:   # reproduce before any paired difference is reported
            raise SystemExit(f"CT-CLIP chest mean AUC {rep:.4f} not within {CTCLIP_TOL} of {CTCLIP_REF}")
        res["ctclip"], res["headline_minus_ctclip"] = {}, {}
    res["headline_minus_voxelfm_patch"] = {}
    for un, mk in units(vc).items():
        idx = np.flatnonzero(mk)
        rs = patient_resamples(patients[idx])
        for lname, cols in (("mean_auc", np.ones(len(names), bool)), ("mean_auc16", keep16)):
            def draws(sc):
                return np.array([np.mean([np.nanmean(auc_matrix(s[idx][j], lv[idx][j])[cols]) for s in sc])
                                 for j in rs])
            dh = draws(sh["TC"])
            d = dh - draws(sp["TC"])
            res["headline_minus_voxelfm_patch"][f"{un}/{lname}"] = [float(v) for v in np.percentile(d, [2.5, 97.5])]
            if a.ctclip:
                res["ctclip"][f"{un}/{lname}"] = float(np.nanmean(auc_matrix(ct[0][idx], lv[idx])[cols]))
                d = dh - draws(ct)
                res["headline_minus_ctclip"][f"{un}/{lname}"] = [float(v) for v in np.percentile(d, [2.5, 97.5])]
    if os.path.dirname(a.out):
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "heldout_tc"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
