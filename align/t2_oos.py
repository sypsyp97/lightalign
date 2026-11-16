"""T2 out-of-sample check: pair concordance of the predictors with the target for every encoder pair that
contains a new encoder, Spearman over all encoders and over family means.

    python -m align.t2_oos --rows DIR/encoders.csv --new videomae_b,dinov2_2d,medicalnet50 --out FILE.json
"""
import argparse
import itertools
import json

import pandas as pd
from scipy.stats import spearmanr

FAMILIES = {"voxelfm": "VoxelFM", "voxelfm_patch": "VoxelFM", "ctfm": "CT-FM", "vjepa2_l": "V-JEPA2",
            "vjepa2_h": "V-JEPA2", "vjepa2_g": "V-JEPA2", "wan_vae": "Wan", "wan_dit": "Wan", "ltx_vae": "LTX",
            "dinov3_2d": "DINO", "dinov3_25d": "DINO", "dinov2_2d": "DINO", "dalect0": "DALE-CT", "dalect1s": "DALE-CT",
            "videomae_b": "VideoMAE", "medicalnet50": "MedicalNet"}
PREDICTORS = ("pred_ridge_r2", "pred_cka_measures", "ref_cka_report_test")
TARGET = "procrustes_auc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.rows).set_index("encoder")
    new = a.new.split(",")
    pairs = [(i, j) for i, j in itertools.combinations(df.index, 2) if i in new or j in new]
    fam = df.assign(family=[FAMILIES[e] for e in df.index]).groupby("family").mean(numeric_only=True)
    out = {"n_encoders": len(df), "n_pairs_with_new": len(pairs), "n_families": len(fam)}
    for p in PREDICTORS:
        conc = [(df.loc[i, p] - df.loc[j, p]) * (df.loc[i, TARGET] - df.loc[j, TARGET]) > 0 for i, j in pairs]
        out[p] = {"pair_concordance_new": float(sum(conc) / len(conc)),
                  "spearman_all": spearmanr(df[p], df[TARGET])[0],
                  "spearman_family_means": spearmanr(fam[p], fam[TARGET])[0]}
    out["pass_ridge_r2_concordance_ge_0.7"] = bool(out["pred_ridge_r2"]["pair_concordance_new"] >= 0.7)
    json.dump(out, open(a.out, "w"), indent=1)
    print(df.loc[new].to_string())
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
