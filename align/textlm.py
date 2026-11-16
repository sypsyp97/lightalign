"""Text-model family sweep summary.

Reads align.spiked cells/estimates of the sweep (middle layer, d = 64; several text models per encoder file), the
Qwen3.5 4B estimates of the main run, and the align.downstream curves of every (encoder, text model). Criteria F1-F3.

    python -m align.textlm --spiked DIR --ref-spiked DIR --downstream DIR --models DIR --out DIR
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from align.downstream import D, MODEL
from align.spiked import ENCODERS, rc

EXPECTED = ("gemma4_31b", "ministral3_14b", "olmo31_32b", "granite42_8b", "medgemma15_4b", "qwen3vlemb_8b", "cxrbert")
N_LAW = 200                   # smallest n of the downstream grid above 2d (the law needs d/n < 1/2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spiked", required=True)
    ap.add_argument("--ref-spiked", required=True)
    ap.add_argument("--downstream", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", required=True, help="download directory; <name>.denied marks a gated model without access")
    a = ap.parse_args()
    cells = pd.concat([pd.read_csv(f"{a.spiked}/{e}_cells.csv") for e in ENCODERS])
    cells = cells[(cells.layer == "mid") & (cells.d == D)]
    denied = {m for m in EXPECTED if os.path.exists(f"{a.models}/{m}.denied")}
    models = sorted(set(EXPECTED) - denied)
    if set(cells.model.unique()) != set(models):
        raise ValueError(f"cells have {sorted(cells.model.unique())}, expected {models} (denied: {sorted(denied)})")
    res = {"models": models, "dropped_access_denied": sorted(denied), "F1": {}}
    for m, g in cells.groupby("model"):
        err = np.abs(g.obs1 - g.pred1)          # the largest canonical correlation only
        sim = np.abs(g.sim1 - g.pred1)
        tol = max(0.03, 1.5 * float(np.median(sim)))
        res["F1"][m] = {"median_abs_err": float(np.median(err)), "tol": tol, "pass": bool(np.median(err) <= tol)}
    rows = []
    ref = {e: json.load(open(f"{a.ref_spiked}/{e}_estimates.json"))[f"{MODEL}/mid/{D}"]["r_hat"] for e in ENCODERS}
    for e in ENCODERS:
        est = json.load(open(f"{a.spiked}/{e}_estimates.json"))
        for m in models:
            r = np.array(est[f"{m}/mid/{D}"]["r_hat"])
            curve = pd.read_csv(f"{a.downstream}/{e}_curve_{m}.csv").groupby("n").mean_auc.mean()
            rows.append({"encoder": e, "model": m, "r1": float(r[0]) if len(r) else 0.0, "k": len(r),
                         "S200": float(r[r > rc(D / N_LAW, D / N_LAW)].sum()), "auc0": float(curve.loc[0]),
                         "auc100": float(curve.loc[100]), "auc200": float(curve.loc[200]), "auc1600": float(curve.loc[1600])})
    df = pd.DataFrame(rows)
    ref_r1 = pd.Series({e: (v[0] if v else 0.0) for e, v in ref.items()})
    rho = {m: float(spearmanr(g.set_index("encoder").r1.loc[list(ENCODERS)], ref_r1.loc[list(ENCODERS)]).statistic)
           for m, g in df.groupby("model")}
    res["F2_rho_r1_vs_qwen4B"] = rho
    res["F2_pass"] = bool(np.median(list(rho.values())) >= 0.8)
    res["F3_rho_S200_auc200"] = float(spearmanr(df.S200, df.auc200).statistic)
    res["F3_pass"] = bool(res["F3_rho_S200_auc200"] >= 0.7)
    res["F1_pass"] = bool(all(v["pass"] for v in res["F1"].values()))
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/textlm_table.csv", index=False)
    json.dump(res, open(f"{a.out}/textlm_summary.json", "w"), indent=1)
    print(df.round(3).to_string())
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
