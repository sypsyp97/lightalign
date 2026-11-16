"""Where does the image-report shared geometry break? Diagnostics on the cached embeddings (CPU, seconds).

1. Paired test CKA image-text for every image encoder and the last and the middle LLM layer of every text model;
   the same on 2,000 train cases ("_trainsel"), which is what encoder selection may look at.
2. Text-text CKA between model sizes (how stable the report geometry is across sizes).
3. CKA of image features and of text features with the structured metadata of the same cases
   (organ volumes, lesion counts and sizes, age, sex): which side carries the clinical content.
4. Supervised ceilings with all train pairs: orthogonal Procrustes and ridge regression, test FOSCTTM.

    python -m align.diagnose --emb DIR --out FILE.json --models 0.8B,2B,...
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from align.align import foscttm, linear_cka, normalize, polar
from align.extract import ATLAS

META = ["age", "liver volume (cm^3)", "total liver lesion volume (cm^3)", "number of liver lesion instances",
        "largest liver lesion diameter (cm)", "pancreas volume (cm^3)", "total pancreatic lesion volume (cm^3)",
        "number of pancreatic lesion instances", "largest pancreatic lesion diameter (cm)",
        "kidney volume (cm^3)", "total kidney lesion volume (cm^3)", "number of kidney lesion instances",
        "largest kidney lesion diameter (cm)", "spleen volume (cm^3)"]


def metadata(ids):
    df = pd.read_csv(f"{ATLAS}/AbdomenAtlas3.0MiniWithMeta.csv").set_index("BDMAP ID").loc[ids]
    m = df[META].apply(pd.to_numeric, errors="coerce")
    m["sex"] = (df["sex"] == "M").astype(float)
    m = m.fillna(m.median())
    return ((m - m.mean()) / m.std().replace(0, 1)).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--encoders", default="voxelfm", help="image feature directories under --emb")
    a = ap.parse_args()
    models = a.models.split(",")
    cases = pd.read_json(f"{a.emb}/cases.jsonl", lines=True)
    cases = cases[cases.split != "none"].reset_index(drop=True)
    tr, te = np.flatnonzero(cases.split == "train"), np.flatnonzero(cases.split == "test")
    sel = tr[:2000]          # selection split: encoder choice uses train cases only, never the test set
    if os.path.exists(f"{a.emb}/labels.json"):   # CT-RATE: the 18 abnormality labels (derived from the reports)
        names = json.load(open(f"{a.emb}/labels.json"))["labels"]
        meta = cases[names].to_numpy(float)
        meta = (meta - meta.mean(0)) / np.where(meta.std(0) > 0, meta.std(0), 1)
    else:
        meta = metadata(cases["id"].tolist())
    out = {"n_train": len(tr), "n_test": len(te)}
    texts = {}
    for m in models:
        for layer, prefix in [("last", "text"), ("mid", "textmid")]:
            texts[(m, layer)] = normalize(np.load(f"{a.emb}/{prefix}_{m}.npy").astype(float))
            out[f"cka_text_meta_{m}_{layer}"] = linear_cka(texts[(m, layer)][te], meta[te])
    for enc in a.encoders.split(","):
        z = normalize(np.stack([np.load(f"{a.emb}/{enc}/{os.path.basename(k)}.npy") for k in cases["key"]]).astype(float))
        out[f"dim_{enc}"] = z.shape[1]
        out[f"cka_image_meta_{enc}_trainsel"] = linear_cka(z[sel], meta[sel])
        out[f"cka_image_meta_{enc}"] = linear_cka(z[te], meta[te])
        for (m, layer), y in texts.items():
            out[f"cka_image_text_{enc}_{m}_{layer}_trainsel"] = linear_cka(z[sel], y[sel])
            out[f"cka_image_text_{enc}_{m}_{layer}"] = linear_cka(z[te], y[te])
        for (m, layer), y in texts.items():
            mx, my = z[tr].mean(0), y[tr].mean(0)
            xtr, ytr, xte, yte = (normalize(v - c) for v, c in [(z[tr], mx), (y[tr], my), (z[te], mx), (y[te], my)])
            w = polar(xtr.T @ ytr)
            out[f"ceiling_procrustes_all_{enc}_{m}_{layer}"] = foscttm(normalize(xte @ w) @ yte.T)
            lam = 1e-2 * np.trace(xtr.T @ xtr) / xtr.shape[1]
            r = np.linalg.solve(xtr.T @ xtr + lam * np.eye(xtr.shape[1]), xtr.T @ ytr)
            out[f"ceiling_ridge_all_{enc}_{m}_{layer}"] = foscttm(normalize(xte @ r) @ yte.T)
    for i, m in enumerate(models[1:], 1):
        out[f"cka_text_{models[i - 1]}_vs_{m}_last"] = linear_cka(texts[(models[i - 1], "last")][te], texts[(m, "last")][te])
    for k, v in out.items():
        print(f"[diag] {k} = {v:.4f}" if isinstance(v, float) else f"[diag] {k} = {v}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
