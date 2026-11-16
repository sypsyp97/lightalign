"""Equal-reader comparison. Writes score files in the format of the CT-CLIP rerun (scores.csv: VolumeName
x findings) so that align.official_ablation can bootstrap our cell against each of them (--ctclip):
  ctclip_prompt.csv  CT-CLIP read by its own prompts from its exported latents (sanity: reproduces the rerun, 0.733)
  ctclip_tc.csv      CT-CLIP read by the report classifier: logistic regression per finding (as official_ablation TC)
                     fitted on CT-CLIP's text latents of the train reports with their report labels, applied to its
                     image latents of the validation volumes
  image_probe.csv    logistic regression per finding on the frozen image features of every train image with the labels
                     of its report (no text), applied to the validation volumes

    python -m align.eqr --train DIR --valid DIR --ctclip-embed DIR --image E --out DIR
"""
import argparse
import os

import numpy as np
import pandas as pd

from align.align import label_names, normalize
from align.ctclip_rerun import FINDINGS


def classifier_scores(xt, lt, xv):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return np.stack([make_pipeline(StandardScaler(), LogisticRegression()).fit(xt, lt[:, j]).decision_function(xv)
                     for j in range(lt.shape[1])], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--ctclip-embed", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    names = label_names(a.train)
    assert names == FINDINGS, "label order differs from the CT-CLIP finding order"
    tc = pd.read_json(f"{a.train}/cases.jsonl", lines=True)
    vc = pd.read_json(f"{a.valid}/cases.jsonl", lines=True)
    lt = tc[names].to_numpy(float)
    vols = [f"{i}.nii.gz" for i in vc["id"]]
    img = np.stack([np.load(f"{a.ctclip_embed}/image/{os.path.basename(k)}.npy") for k in vc["key"]]).astype(np.float64)
    rep = np.load(f"{a.ctclip_embed}/text_reports.npy").astype(np.float64)
    prm = np.load(f"{a.ctclip_embed}/text_prompts.npy").astype(np.float64)
    if len(rep) != len(tc):
        raise ValueError(f"{len(rep)} report latents for {len(tc)} train cases")
    os.makedirs(a.out, exist_ok=True)
    save = lambda s, tag: pd.DataFrame(s, index=pd.Index(vols, name="VolumeName"), columns=names).to_csv(f"{a.out}/{tag}.csv")
    logits = img @ prm.T                                 # softmax over (present, not present) is monotone in the gap
    save(logits[:, 0::2] - logits[:, 1::2], "ctclip_prompt")
    save(classifier_scores(rep, lt, img), "ctclip_tc")
    xt = normalize(np.stack([np.load(f"{a.train}/{a.image}/{os.path.basename(k)}.npy") for k in tc["key"]]).astype(np.float64))
    xv = normalize(np.stack([np.load(f"{a.valid}/{a.image}/{os.path.basename(k)}.npy") for k in vc["key"]]).astype(np.float64))
    save(classifier_scores(xt, lt, xv), f"image_probe_{a.image}")
    print("EQR_EXIT rc=0", flush=True)


if __name__ == "__main__":
    main()
