"""RAD-ChestCT (Duke, 3,630 chest CTs, 16 abnormality labels comparable to CT-CLIP's external evaluation): case list and
image features, in the directory layout of align.ctrate (cases.jsonl, labels.json, <encoder>/<case>.npy).

    python -m align.radchest cases  --out DIR
    python -m align.radchest images --out DIR [--encoders E1,E2] --rank R --world N
    python -m align.radchest split-depth --out DIR

Volumes: RAD/files/{trn,val,tst}NNNNN.npz, key "ct", int16 HU clipped to [-1000, 1000], 0.8 mm isotropic, axis 0
superior -> inferior, axis 1 anterior -> posterior, axis 2 right -> left. align.ctrate expects LPS zyx (the index grows
towards superior, posterior, left), so only axis 0 is flipped. The preprocessing is that of CT-RATE (to_1mm,
encode_volume); it is not changed for the narrower HU range of these volumes.

Cases: the three label files (imgtrain / imgvalid / imgtest, split "train" / "valid" / "test"). A CT-RATE label is
positive when any location column of any of its RAD abnormalities is positive (the location suffix after "*" is
dropped, then the maximum is taken). The label set is the 16 of CT-CLIP's external evaluation: the 18 CT-RATE labels
without "Mosaic attenuation pattern" (no RAD counterpart) and with "Arterial wall calcification" and "Coronary artery
wall calcification" merged into one label (RAD has a single "calcification"); the merged label keeps the name
"Arterial wall calcification". Every other CT-RATE label has a RAD counterpart (LABEL_MAP).

images: default encoders are voxelfm_depth (VoxelFM CLS after blocks 2 ... 12; split-depth writes voxelfm_b<k>, with
voxelfm_b12 equal to the voxelfm feature) and every encoder of align.external.ENCODERS except voxelfm.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from align.ctrate import run_images, split_depth
from align.external import ENCODERS

RAD = os.path.join(os.environ.get("LIGHTALIGN_DATA", "data"), "rad-chestct")
SPACING_MM = (0.8, 0.8, 0.8)
SPLITS = {"imgtrain": "train", "imgvalid": "valid", "imgtest": "test"}
LABEL_MAP = {
    "Medical material": ("tracheal_tube", "chest_tube", "gi_tube", "catheter_or_port", "pacemaker_or_defib",
                         "breast_implant", "stent", "heart_valve_replacement", "staple", "clip", "suture", "hardware"),
    "Arterial wall calcification": ("calcification",),
    "Cardiomegaly": ("cardiomegaly",),
    "Pericardial effusion": ("pericardial_effusion",),
    "Hiatal hernia": ("hernia",),
    "Lymphadenopathy": ("lymphadenopathy",),
    "Emphysema": ("emphysema",),
    "Atelectasis": ("atelectasis",),
    "Lung nodule": ("nodule",),
    "Lung opacity": ("opacity",),
    "Pulmonary fibrotic sequela": ("fibrosis",),
    "Pleural effusion": ("pleural_effusion",),
    "Peribronchial thickening": ("bronchial_wall_thickening",),
    "Consolidation": ("consolidation",),
    "Bronchiectasis": ("bronchiectasis",),
    "Interlobular septal thickening": ("septal_thickening",),
}
DEFAULT_ENCODERS = ["voxelfm_depth"] + [e for e in ENCODERS if e != "voxelfm"]


class Npz(torch.utils.data.Dataset):
    """RAD-ChestCT npz -> (key, float32 HU in LPS zyx, spacing zyx), the item format of align.ctrate.Nifti."""

    def __init__(self, keys):
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        ct = np.load(f"{RAD}/files/{self.keys[i]}.npz")["ct"]
        return self.keys[i], torch.from_numpy(np.ascontiguousarray(ct[::-1], dtype=np.float32)), SPACING_MM


def build_cases(out, label_dir):
    rows = []
    for stem, split in SPLITS.items():
        df = pd.read_csv(f"{label_dir}/{stem}_Abnormality_and_Location_Labels.csv")
        abnormality = {c: c.split("*")[0] for c in df.columns if c != "NoteAcc_DEID"}
        labels = {}
        for name, rads in LABEL_MAP.items():
            cols = [c for c, a in abnormality.items() if a in rads]
            if {abnormality[c] for c in cols} != set(rads):
                raise KeyError(f"{stem}: RAD abnormalities of {name} missing from the label file: "
                               f"{set(rads) - {abnormality[c] for c in cols}}")
            labels[name] = (df[cols].max(axis=1) > 0).astype(int)
        for i, key in enumerate(df["NoteAcc_DEID"]):
            if not os.path.exists(f"{RAD}/files/{key}.npz"):
                raise FileNotFoundError(f"{key}.npz of {stem} not in {RAD}/files")
            rows.append({"id": key, "key": key, "split": split, **{n: int(v.iloc[i]) for n, v in labels.items()}})
    cases = pd.DataFrame(rows)
    if cases["key"].duplicated().any():
        raise ValueError("a volume occurs in more than one label file")
    os.makedirs(out, exist_ok=True)
    cases.to_json(f"{out}/cases.jsonl.tmp", orient="records", lines=True)
    os.replace(f"{out}/cases.jsonl.tmp", f"{out}/cases.jsonl")
    names = list(LABEL_MAP)
    prompts = [p for c in names for p in (f"IMPRESSION:\n{c} is present.", f"IMPRESSION:\nNo {c.lower()}.")]
    json.dump({"labels": names, "prompts": prompts}, open(f"{out}/labels.json.tmp", "w"), indent=1)
    os.replace(f"{out}/labels.json.tmp", f"{out}/labels.json")
    print(f"[radchest] cases={len(cases)} {cases.split.value_counts().to_dict()} "
          f"prevalence={ {c: round(float(cases[c].mean()), 3) for c in names} }", flush=True)


def check(out, names):
    """Every case of cases.jsonl has a readable finite feature in every directory of names; prints the number of
    incomplete directories (0 = done) on the last stdout line."""
    from align.extract import feature_ok

    keys = pd.read_json(f"{out}/cases.jsonl", lines=True)["key"]
    bad = []
    for n in names:
        miss = sum(not feature_ok(f"{out}/{n}/{k}.npy", 2 if n == "voxelfm_depth" else 1) for k in keys)
        print(f"[radchest] {n} missing_or_bad={miss}/{len(keys)}", flush=True)
        bad += [n] if miss else []
    print(len(bad), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["cases", "images", "split-depth", "check"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-dir", default=f"{RAD}/files", help="directory of the img{train,valid,test}_Abnormality_and_Location_Labels.csv")
    ap.add_argument("--encoders", default=",".join(DEFAULT_ENCODERS))
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    a = ap.parse_args()
    if a.step == "cases":
        build_cases(a.out, a.label_dir)
    elif a.step == "split-depth":
        split_depth(a.out)
    elif a.step == "check":
        check(a.out, a.encoders.split(","))
    else:
        run_images(a.out, a.encoders.split(","), a.rank, a.world, dataset=Npz)


if __name__ == "__main__":
    main()
