"""CT-RATE validation subset (chest CT, radiologist-written reports): case list and image features.

    python -m align.ctrate cases  --out DIR [--subset valid|train|valid-all]
    python -m align.ctrate images --out DIR --encoders voxelfm,dalect0,dinov3_2d --rank R --world N

Cases: one reconstruction per patient (scan "a", reconstruction 1 of valid_fixed), non-chest scans removed. Patients are
split 80 / 20 into "train" (unpaired halves and known pairs) and "test" (paired evaluation) by a hash of the patient
id.
--subset train: the same from train_fixed, restricted to a hashed 30 % of the patients and split three ways by a second
hash digit: "sel" (image-only alignability predictor; its reports are never read by the analysis), "train" (pairs) and
"test" (paired evaluation, untouched before the pre-registered analysis). The report is Findings + Impressions. The 18 CT-RATE abnormality labels drive the zero-shot read-out; they and
their prompts are written to labels.json, which align.align and align.extract read when present.
"""
import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from align.extract import feature_ok

CTRATE = os.path.join(os.environ.get("LIGHTALIGN_DATA", "data"), "ct-rate", "dataset")


SUBSETS = {
    "valid": ("validation_reports.csv", "valid_predicted_labels.csv", "no_chest_valid.txt", "valid_fixed"),
    "train": ("train_reports.csv", "train_predicted_labels.csv", "no_chest_train.txt", "train_fixed"),
}
TRAIN_SPLITS = ("sel", "train", "test")


def split_of(patient, subset):
    h = int(hashlib.sha256(patient.encode()).hexdigest(), 16)
    if subset == "valid":
        return "test" if h % 5 == 0 else "train"
    if h % 10 >= 3:
        return None
    return TRAIN_SPLITS[(h // 10) % 3]


def build_cases(out, subset="valid"):
    report_csv, label_csv, no_chest_txt, volume_dir = SUBSETS[subset]
    reports = pd.read_csv(f"{CTRATE}/radiology_text_reports/{report_csv}").set_index("VolumeName")
    labels = pd.read_csv(f"{CTRATE}/multi_abnormality_labels/{label_csv}").set_index("VolumeName")
    no_chest = {os.path.basename(p) for p in open(f"{CTRATE}/metadata/{no_chest_txt}").read().split()}
    files = sorted(glob.glob(f"{CTRATE}/{volume_dir}/*/*_a/*_a_1.nii.gz"))
    rows = []
    for f in files:
        name = os.path.basename(f)
        if name in no_chest:
            continue
        patient = name.split("_")[1]
        split = split_of(patient, subset)
        if split is None:
            continue
        if name not in reports.index or name not in labels.index:
            raise KeyError(f"{name} has no report or no labels")
        r = reports.loc[name]
        text = f"{str(r['Findings_EN']).strip()}\nIMPRESSION:\n{str(r['Impressions_EN']).strip()}"
        rows.append({"id": name.replace(".nii.gz", ""), "key": os.path.relpath(f, CTRATE).replace(".nii.gz", ""),
                     "split": split, "report": text, **{c: int(labels.loc[name, c]) for c in labels.columns}})
    cases = pd.DataFrame(rows)
    os.makedirs(out, exist_ok=True)
    cases.to_json(f"{out}/cases.jsonl.tmp", orient="records", lines=True)
    os.replace(f"{out}/cases.jsonl.tmp", f"{out}/cases.jsonl")
    names = list(labels.columns)
    prompts = [p for c in names for p in (f"IMPRESSION:\n{c} is present.", f"IMPRESSION:\nNo {c.lower()}.")]
    json.dump({"labels": names, "prompts": prompts}, open(f"{out}/labels.json.tmp", "w"), indent=1)
    os.replace(f"{out}/labels.json.tmp", f"{out}/labels.json")
    print(f"[ctrate] files={len(files)} no_chest_removed={len(files) - len(cases)} cases={len(cases)} {cases.split.value_counts().to_dict()} "
          f"positives(test)={ {c: int(cases.loc[cases.split == 'test', c].sum()) for c in names} }", flush=True)


def build_cases_all(out):
    """Every reconstruction of the official validation split (valid_fixed), for comparison with published CT-RATE
    numbers: all are "test"; columns no_chest (dataset list of non-chest volumes), study (patient + scan letter) and
    recon (reconstruction number) let the evaluation select the published units (all 3,039, 3,002 without no_chest,
    one per study)."""
    report_csv, label_csv, no_chest_txt, volume_dir = SUBSETS["valid"]
    reports = pd.read_csv(f"{CTRATE}/radiology_text_reports/{report_csv}").set_index("VolumeName")
    labels = pd.read_csv(f"{CTRATE}/multi_abnormality_labels/{label_csv}").set_index("VolumeName")
    no_chest = {os.path.basename(p) for p in open(f"{CTRATE}/metadata/{no_chest_txt}").read().split()}
    files = sorted(glob.glob(f"{CTRATE}/{volume_dir}/*/*/*.nii.gz"))
    rows = []
    for f in files:
        name = os.path.basename(f)
        if name not in reports.index or name not in labels.index:
            raise KeyError(f"{name} has no report or no labels")
        _, patient, scan, recon = name.replace(".nii.gz", "").split("_")
        r = reports.loc[name]
        text = f"{str(r['Findings_EN']).strip()}\nIMPRESSION:\n{str(r['Impressions_EN']).strip()}"
        rows.append({"id": name.replace(".nii.gz", ""), "key": os.path.relpath(f, CTRATE).replace(".nii.gz", ""),
                     "split": "test", "report": text, "no_chest": name in no_chest, "study": f"{patient}_{scan}",
                     "recon": int(recon), **{c: int(labels.loc[name, c]) for c in labels.columns}})
    cases = pd.DataFrame(rows)
    if len(cases) != len(labels):
        raise ValueError(f"{len(cases)} volumes on disk, {len(labels)} in the label table")
    os.makedirs(out, exist_ok=True)
    cases.to_json(f"{out}/cases.jsonl.tmp", orient="records", lines=True)
    os.replace(f"{out}/cases.jsonl.tmp", f"{out}/cases.jsonl")
    names = list(labels.columns)
    prompts = [p for c in names for p in (f"IMPRESSION:\n{c} is present.", f"IMPRESSION:\nNo {c.lower()}.")]
    json.dump({"labels": names, "prompts": prompts}, open(f"{out}/labels.json.tmp", "w"), indent=1)
    os.replace(f"{out}/labels.json.tmp", f"{out}/labels.json")
    print(f"[ctrate] valid-all volumes={len(cases)} no_chest={int(cases.no_chest.sum())} studies={cases.study.nunique()} "
          f"patients={cases.study.str.split('_').str[0].nunique()}", flush=True)


class Nifti(torch.utils.data.Dataset):
    """CT-RATE NIfTI -> (int16-range float HU array in LPS zyx, spacing zyx). Orientation from the affine:
    nibabel's closest canonical is RAS+, LPS flips the first two axes."""

    def __init__(self, keys):
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        import nibabel as nib

        img = nib.as_closest_canonical(nib.load(f"{CTRATE}/{self.keys[i]}.nii.gz"))
        a = np.asarray(img.dataobj, dtype=np.float32)[::-1, ::-1, :]           # RAS -> LPS (x, y, z)
        sx, sy, sz = (float(v) for v in img.header.get_zooms()[:3])
        return self.keys[i], torch.from_numpy(np.ascontiguousarray(a.transpose(2, 1, 0))), (sz, sy, sx)


def to_1mm(vol, spacing):
    d, h, w = vol.shape
    size = [max(1, round(n * s)) for n, s in zip((d, h, w), spacing)]
    return F.interpolate(vol[None, None], size=size, mode="trilinear", align_corners=False)[0, 0]


VOXELFM_DEPTH = (2, 4, 6, 8, 10, 11, 12)    # blocks after which the normed CLS token is read; 12 = the voxelfm feature


def voxelfm_depth(model, x):
    """VoxelFM forward (dinov2 DinoVisionTransformer.forward of the VoxelFM repo), reading the CLS token after the
    blocks in VOXELFM_DEPTH through the final norm. Returns (len(VOXELFM_DEPTH), embed_dim)."""
    t, (d, h, w) = model.prepare_tokens_with_masks(x)
    sin, cos = model.rope_embed(D=d, H=h, W=w)
    out = []
    for i, blk in enumerate(model.blocks, start=1):
        t = blk(t, sin=sin, cos=cos, num_special=1 + model.num_register_tokens)
        if i in VOXELFM_DEPTH:
            out.append(model.norm(t)[:, 0])
    return torch.stack(out, 1)[0].float()


def split_depth(out):
    """voxelfm_depth/<case>.npy (layers x dim) -> voxelfm_b<k>/<case>.npy, one encoder directory per block."""
    keys = pd.read_json(f"{out}/cases.jsonl", lines=True)["key"].map(os.path.basename)
    for k in VOXELFM_DEPTH:
        os.makedirs(f"{out}/voxelfm_b{k}", exist_ok=True)
    for key in keys:
        a = np.load(f"{out}/voxelfm_depth/{key}.npy")
        for j, k in enumerate(VOXELFM_DEPTH):
            np.save(f"{out}/voxelfm_b{k}/{key}.npy", a[j])
    print(f"[ctrate] split {len(keys)} cases into voxelfm_b{VOXELFM_DEPTH}", flush=True)


def build_encoders(names):
    from align import zoo
    from align.extract import build_encoder

    cache = {}
    return {n: build_encoder("voxelfm", f"{zoo.MODELS}/VoxelFM") if n in ("voxelfm", "voxelfm_depth")
            else zoo.build(n, cache) for n in names}


def encode_volume(encoders, hu):
    """name -> feature of one 1 mm HU volume (LPS zyx), with the preprocessing each encoder had on CT-RATE."""
    from align.zoo import crop_body

    body = crop_body(hu)
    out = {}
    for n, enc in encoders.items():
        if n == "voxelfm":
            x = enc.prepare(hu)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                feat = enc.model(x)["x_norm_clstoken"].float()[0]
        elif n == "voxelfm_depth":
            x = enc.prepare(hu)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                feat = voxelfm_depth(enc.model, x)
        elif n in ("voxelfm_patch", "ctfm"):
            feat = enc(hu)                   # these crop and resample in their own pipelines
        else:
            feat = enc(body)
        if not torch.isfinite(feat).all():
            raise RuntimeError(f"non-finite {n} feature")
        out[n] = feat
    return out


def run_images(out, names, rank, world, dataset=Nifti):
    """dataset(keys) yields (key, HU volume in LPS zyx, spacing zyx in mm); Nifti reads the CT-RATE volumes."""
    cases = pd.read_json(f"{out}/cases.jsonl", lines=True)
    keys = cases["key"].tolist()[rank::world]
    path = lambda n, k: f"{out}/{n}/{os.path.basename(k)}.npy"
    for n in names:
        os.makedirs(f"{out}/{n}", exist_ok=True)
    ok = lambda n, k: feature_ok(path(n, k), 2 if n == "voxelfm_depth" else 1)
    todo = [k for k in keys if not all(ok(n, k) for n in names)]
    print(f"[ctrate] rank={rank} todo={len(todo)} encoders={names}", flush=True)
    encoders = build_encoders(names)
    loader = torch.utils.data.DataLoader(dataset(todo), batch_size=None, num_workers=6, prefetch_factor=2)
    t0 = time.perf_counter()
    for i, (key, vol, spacing) in enumerate(loader):
        hu = to_1mm(vol.cuda(non_blocking=True), [float(s) for s in spacing])
        if i == 0:
            print(f"[ctrate] first volume {key} spacing={spacing} shape1mm={tuple(hu.shape)} "
                  f"HU p1/p50/p99={[round(float(q)) for q in torch.quantile(hu.flatten()[::97], torch.tensor([.01, .5, .99], device=hu.device))]}",
                  flush=True)
        feats = encode_volume({n: e for n, e in encoders.items() if not ok(n, key)}, hu)
        for n, feat in feats.items():
            p = path(n, key)
            np.save(p + ".tmp.npy", feat.cpu().numpy())
            os.replace(p + ".tmp.npy", p)
        if i % 50 == 0 or i == len(todo) - 1:
            print(f"[ctrate] rank={rank} {i + 1}/{len(todo)} {(i + 1) / (time.perf_counter() - t0):.2f} vol/s", flush=True)
    print(f"[ctrate] rank={rank} done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["cases", "images", "split-depth"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoders", default="voxelfm,dalect0,dinov3_2d")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--subset", choices=[*SUBSETS, "valid-all"], default="valid")
    a = ap.parse_args()
    if a.step == "cases" and a.subset == "valid-all":
        build_cases_all(a.out)
    elif a.step == "cases":
        build_cases(a.out, a.subset)
    elif a.step == "split-depth":
        split_depth(a.out)
    else:
        run_images(a.out, a.encoders.split(","), a.rank, a.world)


if __name__ == "__main__":
    main()
