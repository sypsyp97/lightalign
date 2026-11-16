"""CT-CLIP rerun for test V2: the released CT-CLIP checkpoint, zero-shot, on every CT-RATE validation
reconstruction, scored exactly as the authors' scripts/zero_shot.py and scripts/data_inference_nii.py (CT-CLIP repo
github.com/ibrahimethemhamamci/CT-CLIP @ a2a155c): prompts "{finding} is present." / "{finding} is not present.",
score = softmax over the two prompt logits, entry 0; HU clipped to [-1000, 1000], resampled to 0.75 x 0.75 x 1.5 mm
(trilinear, float64), divided by 1000, centre crop / pad (value -1) to 480 x 480 x 240.
Two deliberate implementation differences, neither changes a score: the image is encoded once and all 36 prompts are
scored against it (the authors call the model once per finding with the same image), and the resampling runs on the
GPU in float64. valid_fixed is CT-RATE v2: the RescaleSlope / RescaleIntercept correction of the v1 release is already
applied to the stored voxels (dataset/data_correction_note.md), so the authors' metadata rescale, written for v1, is not
applied again; each volume is checked to contain air (1st percentile below -900 HU) before scoring.

    python -m align.ctclip_rerun score --cases CASES.jsonl --ctrate DATASET_DIR --repo CT-CLIP --ckpt CT-CLIP_v2.pt \
        --bert CXR-BERT-DIR --out DIR --rank R --world N
    python -m align.ctclip_rerun merge --cases CASES.jsonl --out DIR
    python -m align.ctclip_rerun embed --cases CASES.jsonl --ctrate DATASET_DIR --repo CT-CLIP --ckpt CT-CLIP_v2.pt \
        --bert CXR-BERT-DIR --out DIR --rank R --world N [--reports TRAIN_CASES.jsonl]
embed: CT-CLIP's own 512-d latents (l2-normalised, as in its forward): one image latent per validation volume
(<out>/image/<volume>.npy, same preprocessing as score) and, on rank 0 with --reports, the text latents of every
train report and of the 36 prompts (<out>/text_reports.npy, <out>/text_prompts.npy), for the report-classifier
reader on CT-CLIP's own spaces.
"""
import argparse
import glob
import importlib
import os
import sys
import types

import numpy as np
import pandas as pd

FINDINGS = ['Medical material', 'Arterial wall calcification', 'Cardiomegaly', 'Pericardial effusion',
            'Coronary artery wall calcification', 'Hiatal hernia', 'Lymphadenopathy', 'Emphysema', 'Atelectasis',
            'Lung nodule', 'Lung opacity', 'Pulmonary fibrotic sequela', 'Pleural effusion',
            'Mosaic attenuation pattern', 'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
            'Interlobular septal thickening']


def build_model(repo, ckpt, bert):
    import torch
    from transformers import BertModel, BertTokenizer

    # import the two packages without their __init__ (which pulls in training-only modules)
    for name, path in (("transformer_maskgit", f"{repo}/transformer_maskgit/transformer_maskgit"),
                       ("ct_clip", f"{repo}/CT_CLIP/ct_clip")):
        pkg = types.ModuleType(name)
        pkg.__path__ = [path]
        sys.modules[name] = pkg
    CTViT = importlib.import_module("transformer_maskgit.ctvit").CTViT
    CTCLIP = importlib.import_module("ct_clip.ct_clip").CTCLIP

    tokenizer = BertTokenizer.from_pretrained(bert, do_lower_case=True)
    text_encoder = BertModel.from_pretrained(bert)
    text_encoder.resize_token_embeddings(len(tokenizer))
    image_encoder = CTViT(dim=512, codebook_size=8192, image_size=480, patch_size=20, temporal_patch_size=10,
                          spatial_depth=4, temporal_depth=4, dim_head=32, heads=8)
    clip = CTCLIP(image_encoder=image_encoder, text_encoder=text_encoder, dim_image=294912, dim_text=768,
                  dim_latent=512, extra_latent_projection=False, use_mlm=False, downsample_image_embeds=False,
                  use_all_token_embeds=False)
    state = torch.load(ckpt, map_location="cpu")
    missing, unexpected = clip.load_state_dict(state, strict=False)
    bad = [k for k in missing + unexpected if not k.endswith("position_ids")]
    if bad:
        raise RuntimeError(f"checkpoint mismatch: {bad[:10]}")
    print(f"[ctclip] loaded {ckpt}; ignored keys {missing + unexpected}", flush=True)
    return clip.eval(), tokenizer


def load_volume(path, meta, device):
    """data_inference_nii.CTReportDatasetinfer.nii_img_to_tensor, HU from the v2 header (see module docstring)."""
    import nibabel as nib
    import torch
    import torch.nn.functional as F

    img = nib.load(path)
    data = img.get_fdata()                                         # float64; v2 voxels are already HU
    row = meta.loc[os.path.basename(path)]
    p1 = float(np.percentile(data[::4, ::4, ::4], 1))
    if p1 > -900:
        raise ValueError(f"{path}: 1st percentile {p1:.0f} HU, no air; not a v2 HU volume")
    how = "v2-hu"
    xy = float(row["XYSpacing"][1:][:-2].split(",")[0])
    z = float(row["ZSpacing"])
    data = np.clip(data, -1000, 1000).transpose(2, 0, 1)
    t = torch.tensor(data, device=device)[None, None]
    shape = [int(n * s) for n, s in zip(t.shape[2:], (z / 1.5, xy / 0.75, xy / 0.75))]
    t = F.interpolate(t, size=shape, mode="trilinear", align_corners=False)[0, 0]
    t = (t.permute(1, 2, 0) / 1000).float()
    h, w, d = t.shape
    dh, dw, dd = 480, 480, 240
    hs, ws, ds = max((h - dh) // 2, 0), max((w - dw) // 2, 0), max((d - dd) // 2, 0)
    t = t[hs:min(hs + dh, h), ws:min(ws + dw, w), ds:min(ds + dd, d)]
    ph, pw, pd = dh - t.shape[0], dw - t.shape[1], dd - t.shape[2]
    t = F.pad(t, (pd // 2, pd - pd // 2, pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=-1)
    return t.permute(2, 0, 1)[None, None], how                      # (1, 1, 240, 480, 480)


def score(a):
    import torch

    cases = pd.read_json(a.cases, lines=True)
    meta = pd.read_csv(f"{a.ctrate}/metadata/validation_metadata.csv").set_index("VolumeName")
    keys = cases["key"].tolist()[a.rank::a.world]
    os.makedirs(a.out, exist_ok=True)
    shard = f"{a.out}/scores_rank{a.rank:03d}_of{a.world:03d}.csv"
    done = set(pd.read_csv(shard).VolumeName) if os.path.exists(shard) else set()
    todo = [k for k in keys if f"{os.path.basename(k)}.nii.gz" not in done]
    print(f"[ctclip] rank {a.rank}/{a.world}: {len(keys)} volumes, {len(todo)} to do", flush=True)
    if not todo:
        return
    device = torch.device("cuda")
    clip, tokenizer = build_model(a.repo, a.ckpt, a.bert)
    clip = clip.to(device)
    prompts = [p for f in FINDINGS for p in (f"{f} is present.", f"{f} is not present.")]
    tokens = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
    temp = clip.temperature.exp()
    with torch.no_grad(), open(shard, "a") as fh:
        if not done:
            fh.write("VolumeName,rescale," + ",".join(f'"{f}"' for f in FINDINGS) + "\n")
        for k in todo:
            vol, how = load_volume(f"{a.ctrate}/{k}.nii.gz", meta, device)
            text_lat, image_lat, _ = clip(tokens, vol, device=device, return_latents=True)
            logits = (temp * (text_lat @ image_lat.T)).view(len(FINDINGS), 2)
            prob = torch.softmax(logits, dim=1)[:, 0].double().cpu().numpy()
            fh.write(f"{os.path.basename(k)}.nii.gz,{how}," + ",".join(f"{v:.8f}" for v in prob) + "\n")
            fh.flush()


def embed(a):
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda")
    clip, tokenizer = build_model(a.repo, a.ckpt, a.bert)
    clip = clip.to(device).eval()
    prompts = [p for f in FINDINGS for p in (f"{f} is present.", f"{f} is not present.")]

    def text_latents(texts, batch=32):
        out = []
        for i in range(0, len(texts), batch):
            tok = tokenizer(texts[i:i + batch], return_tensors="pt", padding="max_length", truncation=True,
                            max_length=512).to(device)
            enc = clip.text_transformer(tok.input_ids, attention_mask=tok.attention_mask)[0]
            out.append(F.normalize(clip.to_text_latent(enc[:, 0, :]), dim=-1).float().cpu().numpy())
        return np.concatenate(out)

    os.makedirs(f"{a.out}/image", exist_ok=True)
    with torch.no_grad():
        if a.rank == 0 and a.reports:
            reports = pd.read_json(a.reports, lines=True)["report"].tolist()
            np.save(f"{a.out}/text_reports.npy", text_latents(reports))
            np.save(f"{a.out}/text_prompts.npy", text_latents(prompts))
            print(f"[ctclip] text latents: {len(reports)} reports, {len(prompts)} prompts", flush=True)
        cases = pd.read_json(a.cases, lines=True)
        meta = pd.read_csv(f"{a.ctrate}/metadata/validation_metadata.csv").set_index("VolumeName")
        keys = [k for k in cases["key"].tolist()[a.rank::a.world]
                if not os.path.exists(f"{a.out}/image/{os.path.basename(k)}.npy")]
        tokens = tokenizer(prompts[:2], return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
        print(f"[ctclip] rank {a.rank}/{a.world}: {len(keys)} volumes to embed", flush=True)
        for k in keys:
            vol, _ = load_volume(f"{a.ctrate}/{k}.nii.gz", meta, device)
            _, image_lat, _ = clip(tokens, vol, device=device, return_latents=True)
            p = f"{a.out}/image/{os.path.basename(k)}.npy"
            np.save(p + ".tmp.npy", image_lat[0].float().cpu().numpy())
            os.replace(p + ".tmp.npy", p)


def merge(a):
    from sklearn.metrics import roc_auc_score

    cases = pd.read_json(a.cases, lines=True)
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{a.out}/scores_rank*.csv"))])
    df = df.drop_duplicates("VolumeName").set_index("VolumeName")
    names = [f"{i}.nii.gz" for i in cases["id"]]
    missing = [n for n in names if n not in df.index]
    if missing:
        raise SystemExit(f"{len(missing)} volumes not scored yet")
    df = df.loc[names]
    df.to_csv(f"{a.out}/scores.csv")
    lab = cases.set_index(pd.Index(names))
    keep = ~cases["no_chest"].to_numpy()
    for unit, m in (("all", np.ones(len(cases), bool)), ("chest", keep)):
        aucs = {f: roc_auc_score(lab[f].to_numpy()[m], df[f].to_numpy()[m]) for f in FINDINGS}
        print(f"[ctclip] {unit} ({int(m.sum())} volumes): mean AUC 18 = {np.mean(list(aucs.values())):.4f}; "
              f"rescale {df.rescale.value_counts().to_dict()}", flush=True)
        pd.Series(aucs).to_csv(f"{a.out}/auc_{unit}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["score", "merge", "embed"])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctrate")
    ap.add_argument("--repo")
    ap.add_argument("--ckpt")
    ap.add_argument("--bert")
    ap.add_argument("--reports", help="embed: train cases.jsonl whose reports get CT-CLIP text latents")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    a = ap.parse_args()
    {"score": score, "merge": merge, "embed": embed}[a.step](a)


if __name__ == "__main__":
    main()
