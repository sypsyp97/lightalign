"""Further paired domains for the two-condition test: Flickr30k (natural images), IU X-ray / Open-i
(2D chest radiographs with reports) and Clotho v2.1 (environmental audio with captions). Each becomes an embedding
directory in the CT-RATE layout: cases.jsonl (key / split / report, plus one column per label when the domain has
labels), labels.json (labels and paired zero-shot prompts; empty lists when the domain has none) and
<encoder>/<key>.npy. Splits as align.coco (hash of the item id: 1/6 sel, 1/6 test, 4/6 train). Text features:
align.extract texts / texts-st on the same directory; analyses: align.rosetta ours and align.mknn.

    python -m align.domains cases --domain flickr30k|iuxray|clotho --src DIR --out DIR
    python -m align.domains feats --domain D --src DIR --out DIR --encoder E --model-dir DIR

Sources:
    flickr30k: nlphuji/flickr30k (flickr_annotations_30k.csv, flickr30k-images/); first caption, 12,000 images.
    iuxray:    ykumards/open-i parquet (Indiana University, Open-i copy); report = FINDINGS + IMPRESSION; the study
               feature is the mean over its frontal and lateral images; labels = MeSH terms (first term before "/")
               with >= MIN_POS positives.
    clotho:    Clotho v2.1 development + evaluation audio and captions; first caption.
    chexpert:  CheXpert Plus (align.chexpert): one frontal PNG per study for 6,000 random train studies (hash splits as
               above) and every valid study (split "valid"); report = FINDINGS + IMPRESSION sections (whole report when
               both are empty), split by the hash of the patient id; labels = CheXbert labels of the IMPRESSION and
               FINDINGS sections (impression_fixed.json, findings_fixed.json), positive when either section is
               positive (1), else 0 (uncertain / negative / not mentioned); kept when >= MIN_POS positives.
               report_fixed.json is not used: its labels contradict the report text (train/patient00011/study13:
               effusion and pneumothorax positive for "no pleural effusion or pneumothorax"; text features predict
               its labels at chance, AUROC 0.49-0.54).
    chexpert_test: the official CheXpert test (500 studies) and valid (200 studies) sets of the AIMI CheXlocalize
               container (align.chexpert testset), with their radiologist labels of the five competition findings; one
               frontal image per study (the first frontal view), no report. Image features only: the evaluation set of
               the chest X-ray payoff (align.chexpert_payoff).
"""
import argparse
import glob
import json
import os
import hashlib

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from align.coco import split_of
from align.extract import feature_ok, kept_cases
from align.zoo import IMAGENET_MEAN, IMAGENET_STD

N_FLICKR = 12000
MIN_POS = 40
SR = 16000
MIN_WINDOW = SR // 10                   # 0.1 s, well above one 400-sample fbank frame
AST_SECONDS = 10.24                     # AST / EAT input: 1,024 frames of 10 ms
EAT_MEAN, EAT_STD = -4.268, 4.569       # EAT fbank normalisation (its model card)
# MeSH major terms that name a body part, not a finding ("Lung/hypoinflation" -> "Lung"): their label means "this part is
# abnormal", so their prompts contrast abnormal with normal instead of present with absent
ANATOMY = {"Lung", "Thoracic Vertebrae", "Spine", "Aorta", "Diaphragm", "Mediastinum", "Heart", "Bone and Bones",
           "Pulmonary Artery", "Ribs", "Hilum", "Pleura", "Thorax", "Costophrenic Angle", "Trachea", "Lumbar Vertebrae",
           "Shoulder", "Clavicle", "Lung, Hyperlucent", "Cardiac Shadow", "Aorta, Thoracic"}
ENCODERS = {"flickr30k": ("dinov2", "dinov2_mean", "dinov3"), "iuxray": ("dinov2", "dinov2_mean", "rad_dino"),
            "clotho": ("ast", "eat"), "chexpert": ("dinov2", "dinov3", "rad_dino"),
            "chexpert_test": ("dinov2", "dinov3", "rad_dino")}
COMPETITION = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion")


def _save_cases(rows, out, labels, prompts, tag):
    os.makedirs(out, exist_ok=True)
    df = pd.DataFrame(rows)
    if df["key"].duplicated().any():
        raise ValueError(f"[{tag}] duplicate keys")
    df.to_json(f"{out}/cases.jsonl", orient="records", lines=True)
    json.dump({"labels": labels, "prompts": prompts}, open(f"{out}/labels.json", "w"), indent=1)
    print(f"[{tag}] cases={len(df)} splits={df.split.value_counts().to_dict()} labels={len(labels)}", flush=True)
    if labels:
        print(f"[{tag}] prevalence {df[labels].mean().round(3).to_dict()}", flush=True)


def cases_flickr(src, out):
    ann = pd.read_csv(f"{src}/flickr_annotations_30k.csv")
    ids = np.random.default_rng(0).choice(len(ann), N_FLICKR, replace=False)
    rows = []
    for i in sorted(ids):
        r = ann.iloc[i]
        caps = json.loads(r["raw"])
        rows.append({"id": int(r["img_id"]), "key": "flickr_" + os.path.splitext(r["filename"])[0], "file": r["filename"],
                     "split": split_of(f"flickr_{int(r['img_id'])}"), "report": caps[0].strip()})
    _save_cases(rows, out, [], [], "flickr30k")


def cases_iuxray(src, out):
    """ykumards/open-i (copy of the Kaggle raddar Indiana University set: 3,851 studies, frontal and lateral PNG bytes,
    MeSH string "Term/qualifier;Term/qualifier" or "normal"). The Open-i tgz URLs now return an error page. Images are
    written once to src/images/<uid>_<view>.png."""
    rows = []
    os.makedirs(f"{src}/images", exist_ok=True)
    for part in sorted(glob.glob(f"{src}/data/train-*.parquet")):
        for r in pd.read_parquet(part).itertuples(index=False):
            findings = (r.findings or "").strip() if isinstance(r.findings, str) else ""
            impression = (r.impression or "").strip() if isinstance(r.impression, str) else ""
            files = []
            for view in ("frontal", "lateral"):
                data = getattr(r, f"img_{view}")
                if data is None or len(data) == 0:
                    continue
                f = f"{int(r.uid)}_{view}"
                if not os.path.exists(f"{src}/images/{f}.png"):
                    with open(f"{src}/images/{f}.png.tmp", "wb") as fh:
                        fh.write(data)
                    os.replace(f"{src}/images/{f}.png.tmp", f"{src}/images/{f}.png")
                files.append(f)
            if not (findings or impression) or not files:
                continue
            report = " ".join(t for t in (f"FINDINGS: {findings}" if findings else "",
                                          f"IMPRESSION: {impression}" if impression else "") if t)
            mesh = r.MeSH if isinstance(r.MeSH, str) else ""
            terms = {m.split("/")[0].strip() for m in mesh.split(";") if m.strip()}
            uid = str(int(r.uid))
            rows.append({"id": uid, "key": f"iu_{uid}", "files": files, "split": split_of(f"iu_{uid}"), "report": report,
                         "terms": sorted(terms)})
    counts = pd.Series([t for r in rows for t in r["terms"]]).value_counts()
    labels = sorted(t for t, c in counts.items() if c >= MIN_POS and t.lower() != "normal")
    for r in rows:
        terms = set(r.pop("terms"))
        r.update({t: float(t in terms) for t in labels})
    prompts = [p for t in labels for p in ((f"IMPRESSION: Abnormal {t.lower()}.", f"IMPRESSION: Normal {t.lower()}.")
                                           if t in ANATOMY else (f"IMPRESSION: {t} is present.", f"IMPRESSION: No {t.lower()}."))]
    _save_cases(rows, out, labels, prompts, "iuxray")


def cases_clotho(src, out):
    rows = []
    for part in ("development", "evaluation"):
        caps = pd.read_csv(f"{src}/clotho_captions_{part}.csv")
        for _, r in caps.iterrows():
            f = f"{src}/{part}/{r['file_name']}"
            if not os.path.exists(f):
                raise FileNotFoundError(f)
            # names differ only in punctuation ("Car_Keys.wav" / "Car Keys.wav") => key from a hash of the raw name
            key = f"clotho_{part[:3]}_{hashlib.sha1(r['file_name'].encode()).hexdigest()[:16]}"
            rows.append({"id": key, "key": key, "file": f"{part}/{r['file_name']}", "split": split_of(key),
                         "report": str(r["caption_1"]).strip()})
    _save_cases(rows, out, [], [], "clotho")


class Images(torch.utils.data.Dataset):
    """align.coco.Images (whole image to 224 x 224, ImageNet normalisation) for a list of files per item; grayscale
    radiographs are replicated to three channels. Returns the item index and a stack of its images."""

    def __init__(self, paths):
        self.paths = paths
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def load(self, p):
        from PIL import Image
        x = torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).copy()).permute(2, 0, 1).float()[None]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)[0]
        return (x.round().clamp(0, 255) / 255.0 - self.mean) / self.std

    def __getitem__(self, i):
        return i, torch.stack([self.load(p) for p in self.paths[i]])


class RadDinoImages(Images):
    """RAD-DINO's own image processor (microsoft/rad-dino AutoImageProcessor)."""

    def __init__(self, paths, model_dir):
        from transformers import AutoImageProcessor
        super().__init__(paths)
        self.proc = AutoImageProcessor.from_pretrained(model_dir)

    def load(self, p):
        from PIL import Image
        return self.proc(images=Image.open(p).convert("RGB"), return_tensors="pt")["pixel_values"][0]


def cases_chexpert(src, out, plan=None, ref=None):
    """plan (scale-up): align.chexpert scale-plan file; the studies of subset.json come first, in the same order, then
    those of every chunk. ref: earlier cases directory whose label list the new one must reproduce."""
    sub = json.load(open(f"{src}/subset.json"))
    if plan:
        sub["train"] = sub["train"] + [p for c in json.load(open(plan))["chunks"] for p in c]
    rep = pd.read_csv(f"{src}/reports.csv", usecols=["path_to_image", "report", "section_findings",
                                                     "section_impression"]).set_index("path_to_image")
    lab = {}
    for section in ("impression_fixed.json", "findings_fixed.json"):
        for line in open(f"{src}/{section}"):
            r = json.loads(line)
            d = lab.setdefault(r.pop("path_to_image"), {k: 0.0 for k in r})
            for k, v in r.items():
                d[k] = max(d[k], float(v == 1.0))
    rows = []
    for part, paths in (("train", sub["train"]), ("valid", sub["valid"])):
        for p in paths:
            r = rep.loc[p]
            f, i = (str(r[c]).strip() if isinstance(r[c], str) else "" for c in ("section_findings", "section_impression"))
            report = " ".join(t for t in (f"FINDINGS: {f}" if f else "", f"IMPRESSION: {i}" if i else "") if t) \
                or str(r["report"]).strip()
            key = "cx_" + p.split("/", 1)[1].rsplit("/", 1)[0].replace("/", "_")      # cx_patient00003_study1
            patient = key.split("_")[1]                 # every study of a patient in the same split
            png = p.split("/", 1)[1].rsplit(".", 1)[0].replace("/", "_") + ".png"
            rows.append({"id": p, "key": key, "files": [png], "split": split_of(patient) if part == "train" else "valid",
                         "report": report, **lab[p]})
    names = [k for k in lab[sub["train"][0]] if k != "No Finding"]
    pos = pd.DataFrame(rows)[names].sum()
    labels = [k for k in names if pos[k] >= MIN_POS]
    for r in rows:
        for k in names:
            if k not in labels:
                r.pop(k)
    prompts = [p for t in labels for p in (f"IMPRESSION: {t} is present.", f"IMPRESSION: No {t.lower()}.")]
    if ref and json.load(open(f"{ref}/labels.json"))["labels"] != labels:
        raise ValueError(f"label list differs from {ref}/labels.json: {labels}")
    _save_cases(rows, out, labels, prompts, "chexpert")


def cases_chexpert_test(src, out):
    rows = []
    for part, csv, folder in (("test", "test_labels.csv", "test/"), ("valid", "val_labels.csv", "val/")):
        lab = pd.read_csv(f"{src}/CheXpert/{csv}")
        lab["rel"] = folder + lab["Path"].str.split(r"(?:^|/)(?:test|valid)/", regex=True).str[-1]
        lab = lab[lab["rel"].str.contains("_frontal")]
        lab["study"] = lab["rel"].str.rsplit("/", n=1).str[0]
        lab = lab.sort_values("rel").drop_duplicates("study")
        for _, r in lab.iterrows():
            f = f"CheXpert/{r['rel']}"
            if not os.path.exists(f"{src}/{f}"):
                raise FileNotFoundError(f"{src}/{f}")
            rows.append({"id": r["Path"], "key": "cxo_" + r["study"].replace("/", "_"), "files": [f], "split": part,
                         "report": "", **{k: float(r[k] == 1.0) for k in COMPETITION}})
    prompts = [p for t in COMPETITION for p in (f"IMPRESSION: {t} is present.", f"IMPRESSION: No {t.lower()}.")]
    _save_cases(rows, out, list(COMPETITION), prompts, "chexpert_test")


def _image_paths(domain, src, todo):
    if domain == "chexpert_test":
        return [[f"{src}/{f}" for f in fs] for fs in todo["files"]]
    if domain == "chexpert":
        return [[f"{src}/png/{f}" for f in fs] for fs in todo["files"]]
    if domain == "flickr30k":
        return [[f"{src}/flickr30k-images/{f}"] for f in todo["file"]]
    return [[f"{src}/images/{f}.png" for f in fs] for fs in todo["files"]]


def _audio(path):
    import soundfile as sf
    from scipy.signal import resample_poly

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(1)
    if sr != SR:
        g = np.gcd(sr, SR)
        wav = resample_poly(wav, SR // g, sr // g).astype(np.float32)
    if not np.isfinite(wav).all() or len(wav) == 0:
        raise ValueError(f"bad audio {path}")
    return wav


def _windows(wav):
    """Non-overlapping AST_SECONDS windows covering the clip (the last one zero-padded by the feature extractor). A
    window shorter than one fbank frame (400 samples) cannot be transformed, so it is zero-padded to MIN_WINDOW."""
    n = int(AST_SECONDS * SR)
    return [np.pad(w, (0, max(0, MIN_WINDOW - len(w)))) for w in (wav[s:s + n] for s in range(0, max(len(wav), 1), n))]


@torch.no_grad()
def feats(domain, src, out, encoder, model_dir):
    if encoder not in ENCODERS[domain]:
        raise ValueError(f"{encoder} is not an encoder of {domain}")
    cases = pd.read_json(f"{out}/cases.jsonl", lines=True)
    cases = cases[kept_cases(cases, out)]                  # no feature for a study whose image never decodes
    path = lambda k: f"{out}/{encoder}/{k}.npy"
    os.makedirs(f"{out}/{encoder}", exist_ok=True)
    todo = cases[[not feature_ok(path(k)) for k in cases["key"]]].reset_index(drop=True)
    print(f"[{domain}] {encoder} todo={len(todo)} of {len(cases)}", flush=True)
    keys = todo["key"].tolist()

    def save(i, f):
        if not np.isfinite(f).all():
            raise RuntimeError(f"non-finite {encoder} feature for {keys[i]}")
        p = path(keys[i])
        np.save(p + ".tmp.npy", f)
        os.replace(p + ".tmp.npy", p)

    if domain == "clotho":
        from transformers import ASTFeatureExtractor, AutoModel
        fe = ASTFeatureExtractor.from_pretrained(model_dir) if encoder == "ast" else ASTFeatureExtractor(
            num_mel_bins=128, max_length=1024, do_normalize=False)
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=encoder == "eat").cuda().eval()
        for i, f in enumerate(todo["file"]):
            mel = torch.from_numpy(np.stack([fe(w, sampling_rate=SR, return_tensors="np")["input_values"][0]
                                             for w in _windows(_audio(f"{src}/{f}"))])).cuda()
            if encoder == "ast":
                z = model(input_values=mel).pooler_output
            else:                               # EAT: utterance (CLS) token of extract_features on normalised fbank
                z = model.extract_features(((mel - EAT_MEAN) / (EAT_STD * 2))[:, None])[:, 0]
            save(i, z.float().mean(0).cpu().numpy())
        print(f"[{domain}] {encoder} done", flush=True)
        return

    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    paths = _image_paths(domain, src, todo)
    ds = RadDinoImages(paths, model_dir) if encoder == "rad_dino" else Images(paths)
    loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=8)
    for i, x in loader:
        o = model(pixel_values=x.cuda().to(torch.bfloat16))
        z = o.last_hidden_state.mean(1) if encoder.endswith("_mean") else o.pooler_output
        save(i, z.float().mean(0).cpu().numpy())          # mean over the item's images (one for Flickr30k)
    print(f"[{domain}] {encoder} done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["cases", "feats"])
    ap.add_argument("--domain", choices=sorted(ENCODERS), required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder")
    ap.add_argument("--model-dir")
    ap.add_argument("--plan", help="cases, chexpert: align.chexpert scale-plan file; add the chunk studies after subset.json")
    ap.add_argument("--ref-labels", help="cases, chexpert: directory with the labels.json the new label list must equal")
    a = ap.parse_args()
    if a.step == "cases" and a.domain == "chexpert":
        cases_chexpert(a.src, a.out, a.plan, a.ref_labels)
    elif a.step == "cases":
        {"flickr30k": cases_flickr, "iuxray": cases_iuxray, "clotho": cases_clotho,
         "chexpert_test": cases_chexpert_test}[a.domain](a.src, a.out)
    else:
        feats(a.domain, a.src, a.out, a.encoder, a.model_dir)


if __name__ == "__main__":
    main()
