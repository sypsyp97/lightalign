"""COCO 2017 train for the natural-image mechanism test: case list in the layout of the CT embedding
directories (cases.jsonl with key / split / report and one column per label, labels.json with paired prompts), and
frozen natural-image encoder features (CLS token) of each image. Text features: align.extract texts on the same
directory; shared directions, GW and the law: align.budget and align.spiked.

    python -m align.coco cases  --src DIR --out DIR
    python -m align.coco images --src DIR --out DIR --encoder dinov2 --model-dir DIR --rank R --world N
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from align.extract import feature_ok
from align.zoo import IMAGENET_MEAN, IMAGENET_STD

N_IMAGES = 12000
SPLITS = ("sel", "test", "train", "train", "train", "train")        # 1/6 sel, 1/6 test, 4/6 train
ENCODERS = ("dinov2", "dinov3", "dinov2_mean")   # dinov2_mean: mean over all output tokens (Schnaus et al.)


def split_of(image_id):
    return SPLITS[int(hashlib.sha256(str(image_id).encode()).hexdigest(), 16) % len(SPLITS)]


def build_cases(src, out):
    caps = json.load(open(f"{src}/annotations/captions_train2017.json"))
    inst = json.load(open(f"{src}/annotations/instances_train2017.json"))
    first = {}
    for c in caps["annotations"]:                                     # first caption in annotation order
        first.setdefault(c["image_id"], c["caption"].strip())
    files = {im["id"]: im["file_name"] for im in caps["images"]}
    ids = np.random.default_rng(0).choice(sorted(first), N_IMAGES, replace=False)
    names = [c["name"] for c in sorted(inst["categories"], key=lambda c: c["id"])]
    cat = {c["id"]: c["name"] for c in inst["categories"]}
    present = {}
    for a in inst["annotations"]:
        present.setdefault(a["image_id"], set()).add(cat[a["category_id"]])
    rows = [{"id": int(i), "key": "coco_" + os.path.splitext(files[i])[0], "file": files[i], "split": split_of(int(i)),
             "report": first[i], **{n: float(n in present.get(i, ())) for n in names}} for i in ids]
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_json(f"{out}/cases.jsonl", orient="records", lines=True)
    prompts = [p for n in names for p in (f"There is a {n} in the image.", f"There is no {n} in the image.")]
    json.dump({"labels": names, "prompts": prompts}, open(f"{out}/labels.json", "w"), indent=1)
    df = pd.DataFrame(rows)
    print(f"[coco] cases={len(df)} splits={df.split.value_counts().to_dict()} "
          f"label prevalence min/median/max={df[names].mean().min():.3f}/{df[names].mean().median():.3f}/"
          f"{df[names].mean().max():.3f}", flush=True)


class Images(torch.utils.data.Dataset):
    """The authors' dinov2_transform (Schnaus et al., modalities/image_captions/encoders.py): RGB uint8, the whole image
    resized to 224 x 224 (bilinear, antialiased, kept in uint8), ImageNet normalisation. No crop: the caption describes
    the whole image."""

    def __init__(self, src, files):
        self.src, self.files = src, files
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image
        x = torch.from_numpy(np.asarray(Image.open(f"{self.src}/train2017/{self.files[i]}").convert("RGB")).copy())
        x = x.permute(2, 0, 1).float()[None]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)[0]
        return i, (x.round().clamp(0, 255) / 255.0 - self.mean) / self.std


@torch.no_grad()
def run_images(src, out, encoder, model_dir, rank, world):
    from transformers import AutoModel

    cases = pd.read_json(f"{out}/cases.jsonl", lines=True).iloc[rank::world]
    path = lambda k: f"{out}/{encoder}/{k}.npy"
    os.makedirs(f"{out}/{encoder}", exist_ok=True)
    todo = cases[[not feature_ok(path(k)) for k in cases["key"]]]
    print(f"[coco] rank={rank} {encoder} todo={len(todo)}", flush=True)
    model = AutoModel.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    loader = torch.utils.data.DataLoader(Images(src, todo["file"].tolist()), batch_size=64, num_workers=8)
    keys = todo["key"].tolist()
    for idx, x in loader:
        o = model(pixel_values=x.cuda().to(torch.bfloat16))
        feats = (o.last_hidden_state.mean(1) if encoder.endswith("_mean") else o.pooler_output).float()
        if not torch.isfinite(feats).all():
            raise RuntimeError(f"non-finite {encoder} feature")
        for i, f in zip(idx.tolist(), feats.cpu().numpy()):
            p = path(keys[i])
            np.save(p + ".tmp.npy", f)
            os.replace(p + ".tmp.npy", p)
    print(f"[coco] rank={rank} {encoder} done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["cases", "images"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", choices=ENCODERS)
    ap.add_argument("--model-dir")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    a = ap.parse_args()
    if a.step == "cases":
        build_cases(a.src, a.out)
    else:
        run_images(a.src, a.out, a.encoder, a.model_dir, a.rank, a.world)


if __name__ == "__main__":
    main()
