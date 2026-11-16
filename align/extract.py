"""Embed AbdomenAtlas 3.0 CT volumes with a frozen 3D encoder (VoxelFM, or CT-FM) and their narrative reports
with the language model of Qwen3.5 (vision tower removed).

    python -m align.extract cases  --out DIR
    python -m align.extract images --out DIR --encoder voxelfm --model MODEL_DIR --rank R --world N
    python -m align.extract texts  --out DIR --model QWEN35_DIR --name NAME
"""
import argparse
import hashlib
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ATLAS = os.path.join(os.environ.get("LIGHTALIGN_DATA", "data"), "abdomenatlas3-mini")
VOLUMES = os.path.join(os.environ.get("LIGHTALIGN_DATA", "data"), "abdomenatlas3-mini", "volumes_1mm")
LESIONS = {
    "liver": "number of liver lesion instances",
    "pancreas": "number of pancreatic lesion instances",
    "kidney": "number of kidney lesion instances",
}
PROMPTS = {
    "liver": ("IMPRESSION:\nA tumor is observed in the liver.", "IMPRESSION:\nNo evidence of tumor is observed in the liver."),
    "pancreas": ("IMPRESSION:\nA tumor is observed in the pancreas.", "IMPRESSION:\nNo evidence of tumor is observed in the pancreas."),
    "kidney": ("IMPRESSION:\nA tumor is observed in the kidneys.", "IMPRESSION:\nNo evidence of tumor is observed in the kidneys."),
}


def build_cases(out):
    df = pd.read_csv(f"{ATLAS}/AbdomenAtlas3.0MiniWithMeta.csv")
    keys = {}
    for f in sorted(glob.glob(f"{VOLUMES}/index.*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r["dataset"] != "abdomenatlas3-mini":
                continue
            pid = r["patient"].split(":", 1)[1]
            if pid in keys:
                raise ValueError(f"two volumes for {pid}: {keys[pid]} and {r['key']}")
            keys[pid] = r["key"]
    report = df["narrative report"].fillna("").str.strip()
    df = df[df["BDMAP ID"].isin(keys) & report.ne("")].sort_values("BDMAP ID").reset_index(drop=True)
    train = set(pd.read_csv(f"{ATLAS}/TrainTestIDS/IID_train.csv")["BDMAP ID"])
    test = set(pd.read_csv(f"{ATLAS}/TrainTestIDS/IID_test.csv")["BDMAP ID"])
    if train & test:
        raise ValueError("IID train and test overlap")
    split = df["BDMAP ID"].map(lambda i: "train" if i in train else "test" if i in test else "none")
    cases = pd.DataFrame({"id": df["BDMAP ID"], "key": df["BDMAP ID"].map(keys), "split": split,
                          "report": df["narrative report"].str.strip()})
    for organ, col in LESIONS.items():
        cases[organ] = (df[col].fillna(0) > 0).astype(int)
    os.makedirs(out, exist_ok=True)
    cases.to_json(f"{out}/cases.jsonl", orient="records", lines=True)
    print(f"[cases] n={len(cases)}", cases["split"].value_counts().to_dict(),
          {o: int(cases.loc[cases.split == 'test', o].sum()) for o in LESIONS})


class Volumes(torch.utils.data.Dataset):
    def __init__(self, keys):
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        return self.keys[i], torch.from_numpy(np.load(f"{VOLUMES}/{self.keys[i]}.npy"))


def prepare_volume(vol):
    """CT-FM input: stored array is LPS zyx = SPL axcodes; HU [-1024, 2048] -> [0, 1]; crop air; 3 mm slices."""
    x = vol.cuda(non_blocking=True).float()
    fg = (x > -1024).nonzero()
    lo, hi = fg.min(0).values, fg.max(0).values + 1
    x = x[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    x = ((x.clamp(-1024, 2048) + 1024) / 3072)[None, None]
    z = max(1, round(x.shape[2] / 3))
    x = F.interpolate(x, size=(z, x.shape[3], x.shape[4]), mode="trilinear", align_corners=False)
    return x.contiguous(memory_format=torch.channels_last_3d)


@torch.no_grad()
def embed_volume(model, x, bf16):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        out = model(x)[-1]
    return out.float().mean(dim=(2, 3, 4))[0]


def sdpa_attention(self, x, sin, cos, num_special):
    """VoxelFM Attention.forward with the explicit softmax(QK^T)V replaced by fused SDPA (same math, same scale)."""
    from dinov2.layers.attention import apply_rope_to_patches

    B, N, C = x.shape
    q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k = apply_rope_to_patches(q, k, sin, cos, num_special=num_special)
    return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C))


class VoxelFM:
    """VoxelFM ViT-B 3D teacher, preprocessing as in its official embedding pipeline (dinov2/inference/distributed.py):
    HU clipped to [-1000, 1900], background crop k=21, at most 25,000 patches, patch crop, (x - mean) / std, CLS token.
    The stored LPS zyx array matches its DICOM loader order (slices sorted by z, rows = y, columns = x)."""

    MAX_PATCHES = 25000

    def __init__(self, model_dir):
        from omegaconf import OmegaConf
        from dinov2.models import DinoVisionTransformer

        self.config = OmegaConf.load(f"{model_dir}/vitb_3d/config.yaml")
        s = self.config.student
        # Same construction and key mapping as dinov2.inference.build_model, but with a tensors-only checkpoint load.
        self.model = DinoVisionTransformer(
            embed_dim=s.embed_dim, depth=s.depth, num_heads=s.num_heads, mlp_ratio=s.mlp_ratio,
            patch_size=s.patch_size, init_values=s.layerscale, rope_base=s.rope_base,
            rope_rescale_coords=s.rope_rescale_coords, ffn_layer=s.ffn_layer, qkv_bias=s.qkv_bias,
            proj_bias=s.proj_bias, ffn_bias=s.ffn_bias, num_register_tokens=s.num_register_tokens)
        state = torch.load(f"{model_dir}/vitb_3d/checkpoints/99999.pth", map_location="cpu", weights_only=True)["teacher"]
        state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
        state = {k: v for k, v in state.items() if not k.startswith(("dino_head", "ibot_head"))}
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.cuda().eval()
        self.patch = self.config.student.patch_size
        self.mean, self.std = self.config.datasets[0].norm.mean, self.config.datasets[0].norm.std

    def prepare(self, vol, max_patches=MAX_PATCHES):
        from align.voxelfm_preprocess import crop_volume, patch_crop, resize_max_patches

        dev = torch.device("cuda")
        x = vol.to(dev, non_blocking=True).float().clamp(-1000, 1900)
        x = crop_volume(x, dev, 21)
        x = patch_crop(resize_max_patches(x, self.patch, max_patches, dev), self.patch)
        return ((x - self.mean) / self.std)[None]

    @torch.no_grad()
    def __call__(self, x, bf16):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            return self.model(x)["x_norm_clstoken"].float()[0]

    def check_sdpa(self, vol):
        """Fused attention must reproduce the reference attention on a reduced-size volume."""
        from dinov2.layers import attention

        x = self.prepare(vol, 3000)
        fused = self(x, False)
        attention.Attention.forward = REFERENCE_ATTENTION
        ref = self(x, False)
        attention.Attention.forward = sdpa_attention
        cos = F.cosine_similarity(ref, fused, dim=0).item()
        print(f"[images] sdpa vs reference attention cos={cos:.6f}", flush=True)
        if cos < 0.9999:
            raise RuntimeError(f"fused attention differs from the reference: cos={cos}")


class CTFM:
    def __init__(self, model_dir):
        from lighter_zoo import SegResEncoder

        self.model = SegResEncoder.from_pretrained(model_dir).cuda().eval().to(memory_format=torch.channels_last_3d)

    def prepare(self, vol):
        return prepare_volume(vol)

    def __call__(self, x, bf16):
        return embed_volume(self.model, x, bf16)


REFERENCE_ATTENTION = None


def build_encoder(encoder, model_dir):
    global REFERENCE_ATTENTION
    if encoder == "ctfm":
        return CTFM(model_dir)
    from dinov2.layers import attention

    REFERENCE_ATTENTION = attention.Attention.forward
    attention.Attention.forward = sdpa_attention
    return VoxelFM(model_dir)


def kept_cases(cases, emb):
    """Boolean row mask of `cases`: False for a case whose image file is listed in EMB/excluded_pngs.txt (written by
    align.chexpert scale-images for source images that never decode). No file: every case is kept."""
    path = f"{emb}/excluded_pngs.txt"
    if not os.path.exists(path):
        return np.ones(len(cases), bool)
    out = set(open(path).read().split())
    return np.array([not out.intersection(fs) for fs in cases["files"]])


def feature_ok(path, ndim=1):
    """A cached feature counts as done only if it loads as a finite array with ndim axes (1: a vector; 2: the
    layers x dim array of voxelfm_depth). A killed job can leave a torn file."""
    if not os.path.exists(path):
        return False
    try:
        v = np.load(path)
    except (ValueError, OSError, EOFError):
        print(f"[images] unreadable cached feature {path}, recomputing", flush=True)
        return False
    if v.ndim != ndim or not np.isfinite(v).all():
        print(f"[images] bad cached feature {path} shape={v.shape}, recomputing", flush=True)
        return False
    return True


def run_images(out, encoder, model_dir, rank, world, reverse=False):
    """reverse=True walks the case list from the end, so a second job can work next to a running forward job."""
    import time

    cases = pd.read_json(f"{out}/cases.jsonl", lines=True)
    cases = cases[cases.split != "none"]
    feat_dir = f"{out}/{encoder}"
    os.makedirs(feat_dir, exist_ok=True)
    keys = cases["key"].tolist()[::-1] if reverse else cases["key"].tolist()
    todo = [k for k in keys[rank::world] if not feature_ok(f"{feat_dir}/{os.path.basename(k)}.npy")]
    print(f"[images] {encoder} rank={rank} todo={len(todo)}", flush=True)
    enc = build_encoder(encoder, model_dir)
    loader = torch.utils.data.DataLoader(Volumes(todo), batch_size=None, num_workers=6, prefetch_factor=2,
                                         pin_memory=True)
    bf16, parity = True, []
    wait = busy = nbytes = 0.0
    t = t0 = time.perf_counter()
    for n, (key, vol) in enumerate(loader):
        got = time.perf_counter()
        wait += got - t
        nbytes += vol.numel() * vol.element_size()
        if n == 0 and encoder == "voxelfm":
            enc.check_sdpa(vol)
        x = enc.prepare(vol)
        if len(parity) < 3:
            ref = enc(x, False)
            parity.append(F.cosine_similarity(ref, enc(x, True), dim=0).item())
            feat = ref
            if len(parity) == 3:
                bf16 = min(parity) >= 0.999
                print(f"[images] rank={rank} bf16 parity cos={parity} -> bf16={bf16}", flush=True)
        else:
            feat = enc(x, bf16)
        path = f"{feat_dir}/{os.path.basename(key)}.npy"
        np.save(path + ".tmp.npy", feat.cpu().numpy())
        os.replace(path + ".tmp.npy", path)
        t = time.perf_counter()
        busy += t - got
        if n % 200 == 0 or n == len(todo) - 1:
            el = t - t0
            print(f"[images] rank={rank} {n + 1}/{len(todo)} {(n + 1) / el:.2f} vol/s {nbytes / el / 1e9:.2f} GB/s "
                  f"wait={wait / el:.0%} compute={busy / el:.0%}", flush=True)
    print(f"[images] rank={rank} done", flush=True)


@torch.no_grad()
def embed_texts(lm, tok, texts, batch=64, pool="mean", max_length=1024):
    """pool="mean": mean over tokens of the last and the middle hidden layer. pool="last": hidden state of the last
    token (for the prompt-based variant, PromptEOL style). Batches in length order to cut padding (right padding)."""
    order = np.argsort([len(s) for s in texts])
    hidden = lm.config.get_text_config().hidden_size
    last = torch.empty(len(texts), hidden)
    mid = torch.empty(len(texts), hidden)
    for s in range(0, len(texts), batch):
        idx = torch.from_numpy(order[s:s + batch])
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").to("cuda")
        hs = lm(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True).hidden_states
        m = enc.attention_mask[..., None].float()
        if pool == "mean":
            last[idx] = ((hs[-1].float() * m).sum(1) / m.sum(1)).cpu()
            mid[idx] = ((hs[len(hs) // 2].float() * m).sum(1) / m.sum(1)).cpu()
        else:
            end = enc.attention_mask.sum(1) - 1
            rows = torch.arange(len(idx), device=end.device)
            last[idx] = hs[-1][rows, end].float().cpu()
            mid[idx] = hs[len(hs) // 2][rows, end].float().cpu()
    return last.numpy(), mid.numpy()


@torch.no_grad()
def embed_layers(lm, tok, texts, layers, batch=64):
    """Mean over tokens of the hidden states at the given indices (0 = input embeddings, len - 1 = last layer),
    same length-ordered batching as embed_texts. Returns {index: (n, hidden)}."""
    order = np.argsort([len(s) for s in texts])
    out = {k: torch.empty(len(texts), lm.config.hidden_size) for k in layers}
    for s in range(0, len(texts), batch):
        idx = torch.from_numpy(order[s:s + batch])
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=1024,
                  return_tensors="pt").to("cuda")
        hs = lm(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True).hidden_states
        m = enc.attention_mask[..., None].float()
        for k in layers:
            out[k][idx] = ((hs[k].float() * m).sum(1) / m.sum(1)).cpu()
    return {k: v.numpy() for k, v in out.items()}


def run_depth(out, model_dir, name, batch, every):
    """LLM depth sweep: reports and zero-shot prompts at hidden-state indices 0, every, 2 * every, ..., last, saved
    as textL<k>_<name>.npy / promptsL<k>_<name>.npy. Index len // 2 equals textmid, the last index equals text."""
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    cases = pd.read_json(f"{out}/cases.jsonl", lines=True)
    cases = cases[cases.split != "none"]
    tok = AutoTokenizer.from_pretrained(model_dir, padding_side="right")
    vlm = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16)
    lm = vlm.model.language_model
    del vlm.model.visual
    lm = lm.cuda().eval()
    depth = lm.config.num_hidden_layers
    layers = sorted(set(range(0, depth + 1, every)) | {depth // 2, depth})
    print(f"[depth] {name}: {depth} layers, saving hidden-state indices {layers}", flush=True)
    for k, v in embed_layers(lm, tok, cases["report"].tolist(), layers, batch).items():
        np.save(f"{out}/textL{k}_{name}.npy", v)
    prompts = json.load(open(f"{out}/labels.json"))["prompts"]
    for k, v in embed_layers(lm, tok, prompts, layers, batch).items():
        np.save(f"{out}/promptsL{k}_{name}.npy", v)
    print(f"[depth] {name} n={len(cases)} done", flush=True)


NUMBER = re.compile(r"\s*(\+/-|±)?\s*-?\d+(\.\d+)?\s*(cc|cm|mm|HU)?")


def text_variant(text, variant):
    """full: the report as is. impression: only the text after "IMPRESSION:" (the whole report if there is none).
    nonum: numbers and their units removed (volumes, HU values). prompt: the report wrapped for last-token pooling."""
    if variant == "full":
        return text
    if variant == "impression":
        return text.split("IMPRESSION:", 1)[1].strip() if "IMPRESSION:" in text else text
    if variant == "nonum":
        t = NUMBER.sub(" ", text)
        t = re.sub(r"\bx\b", " ", t)                   # "0.6 x 0.5 cm" leaves a bare "x"
        t = re.sub(r"\s+([.,;:)])", r"\1", t).replace("()", "")
        return re.sub(r"[ \t]{2,}", " ", t).strip()
    if variant == "prompt":
        return f'This CT report: "{text}" means in one word: "'
    raise ValueError(variant)


TEXT_CHUNK = 4096                       # new reports per saved chunk of `texts --reuse`


def run_texts(out, model_dir, name, batch, variant="full", reuse=None):
    """Qwen3.5 language model only: the vision tower is deleted after loading. Output names: <name> for the full
    report, <name>-<variant> otherwise; the zero-shot prompts go through the same variant. reuse: an earlier output
    directory; its cases (same key, same report) and its prompts keep their saved features, only new cases run."""
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    cases = pd.read_json(f"{out}/cases.jsonl", lines=True)
    cases = cases[cases.split != "none"]
    tok = AutoTokenizer.from_pretrained(model_dir, padding_side="right")
    vlm = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16)
    lm = vlm.model.language_model
    del vlm.model.visual
    lm = lm.cuda().eval()
    pool = "last" if variant == "prompt" else "mean"
    tag = name if variant == "full" else f"{name}-{variant}"
    texts = [text_variant(t, variant) for t in cases["report"].tolist()]
    print(f"[texts] {tag} example: {texts[0][:300]!r}", flush=True)
    path = f"{out}/labels.json"
    base = json.load(open(path))["prompts"] if os.path.exists(path) else [p for o in LESIONS for p in PROMPTS[o]]
    prompts = [text_variant(p, variant) for p in base]
    if reuse:
        old = pd.read_json(f"{reuse}/cases.jsonl", lines=True)
        old = old[old.split != "none"]
        if json.load(open(f"{reuse}/labels.json"))["prompts"] != base:
            raise ValueError(f"prompts of {reuse} differ from {path}")
        row = {k: i for i, k in enumerate(old["key"])}
        take = np.array([row.get(k, -1) for k in cases["key"]])
        same = old["report"].to_numpy()[take[take >= 0]] == cases["report"].to_numpy()[take >= 0]
        if not same.all():
            raise ValueError(f"{(~same).sum()} reports of {reuse} differ from {out}")
        todo = np.flatnonzero(take < 0)
        print(f"[texts] {tag}: {len(cases) - len(todo)} cases reused from {reuse}, {len(todo)} new", flush=True)
        last = np.empty((len(cases), np.load(f"{reuse}/text_{tag}.npy", mmap_mode="r").shape[1]), np.float32)
        mid = np.empty_like(last)
        for src_name, dst in (("text", last), ("textmid", mid)):
            dst[take >= 0] = np.load(f"{reuse}/{src_name}_{tag}.npy")[take[take >= 0]]
        cdir = f"{out}/texts_{tag}_chunks"                  # new reports in fixed chunks of cases order, saved as done
        os.makedirs(cdir, exist_ok=True)
        for j in range(0, len(todo), TEXT_CHUNK):
            part, f = todo[j:j + TEXT_CHUNK], f"{cdir}/chunk{j // TEXT_CHUNK:05d}.npz"
            keys = cases["key"].to_numpy(str)[part]
            sig = hashlib.sha256("\x00".join([model_dir, tag, pool, "last+mid(len//2)", *(texts[i] for i in part)])
                                 .encode()).hexdigest()     # exact input texts, model, name, pooling, layer choice
            if os.path.exists(f):
                z = np.load(f)
                if "sig" in z.files and z["keys"].tolist() == keys.tolist() and str(z["sig"]) == sig \
                        and z["last"].shape == (len(part), last.shape[1]):
                    last[part], mid[part] = z["last"], z["mid"]
                    continue
                print(f"[texts] {f} does not match the cases, texts or model, recomputing", flush=True)
            last[part], mid[part] = embed_texts(lm, tok, [texts[i] for i in part], batch, pool)
            np.savez(f + ".tmp.npz", keys=keys, sig=sig, last=last[part], mid=mid[part])
            os.replace(f + ".tmp.npz", f)
            print(f"[texts] chunk {j // TEXT_CHUNK + 1} of {-(-len(todo) // TEXT_CHUNK)} saved", flush=True)
        for p in ("prompts", "promptsmid"):
            np.save(f"{out}/{p}_{tag}.npy", np.load(f"{reuse}/{p}_{tag}.npy"))
    else:
        last, mid = embed_texts(lm, tok, texts, batch, pool)
        p_last, p_mid = embed_texts(lm, tok, prompts, batch, pool)
        np.save(f"{out}/prompts_{tag}.npy", p_last)
        np.save(f"{out}/promptsmid_{tag}.npy", p_mid)
    np.save(f"{out}/text_{tag}.npy", last)
    np.save(f"{out}/textmid_{tag}.npy", mid)
    print(f"[texts] {tag} n={len(cases)} dim={last.shape[1]}", flush=True)


def run_texts_any(out, model_dir, name, batch):
    """Any Hugging Face text model (decoder LLM, embedding model, BERT-style encoder) loaded with AutoModel: the full
    report with the mean pooling of run_texts over the middle and the last hidden layer. Same output names."""
    from transformers import AutoModel, AutoTokenizer

    cases = pd.read_json(f"{out}/cases.jsonl", lines=True)
    cases = cases[cases.split != "none"]
    tok = AutoTokenizer.from_pretrained(model_dir, padding_side="right", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModel.from_pretrained(model_dir, dtype=torch.bfloat16, trust_remote_code=True)
    if hasattr(lm, "language_model"):       # multimodal checkpoints (Gemma 4, Ministral 3, Qwen3-VL): text part only
        lm = lm.language_model
    lm = lm.cuda().eval()
    positions = getattr(lm.config.get_text_config(), "max_position_embeddings", None)
    if positions is None:
        raise ValueError(f"{model_dir}: config has no max_position_embeddings")
    max_length = min(1024, tok.model_max_length, positions)
    last, mid = embed_texts(lm, tok, cases["report"].tolist(), batch, "mean", max_length)
    np.save(f"{out}/text_{name}.npy", last)
    np.save(f"{out}/textmid_{name}.npy", mid)
    prompts = json.load(open(f"{out}/labels.json"))["prompts"]
    p_last, p_mid = embed_texts(lm, tok, prompts, batch, "mean", max_length)
    np.save(f"{out}/prompts_{name}.npy", p_last)
    np.save(f"{out}/promptsmid_{name}.npy", p_mid)
    print(f"[texts-any] {name} n={len(cases)} dim={last.shape[1]} max_length={max_length}", flush=True)


@torch.no_grad()
def run_texts_st(out, model_dir, name, batch):
    """Sentence-transformer encoder (e.g. all-mpnet-base-v2, the text model of Schnaus et al.): mean over tokens of the
    last hidden layer weighted by the attention mask, then unit norm (sentence-transformers pooling), truncated at the
    model's sentence-transformers max_seq_length (384 for all-mpnet-base-v2, not the tokenizer's 512).
    Saves text_<name>.npy and prompts_<name>.npy."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel, AutoTokenizer

    cases = pd.read_json(f"{out}/cases.jsonl", lines=True)
    cases = cases[cases.split != "none"]
    tok = AutoTokenizer.from_pretrained(model_dir)
    lm = AutoModel.from_pretrained(model_dir).cuda().eval()
    cfg = (f"{model_dir}/sentence_bert_config.json" if os.path.isdir(model_dir)
           else hf_hub_download(model_dir, "sentence_bert_config.json"))
    max_length = json.load(open(cfg))["max_seq_length"]

    def encode(texts):
        if not texts:                               # label-free domains have no prompts
            return np.zeros((0, lm.config.hidden_size), np.float32)
        res = []
        for s in range(0, len(texts), batch):
            enc = tok(texts[s:s + batch], padding=True, truncation=True, max_length=max_length,
                      return_tensors="pt").to("cuda")
            h = lm(**enc).last_hidden_state
            m = enc.attention_mask[..., None].float()
            res.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).cpu())
        return torch.cat(res).numpy()

    np.save(f"{out}/text_{name}.npy", encode(cases["report"].tolist()))
    np.save(f"{out}/prompts_{name}.npy", encode(json.load(open(f"{out}/labels.json"))["prompts"]))
    print(f"[texts-st] {name} n={len(cases)} max_length={max_length}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["cases", "images", "texts", "depth", "texts-any", "texts-st"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model")
    ap.add_argument("--encoder", choices=["ctfm", "voxelfm"], default="voxelfm")
    ap.add_argument("--name")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--batch", type=int, default=64, help="report batch size for texts")
    ap.add_argument("--variant", choices=["full", "impression", "nonum", "prompt"], default="full")
    ap.add_argument("--every", type=int, default=4, help="depth step: hidden-state index stride")
    ap.add_argument("--reuse", help="texts: earlier output directory whose saved features of the same cases are kept")
    a = ap.parse_args()
    if a.step == "cases":
        build_cases(a.out)
    elif a.step == "images":
        run_images(a.out, a.encoder, a.model, a.rank, a.world, a.reverse)
    elif a.step == "depth":
        run_depth(a.out, a.model, a.name, a.batch, a.every)
    elif a.step == "texts-st":
        run_texts_st(a.out, a.model, a.name, a.batch)
    elif a.step == "texts-any":
        run_texts_any(a.out, a.model, a.name, a.batch)
    else:
        run_texts(a.out, a.model, a.name, a.batch, a.variant, a.reuse)


if __name__ == "__main__":
    main()
