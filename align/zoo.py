"""Frozen 3D and video encoders for the CT side, all fed from one read of each volume.

A CT volume is read as a video: axial slices are frames (inferior to superior), and the three CT display windows
(lung, soft tissue, bone) are the colour channels, as in the GAN-CT discriminator (geometry-free-ct
gfct/models/discriminator.py). None of these encoders saw text during pretraining.

    python -m align.zoo --out DIR --encoders voxelfm_patch,ctfm,vjepa2_l,... --rank R --world N [--reverse] [--limit K]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from align.extract import VOLUMES, Volumes, feature_ok

MODELS = os.environ.get("LIGHTALIGN_MODELS", "models")                    # model weights root
DATASETS = os.environ.get("LIGHTALIGN_MODELS", "models")                  # Wan / LTX / DINOv3 weights
WINDOWS_HU = ((-1000.0, 200.0), (-160.0, 240.0), (-500.0, 1500.0))  # lung, soft tissue, bone
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def crop_body(hu):
    """Bounding box of voxels above -800 HU (removes air and most of the table)."""
    m = hu > -800
    idx = [m.any(dim=d).nonzero().squeeze(1) for d in ((1, 2), (0, 2), (0, 1))]
    if any(len(i) == 0 for i in idx):
        raise ValueError("empty body mask")
    return hu[idx[0][0]:idx[0][-1] + 1, idx[1][0]:idx[1][-1] + 1, idx[2][0]:idx[2][-1] + 1]


def as_video(hu, frames, size):
    """(D, H, W) HU, LPS zyx -> (1, 3, frames, size, size) in [0, 1]: the in-plane box is padded to a square with
    air before resizing (keeps the aspect ratio), slices are resampled to `frames`, one CT window per channel."""
    d, h, w = hu.shape
    s = max(h, w)
    hu = F.pad(hu, ((s - w) // 2, s - w - (s - w) // 2, (s - h) // 2, s - h - (s - h) // 2), value=-1000.0)
    v = F.interpolate(hu[None, None], size=(frames, size, size), mode="trilinear", align_corners=False)[0, 0]
    return torch.stack([((v - lo) / (hi - lo)).clamp(0, 1) for lo, hi in WINDOWS_HU])[None]


class VJEPA2:
    """V-JEPA 2 video encoder (self-supervised on video), 64 frames at 256 px, mean of the encoder tokens."""

    def __init__(self, path):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
        self.mean = torch.tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1, 1)

    @torch.no_grad()
    def __call__(self, hu):
        v = (as_video(hu, 64, 256) - self.mean) / self.std                    # (1, 3, T, H, W)
        out = self.model(pixel_values_videos=v.transpose(1, 2).to(torch.bfloat16), skip_predictor=True)
        return out.last_hidden_state.float().mean(1)[0]


class VideoMAE:
    """VideoMAE ViT-B (self-supervised masked video modelling on Kinetics-400, no text), 16 frames at 224 px, mean of
    the encoder tokens."""

    def __init__(self, path):
        from transformers import VideoMAEModel

        self.model = VideoMAEModel.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
        self.mean = torch.tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1, 1)

    @torch.no_grad()
    def __call__(self, hu):
        v = (as_video(hu, 16, 224) - self.mean) / self.std                    # (1, 3, T, H, W)
        out = self.model(pixel_values=v.transpose(1, 2).to(torch.bfloat16))
        return out.last_hidden_state.float().mean(1)[0]


class MedicalNet:
    """MedicalNet 3D ResNet-50 (Chen et al. 2019, supervised segmentation pretraining on 23 medical datasets, no
    text), the architecture of its models/resnet.py: conv1 stride 2, layer3 / layer4 stride 1 with dilation 2 / 4,
    shortcut B. Intensity normalisation of its data loader (z-score over voxels > 0), body crop resized to
    128 x 256 x 256, global mean of layer 4 (2048-d)."""

    def __init__(self, path):
        import torch.nn as nn

        class Bottleneck(nn.Module):
            def __init__(self, cin, planes, stride, dilation, down):
                super().__init__()
                self.conv1 = nn.Conv3d(cin, planes, 1, bias=False)
                self.bn1 = nn.BatchNorm3d(planes)
                self.conv2 = nn.Conv3d(planes, planes, 3, stride, padding=dilation, dilation=dilation, bias=False)
                self.bn2 = nn.BatchNorm3d(planes)
                self.conv3 = nn.Conv3d(planes, planes * 4, 1, bias=False)
                self.bn3 = nn.BatchNorm3d(planes * 4)
                self.downsample = down

            def forward(self, x):
                y = F.relu(self.bn1(self.conv1(x)))
                y = F.relu(self.bn2(self.conv2(y)))
                y = self.bn3(self.conv3(y))
                return F.relu(y + (self.downsample(x) if self.downsample is not None else x))

        def layer(cin, planes, blocks, stride, dilation):
            down = None
            if stride != 1 or cin != planes * 4:
                down = nn.Sequential(nn.Conv3d(cin, planes * 4, 1, stride, bias=False), nn.BatchNorm3d(planes * 4))
            mods = [Bottleneck(cin, planes, stride, dilation, down)]
            mods += [Bottleneck(planes * 4, planes, 1, dilation, None) for _ in range(blocks - 1)]
            return nn.Sequential(*mods)

        net = nn.Module()
        net.conv1 = nn.Conv3d(1, 64, 7, 2, padding=3, bias=False)
        net.bn1 = nn.BatchNorm3d(64)
        net.layer1 = layer(64, 64, 3, 1, 1)
        net.layer2 = layer(256, 128, 4, 2, 1)
        net.layer3 = layer(512, 256, 6, 1, 2)
        net.layer4 = layer(1024, 512, 3, 1, 4)
        state = torch.load(f"{path}/resnet_50_23dataset.pth", map_location="cpu", weights_only=False)["state_dict"]
        state = {k.removeprefix("module."): v for k, v in state.items()}
        missing, unexpected = net.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.endswith("num_batches_tracked")]   # absent from old checkpoints
        if missing or any(not k.startswith("conv_seg.") for k in unexpected):
            raise KeyError(f"MedicalNet weights do not match: missing {missing[:5]}, unexpected {unexpected[:5]}")
        self.net = net.cuda().eval()

    @torch.no_grad()
    def __call__(self, hu):
        n = self.net
        fg = hu[hu > 0]
        x = (hu - fg.mean()) / fg.std()
        x = F.interpolate(x[None, None], size=(128, 256, 256), mode="trilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = F.max_pool3d(F.relu(n.bn1(n.conv1(x))), 3, 2, 1)
            x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
        return x.float().mean(dim=(2, 3, 4))[0]


class WanVAE:
    """Wan2.1 video VAE encoder: normalised latent mean (16 ch) pooled to a 2 x 4 x 4 grid -> 512-d."""

    frames, size = 65, 256

    def __init__(self, wan_dir):
        from diffusers import AutoencoderKLWan

        self.vae = AutoencoderKLWan.from_pretrained(wan_dir, subfolder="vae", torch_dtype=torch.bfloat16).cuda().eval()
        cfg = self.vae.config
        self.lat_mean = torch.tensor(cfg.latents_mean, device="cuda").view(1, -1, 1, 1, 1)
        self.lat_std = torch.tensor(cfg.latents_std, device="cuda").view(1, -1, 1, 1, 1)

    @torch.no_grad()
    def latent(self, hu):
        v = as_video(hu, self.frames, self.size) * 2 - 1
        z = self.vae.encode(v.to(torch.bfloat16)).latent_dist.mode()
        return (z.float() - self.lat_mean) / self.lat_std                      # (1, 16, 17, 32, 32)

    def __call__(self, hu):
        return F.adaptive_avg_pool3d(self.latent(hu), (2, 4, 4)).flatten()


class WanDiT:
    """Wan2.1-T2V-1.3B diffusion transformer features (layer 15 of 30, flow time 0.2, empty prompt), as in the GAN-CT
    backbone probe (gfct/probe/backbones.py WanEncoder), but on the whole CT video: mean of the tokens -> 1536-d."""

    def __init__(self, wan_dir, vae, layer=15, timestep=200.0):
        from diffusers import WanTransformer3DModel

        self.vae = vae
        self.dit = WanTransformer3DModel.from_pretrained(wan_dir, subfolder="transformer",
                                                         torch_dtype=torch.bfloat16).cuda().eval()
        self.dit.blocks = self.dit.blocks[:layer]
        self.sigma = timestep / 1000.0
        context = torch.load(os.path.join(wan_dir, "empty_context.pt"), map_location="cuda", weights_only=True)
        with torch.no_grad():
            _, proj, text, _ = self.dit.condition_embedder(torch.tensor([timestep], device="cuda"),
                                                           context.to(torch.bfloat16))
        self.time_proj, self.text = proj.unflatten(1, (6, -1)), text

    @torch.no_grad()
    def __call__(self, hu):
        z = self.vae.latent(hu).to(torch.bfloat16)
        noise = torch.randn(z.shape, generator=torch.Generator(device="cuda").manual_seed(0), device="cuda",
                            dtype=z.dtype)
        z = (1 - self.sigma) * z + self.sigma * noise
        rope = self.dit.rope(z)
        h = self.dit.patch_embedding(z).flatten(2).transpose(1, 2).contiguous()
        for block in self.dit.blocks:
            h = block(h, self.text, self.time_proj, rope)
        return h.float().mean(1)[0]


def write_empty_context(wan_dir, text_length=512):
    """The umT5 output for the empty prompt that WanDiT reads, as WanPipeline builds it (real tokens, then zeros to
    max_sequence_length); same as geometry-free-ct gfct/probe/backbones.py write_empty_context."""
    from transformers import AutoTokenizer, UMT5EncoderModel

    tok = AutoTokenizer.from_pretrained(wan_dir, subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(wan_dir, subfolder="text_encoder", dtype=torch.bfloat16).eval()
    inputs = tok([""], padding="max_length", max_length=text_length, truncation=True, add_special_tokens=True,
                 return_attention_mask=True, return_tensors="pt")
    length = int(inputs.attention_mask.sum())
    with torch.no_grad():
        hidden = enc(inputs.input_ids, inputs.attention_mask).last_hidden_state[0, :length]
    context = torch.cat([hidden, hidden.new_zeros(text_length - length, hidden.shape[1])])[None]
    torch.save(context.contiguous(), os.path.join(wan_dir, "empty_context.pt"))
    print(f"[empty-context] {length} token(s), saved {tuple(context.shape)} to {wan_dir}/empty_context.pt", flush=True)


class LTXVAE:
    """LTX-2.5 video VAE encoder (loader of gfct/models/discriminator.py)."""

    frames, size = 65, 256

    def __init__(self, path):
        from safetensors import safe_open
        from safetensors.torch import load_file
        from ltx_core.model.video_vae.model_configurator import _prepare_video_encoder_kwargs
        from ltx_core.model.video_vae.video_vae import VideoEncoder

        with safe_open(path, framework="pt") as f:
            config = json.loads(f.metadata()["config"])["vae"]
        state = {}
        for k, v in load_file(path, device="cuda").items():
            if k.startswith(("vae.per_channel_statistics.", "per_channel_statistics.")):
                state["per_channel_statistics." + k.split("per_channel_statistics.", 1)[1]] = v
            elif k.startswith(("vae.encoder.", "encoder.")):
                state[k.split("encoder.", 1)[1]] = v
        with torch.device("cuda"):
            self.encoder = VideoEncoder(**_prepare_video_encoder_kwargs(config))
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise KeyError(f"LTX encoder weights do not match: missing {missing[:5]}, unexpected {unexpected[:5]}")
        self.encoder = self.encoder.to(torch.bfloat16).eval()

    @torch.no_grad()
    def __call__(self, hu):
        """Output of the last down block (the feature path the GAN-CT discriminator uses), global mean."""
        from ltx_core.model.video_vae.ops import patchify

        v = as_video(hu, self.frames, self.size) * 2 - 1
        e = self.encoder
        h = e.conv_in(patchify(v.to(torch.bfloat16), patch_size_hw=e.patch_size, patch_size_t=1))
        for block in e.down_blocks:
            h = block(h)
        return h.float().mean(dim=(2, 3, 4))[0]


class VoxelFMPatch:
    """VoxelFM (see align.extract.VoxelFM) mean of the normalised patch tokens instead of the CLS token."""

    def __init__(self, path):
        from align.extract import build_encoder

        self.enc = build_encoder("voxelfm", path)

    @torch.no_grad()
    def __call__(self, hu):
        x = self.enc.prepare(hu)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.enc.model(x)
        return out["x_norm_patchtokens"].float().mean(1)[0]


class CTFMfp32:
    def __init__(self, path):
        from align.extract import CTFM

        self.enc = CTFM(path)

    def __call__(self, hu):
        """fp32; a volume too large for fp32 activations is recomputed under bf16 autocast (logged)."""
        x = self.enc.prepare(hu)
        try:
            return self.enc(x, False)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
        print(f"[zoo] ctfm bf16 fallback shape={tuple(x.shape)}", flush=True)
        try:
            return self.enc(x, True)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
        # Whole-body scans: encode z-chunks of at most 160 slices (3 mm) and average the pooled features by size.
        chunks = x.split(160, dim=2)
        print(f"[zoo] ctfm z-chunk fallback chunks={len(chunks)}", flush=True)
        feats = torch.stack([self.enc(c.contiguous(memory_format=torch.channels_last_3d), True) for c in chunks])
        w = torch.tensor([c.shape[2] for c in chunks], device=feats.device, dtype=feats.dtype)
        return (feats * w[:, None]).sum(0) / w.sum()


def axial_slices(hu, step, size, mode):
    """Axial slices every `step` mm of the 1 mm volume, each padded to a square with air and resized to `size`."""
    d, h, w = hu.shape
    s = max(h, w)
    hu = F.pad(hu, ((s - w) // 2, s - w - (s - w) // 2, (s - h) // 2, s - h - (s - h) // 2), value=-1000.0)
    idx = torch.arange(step // 2, d, step, device=hu.device)
    return idx, F.interpolate(hu[idx][:, None], size=(size, size), mode=mode)


class DaleCT:
    """DALE-CT 2D slice encoder (LeJEPA ViT-L, chest CT-RATE, no text), preprocessing of its model card
    (HU clip [-997, 888], z-score, nearest resize); CLS of every 4 mm slice, mean over slices."""

    def __init__(self, path, patch):
        import timm
        from safetensors.torch import load_file

        # DALE-CT-0: patch 16 at 512 px; DALE-CT-1S (the Ker-VLJEPA backbone): patch 14 at 518 px.
        self.size = 512 if patch == 16 else 504
        self.model = timm.create_model("vit_large_patch14_dinov2", pretrained=False, num_classes=0, in_chans=1,
                                       patch_size=patch, img_size=512 if patch == 16 else 518, dynamic_img_size=True)
        self.model.load_state_dict(load_file(f"{path}/model.safetensors"), strict=True)
        self.model = self.model.cuda().eval()
        lo, hi, mean, std = -997.0, 888.0, -142.39, 360.97
        self.lo, self.hi, self.mean, self.std = lo, hi, (mean - lo) / (hi - lo), std / (hi - lo)

    @torch.no_grad()
    def __call__(self, hu):
        _, x = axial_slices(hu, 4, self.size, "nearest")
        x = ((x.clamp(self.lo, self.hi) - self.lo) / (self.hi - self.lo) - self.mean) / self.std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = torch.cat([self.model(c) for c in x.split(32)])
        return f.float().mean(0)


class Dinov3Slices:
    """DINOv3 ViT-L/16 (natural images, no text) on axial slices at 224 px, CLS mean over slices. 2D: the three CT
    windows of one slice as RGB. 2.5D: the soft-tissue window of slices z - 3 mm, z, z + 3 mm as RGB."""

    def __init__(self, path, slab):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
        self.slab = slab
        self.mean = torch.tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, hu):
        if self.slab:
            lo, hi = WINDOWS_HU[1]
            v = ((hu - lo) / (hi - lo)).clamp(0, 1)
            d, h, w = v.shape
            s = max(h, w)
            v = F.pad(v, ((s - w) // 2, s - w - (s - w) // 2, (s - h) // 2, s - h - (s - h) // 2), value=0.0)
            idx = torch.arange(2, d, 4, device=v.device)
            x = torch.stack([v[(idx + off).clamp(0, d - 1)] for off in (-3, 0, 3)], 1)     # (n, 3, s, s)
            x = F.interpolate(x, size=(224, 224), mode="bilinear")
        else:
            _, x = axial_slices(hu, 4, 224, "bilinear")
            x = torch.cat([((x - lo) / (hi - lo)).clamp(0, 1) for lo, hi in WINDOWS_HU], 1)
        x = ((x - self.mean) / self.std).to(torch.bfloat16)
        f = torch.cat([self.model(pixel_values=c).pooler_output for c in x.split(64)])
        return f.float().mean(0)


def build(name, cache):
    if name in ("dalect0", "dalect1s"):
        return DaleCT(f"{MODELS}/DALE-CT-{'0' if name == 'dalect0' else '1S'}", 16 if name == "dalect0" else 14)
    if name in ("dinov3_2d", "dinov3_25d"):
        return Dinov3Slices(f"{DATASETS}/dinov3-vitl16", slab=name == "dinov3_25d")
    if name == "dinov2_2d":
        return Dinov3Slices(f"{MODELS}/dinov2-large", slab=False)     # same slice protocol, DINOv2 ViT-L/14 weights
    if name == "videomae_b":
        return VideoMAE(f"{MODELS}/videomae-base")
    if name == "medicalnet50":
        return MedicalNet(f"{MODELS}/MedicalNet-Resnet50")
    if name == "voxelfm_patch":
        return VoxelFMPatch(f"{MODELS}/VoxelFM")
    if name == "ctfm":
        return CTFMfp32(f"{MODELS}/ct_fm_feature_extractor")
    if name.startswith("vjepa2_"):
        return VJEPA2(f"{MODELS}/vjepa2-vit{name.split('_')[1]}-fpc64-256")
    if name == "wan_vae":
        return cache.setdefault("wan_vae", WanVAE(f"{DATASETS}/wan21-t2v-1.3b"))
    if name == "wan_dit":
        vae = cache.setdefault("wan_vae", WanVAE(f"{DATASETS}/wan21-t2v-1.3b"))
        return WanDiT(f"{DATASETS}/wan21-t2v-1.3b", vae)
    if name == "ltx_vae":
        return LTXVAE(f"{DATASETS}/ltx25-vae/vae/ltx-2.5-video-vae-bf16.safetensors")
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoders", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: at most this many volumes per rank")
    a = ap.parse_args()
    names = a.encoders.split(",")
    cases = pd.read_json(f"{a.out}/cases.jsonl", lines=True)
    keys = cases.loc[cases.split != "none", "key"].tolist()
    keys = (keys[::-1] if a.reverse else keys)[a.rank::a.world]
    for n in names:
        os.makedirs(f"{a.out}/{n}", exist_ok=True)
    path = lambda n, k: f"{a.out}/{n}/{os.path.basename(k)}.npy"
    todo = [k for k in keys if not all(feature_ok(path(n, k)) for n in names)]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[zoo] rank={a.rank} todo={len(todo)} encoders={names}", flush=True)
    cache = {}
    encoders = {n: build(n, cache) for n in names}
    loader = torch.utils.data.DataLoader(Volumes(todo), batch_size=None, num_workers=6, prefetch_factor=2,
                                         pin_memory=True)
    spent = {n: 0.0 for n in names}
    t0 = time.perf_counter()
    for i, (key, vol) in enumerate(loader):
        hu = crop_body(vol.cuda(non_blocking=True).float())
        for n, enc in encoders.items():
            p = path(n, key)
            if feature_ok(p):
                continue
            t = time.perf_counter()
            if n in ("voxelfm_patch", "ctfm"):
                feat = enc(vol)                      # these keep the preprocessing of their own pipelines
            else:
                feat = enc(hu)
            if not torch.isfinite(feat).all():
                raise RuntimeError(f"non-finite {n} feature for {key}")
            np.save(p + ".tmp.npy", feat.cpu().numpy())
            os.replace(p + ".tmp.npy", p)
            spent[n] += time.perf_counter() - t
            if i == 0:
                print(f"[zoo] rank={a.rank} {n} dim={feat.numel()}", flush=True)
        if i % 100 == 0 or i == len(todo) - 1:
            el = time.perf_counter() - t0
            print(f"[zoo] rank={a.rank} {i + 1}/{len(todo)} {(i + 1) / el:.2f} vol/s "
                  + " ".join(f"{n}={s / el:.0%}" for n, s in spent.items()), flush=True)
    print(f"[zoo] rank={a.rank} done", flush=True)


if __name__ == "__main__":
    main()
