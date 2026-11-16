"""VoxelFM ViT-B 3D, block-11 CLS token."""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from lightalign.pins import VOXELFM_COMMIT, VOXELFM_FILES, VOXELFM_REPO, VOXELFM_REVISION
from lightalign.voxelfm_preprocess import crop_volume, patch_crop, resize_max_patches

REFERENCE_ATTENTION = None


def sdpa_attention(self, x, sin, cos, num_special):
    """VoxelFM Attention.forward with fused scaled dot-product attention (same scale)."""
    from dinov2.layers.attention import apply_rope_to_patches

    B, N, C = x.shape
    q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k = apply_rope_to_patches(q, k, sin, cos, num_special=num_special)
    return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C))


def install_sdpa():
    """Replace dinov2 Attention.forward before any VoxelFM forward."""
    global REFERENCE_ATTENTION
    from dinov2.layers import attention

    if REFERENCE_ATTENTION is None:
        REFERENCE_ATTENTION = attention.Attention.forward
    attention.Attention.forward = sdpa_attention


def use_voxelfm_code(path):
    """Put a VoxelFM checkout on sys.path. A .git directory must be the pinned commit."""
    if not path:
        raise RuntimeError(
            "VoxelFM source is required. Clone https://github.com/rmaguado/VoxelFM, "
            f"check out {VOXELFM_COMMIT}, and pass --voxelfm-code or set VOXELFM_CODE."
        )
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise RuntimeError(f"VoxelFM source directory does not exist: {path}")
    if not os.path.isdir(os.path.join(path, "dinov2")):
        raise RuntimeError("VoxelFM checkout has no dinov2 package directory")
    git_dir = os.path.join(path, ".git")
    if os.path.isdir(git_dir):
        import subprocess
        rev = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
        if rev != VOXELFM_COMMIT:
            raise RuntimeError(f"VoxelFM checkout HEAD is {rev}, expected {VOXELFM_COMMIT}")
    else:
        print("VoxelFM checkout has no .git directory; the commit was not checked", flush=True)
    sys.path = [p for p in sys.path if os.path.abspath(p) != path]
    sys.path.insert(0, path)
    return path


def resolve_voxelfm_dir(spec):
    """Local directory with vitb_3d weights, or the pinned Hugging Face snapshot."""
    if spec is None:
        spec = VOXELFM_REPO
    if os.path.isdir(spec):
        root = os.path.abspath(spec)
    else:
        if spec != VOXELFM_REPO:
            raise ValueError(
                f"unknown VoxelFM weights {spec!r}; pass a local directory or {VOXELFM_REPO}"
            )
        from huggingface_hub import snapshot_download
        root = snapshot_download(repo_id=spec, revision=VOXELFM_REVISION, allow_patterns=list(VOXELFM_FILES))
    for rel in VOXELFM_FILES:
        if not os.path.isfile(os.path.join(root, rel)):
            raise FileNotFoundError(f"missing VoxelFM file {rel} under {root}")
    return root


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this package does not run the encoder on CPU")


class VoxelFM:
    """ViT-B 3D teacher. Preprocessing matches the VoxelFM embedding pipeline."""

    MAX_PATCHES = 25000

    def __init__(self, model_dir):
        from omegaconf import OmegaConf
        from dinov2.models import DinoVisionTransformer

        require_cuda()
        self.config = OmegaConf.load(os.path.join(model_dir, "vitb_3d", "config.yaml"))
        s = self.config.student
        # Same construction and key mapping as dinov2.inference.build_model, tensors-only load.
        self.model = DinoVisionTransformer(
            embed_dim=s.embed_dim, depth=s.depth, num_heads=s.num_heads, mlp_ratio=s.mlp_ratio,
            patch_size=s.patch_size, init_values=s.layerscale, rope_base=s.rope_base,
            rope_rescale_coords=s.rope_rescale_coords, ffn_layer=s.ffn_layer, qkv_bias=s.qkv_bias,
            proj_bias=s.proj_bias, ffn_bias=s.ffn_bias, num_register_tokens=s.num_register_tokens)
        ckpt = os.path.join(model_dir, "vitb_3d", "checkpoints", "99999.pth")
        state = torch.load(ckpt, map_location="cpu", weights_only=True)["teacher"]
        state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
        state = {k: v for k, v in state.items() if not k.startswith(("dino_head", "ibot_head"))}
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.cuda().eval()
        self.patch = self.config.student.patch_size
        self.mean = self.config.datasets[0].norm.mean
        self.std = self.config.datasets[0].norm.std

    def prepare(self, vol, max_patches=MAX_PATCHES):
        dev = torch.device("cuda")
        x = vol.to(dev, non_blocking=True).float().clamp(-1000, 1900)
        x = crop_volume(x, dev, 21)
        x = patch_crop(resize_max_patches(x, self.patch, max_patches, dev), self.patch)
        return ((x - self.mean) / self.std)[None]

    @torch.no_grad()
    def cls_token(self, x, bf16):
        """Last-layer CLS. Used only by the SDPA check, in fp32 when bf16 is false."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            return self.model(x)["x_norm_clstoken"].float()[0]

    def check_sdpa(self, vol):
        """Fused attention must match the reference attention on a reduced volume, in fp32."""
        from dinov2.layers import attention

        if REFERENCE_ATTENTION is None:
            raise RuntimeError("SDPA was not installed")
        x = self.prepare(vol, 3000)
        fused = self.cls_token(x, False)
        attention.Attention.forward = REFERENCE_ATTENTION
        try:
            ref = self.cls_token(x, False)
        finally:
            attention.Attention.forward = sdpa_attention
        cos = F.cosine_similarity(ref, fused, dim=0).item()
        print(f"sdpa vs reference attention cos={cos:.6f}", flush=True)
        if cos < 0.9999:
            raise RuntimeError(f"fused attention differs from the reference: cos={cos}")

    @torch.no_grad()
    def block11(self, x):
        """Normed CLS token after block 11. Later blocks do not affect this token.

        The forward always runs under bfloat16 autocast. Block index is 1-based.
        """
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t, (d, h, w) = self.model.prepare_tokens_with_masks(x)
            sin, cos = self.model.rope_embed(D=d, H=h, W=w)
            token = None
            for i, blk in enumerate(self.model.blocks, start=1):
                t = blk(t, sin=sin, cos=cos, num_special=1 + self.model.num_register_tokens)
                if i == 11:
                    token = self.model.norm(t)[:, 0]
                    break
            if token is None:
                raise RuntimeError("the network has fewer than 11 blocks")
            feat = token.float()[0]
        if feat.ndim != 1 or not torch.isfinite(feat).all():
            raise RuntimeError("non-finite block-11 feature")
        return feat


def load_voxelfm(code_path, weights_spec):
    use_voxelfm_code(code_path)
    require_cuda()
    install_sdpa()
    return VoxelFM(resolve_voxelfm_dir(weights_spec))


def load_nifti_lps(path):
    """Canonical RAS, then flip x and y to LPS. Array is z, y, x. Spacing is z, y, x.

    Values come from dataobj. Rescale slope and intercept are not applied.
    """
    import nibabel as nib

    img = nib.as_closest_canonical(nib.load(path))
    a = np.asarray(img.dataobj, dtype=np.float32)[::-1, ::-1, :]
    sx, sy, sz = (float(v) for v in img.header.get_zooms()[:3])
    vol = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 1, 0)))
    return vol, (sz, sy, sx)


def to_1mm(vol, spacing):
    d, h, w = vol.shape
    size = [max(1, round(n * s)) for n, s in zip((d, h, w), spacing)]
    return F.interpolate(vol[None, None], size=size, mode="trilinear", align_corners=False)[0, 0]


def embed_nifti(model, path, check):
    """1 mm resample, optional SDPA check, then the block-11 feature as a float32 vector."""
    vol, spacing = load_nifti_lps(path)
    hu = to_1mm(vol.cuda(non_blocking=True), [float(s) for s in spacing])
    if check:
        model.check_sdpa(hu)
    return model.block11(model.prepare(hu)).detach().cpu().numpy()


def feature_ok(path):
    """A cached feature is done only when it loads as a finite 1-D vector."""
    if not os.path.exists(path):
        return False
    try:
        v = np.load(path)
    except (ValueError, OSError, EOFError):
        print(f"unreadable cached feature {path}, recomputing", flush=True)
        return False
    if v.ndim != 1 or not np.isfinite(v).all():
        print(f"bad cached feature {path} shape={tuple(v.shape)}, recomputing", flush=True)
        return False
    return True


class _NiftiPaths(torch.utils.data.Dataset):
    def __init__(self, keys, paths):
        self.keys = keys
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        vol, spacing = load_nifti_lps(self.paths[i])
        return self.keys[i], vol, spacing


def embed_image_dir(dataset_root, emb_dir, code_path, weights_spec, rank, world, workers):
    """Embed every case key under emb_dir/image. rank/world shards the case list."""
    import pandas as pd

    if world < 1 or rank < 0 or rank >= world:
        raise ValueError("rank must satisfy 0 <= rank < world")
    cases_path = os.path.join(emb_dir, "cases.jsonl")
    if not os.path.isfile(cases_path):
        raise FileNotFoundError(f"missing {cases_path}; build the case list first")
    cases = pd.read_json(cases_path, lines=True)
    if "key" not in cases.columns:
        raise KeyError("cases.jsonl has no key column")
    image_dir = os.path.join(emb_dir, "image")
    os.makedirs(image_dir, exist_ok=True)
    keys = [str(k) for k in cases["key"].tolist()]
    bases = [os.path.basename(k) for k in keys]
    if len(bases) != len(set(bases)):
        raise ValueError("two cases share the same volume basename")
    shard = keys[rank::world]
    todo = []
    for key in shard:
        path = os.path.join(image_dir, os.path.basename(key) + ".npy")
        if not feature_ok(path):
            todo.append(key)
    print(f"rank={rank} world={world} todo={len(todo)} of {len(shard)}", flush=True)
    if not todo:
        return
    model = load_voxelfm(code_path, weights_spec)
    paths = [os.path.join(dataset_root, key + ".nii.gz") for key in todo]
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    loader_kw = {"batch_size": None, "num_workers": workers}
    if workers > 0:
        loader_kw["prefetch_factor"] = 2
    loader = torch.utils.data.DataLoader(_NiftiPaths(todo, paths), **loader_kw)
    checked = False
    for i, (key, vol, spacing) in enumerate(loader):
        hu = to_1mm(vol.cuda(non_blocking=True), [float(s) for s in spacing])
        if not checked:
            model.check_sdpa(hu)
            checked = True
            print(f"first volume shape_1mm={tuple(hu.shape)}", flush=True)
        feat = model.block11(model.prepare(hu))
        dest = os.path.join(image_dir, os.path.basename(key) + ".npy")
        tmp = dest + ".tmp.npy"
        np.save(tmp, feat.detach().cpu().numpy())
        os.replace(tmp, dest)
        if i % 50 == 0 or i == len(todo) - 1:
            print(f"rank={rank} {i + 1}/{len(todo)}", flush=True)
    print(f"rank={rank} done", flush=True)
