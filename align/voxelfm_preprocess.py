"""Preprocessing functions copied verbatim from rmaguado/VoxelFM@fba41a3 dinov2/inference/processing.py.

The package dinov2.inference needs Python 3.12 (distributed.py subscripts multiprocessing.Queue at import time);
the cluster venv is Python 3.11, so these three pure functions are vendored instead of imported.
"""
import torch
import torch.nn.functional as F


def crop_volume(img, device, k=13):
    """
    removes empty areas around the patient
    input image must be in hounsfield units
    """

    img_device = img.device
    img = img.to(device)

    x_mask = img > -800
    vol_f = x_mask.float().unsqueeze(0).unsqueeze(0)

    kernel = torch.ones((1, 1, k, k, k), dtype=torch.float32).to(device)

    conv_sum = F.conv3d(vol_f, kernel, padding="same")

    out = conv_sum > k * k * k * 0.5

    mask = out[0, 0].bool()

    coords = mask.nonzero(as_tuple=False)

    if coords.numel() == 0:
        raise ValueError("Mask is empty — no bounding box to extract.")

    z_min, y_min, x_min = coords.min(dim=0).values
    z_max, y_max, x_max = coords.max(dim=0).values

    z_min = max(z_min.item() - k // 2, 0)
    y_min = max(y_min.item() - k // 2, 0)
    x_min = max(x_min.item() - k // 2, 0)

    z_max = min(z_max.item() + k // 2, mask.shape[0] - 1)
    y_max = min(y_max.item() + k // 2, mask.shape[1] - 1)
    x_max = min(x_max.item() + k // 2, mask.shape[2] - 1)

    bbox = (slice(z_min, z_max + 1), slice(y_min, y_max + 1), slice(x_min, x_max + 1))

    cropped_img = img[bbox]
    return cropped_img.to(img_device)


def patch_crop(img, patch_size):
    D, H, W = img.shape

    mod_D, mod_H, mod_W = D % patch_size, H % patch_size, W % patch_size

    return img[
        mod_D // 2 : D - mod_D + mod_D // 2,
        mod_H // 2 : H - mod_H + mod_H // 2,
        mod_W // 2 : W - mod_W + mod_W // 2,
    ]


def resize_max_patches(img, patch_size, max_patches, device):
    D, H, W = img.shape
    pD, pH, pW = D // patch_size, H // patch_size, W // patch_size
    num_patches = pD * pH * pW
    if num_patches <= max_patches:
        return img

    img_device = img.device
    img = img.to(device)

    scale = (max_patches / num_patches) ** (1 / 3)

    def num_patches_for(scale):
        new_D, new_H, new_W = int(D * scale), int(H * scale), int(W * scale)
        return (
            (new_D // patch_size) * (new_H // patch_size) * (new_W // patch_size),
            new_D,
            new_H,
            new_W,
        )

    best = None
    for delta in [0.98, 0.99, 1.0, 1.01, 1.02]:
        patches, new_D, new_H, new_W = num_patches_for(scale * delta)
        if patches <= max_patches:
            if best is None or patches > best[0]:
                best = (patches, new_D, new_H, new_W)

    if best is None:
        best = num_patches_for(scale * 0.95)

    _, new_D, new_H, new_W = best

    img = (
        torch.nn.functional.interpolate(
            img.unsqueeze(0).unsqueeze(0),
            size=(new_D, new_H, new_W),
            mode="trilinear",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
    )
    return img.to(img_device)
