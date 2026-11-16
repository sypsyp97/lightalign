"""Orientation check before any RAD-ChestCT feature: one CT-RATE volume through align.ctrate.Nifti (LPS zyx, the
encoder input convention) next to RAD-ChestCT volumes as stored and with axis 0 flipped, each at the centre
axial / coronal / sagittal slice, with the anatomical direction of every image edge written on it.

    python scripts/radchest_orientation_check.py --ctrate-key <key> --rad FILE [FILE ...] --out PNG
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from align.ctrate import Nifti


def views(a):
    """(z, y, x) LPS volume: axial (y rows, x cols), coronal (z rows, x cols), sagittal (z rows, y cols)."""
    d, h, w = a.shape
    return a[d // 2], a[:, h // 2, :], a[:, :, w // 2]


# LPS zyx: index grows toward Superior (z), Posterior (y), Left (x).
EDGES = {"axial": ("A", "P", "R", "L"), "coronal": ("I", "S", "R", "L"), "sagittal": ("I", "S", "A", "P")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrate-key", required=True)
    ap.add_argument("--rad", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _, vol, spacing = Nifti([a.ctrate_key])[0]
    rows = [(f"CT-RATE {a.ctrate_key} (Nifti, LPS zyx)", vol.numpy())]
    for f in a.rad:
        ct = np.load(f)["ct"].astype(np.float32)
        rows += [(f"RAD {f.split('/')[-1]} as stored", ct), (f"RAD {f.split('/')[-1]} axis0 flipped", ct[::-1])]
    fig, ax = plt.subplots(len(rows), 3, figsize=(10, 3.3 * len(rows)))
    for r, (title, v) in enumerate(rows):
        for c, (name, img) in enumerate(zip(EDGES, views(v))):
            # origin="lower": row index grows upward, so the edge at the top is the direction the index grows toward
            ax[r, c].imshow(img, cmap="gray", vmin=-1000, vmax=1000, origin="lower", aspect="auto")
            lo_row, hi_row, lo_col, hi_col = EDGES[name]
            ax[r, c].set_title(f"{title}\n{name}", fontsize=7)
            ax[r, c].text(0.5, 0.98, hi_row, transform=ax[r, c].transAxes, ha="center", va="top", color="yellow")
            ax[r, c].text(0.5, 0.02, lo_row, transform=ax[r, c].transAxes, ha="center", va="bottom", color="yellow")
            ax[r, c].text(0.02, 0.5, lo_col, transform=ax[r, c].transAxes, ha="left", va="center", color="yellow")
            ax[r, c].text(0.98, 0.5, hi_col, transform=ax[r, c].transAxes, ha="right", va="center", color="yellow")
            ax[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f"spacing ctrate={spacing} rad shapes={[v.shape for _, v in rows[1:]]}")


if __name__ == "__main__":
    main()
