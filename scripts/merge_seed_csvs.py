"""Merge per-seed ORIENT-HO CSVs (align.rosetta ours --seed-index i) into the cell CSV a full run writes: rows in
seed order 0, 1, 2, each seed file exactly one row with the seed of that index. A cell is merged only when all three
seed files exist; the cell CSV is written atomically and the seed files are kept.

    python scripts/merge_seed_csvs.py --out DIR/rosetta
"""
import argparse
import glob
import os

import pandas as pd

from align.rosetta import N_SEEDS
from unpaired_rosetta.randomness import SEEDS, derive_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    stems = sorted({p[:-len("__seed0.csv")] for p in glob.glob(f"{a.out}/*__seed0.csv")})
    for stem in stems:
        if os.path.exists(f"{stem}.csv"):
            continue
        parts = [f"{stem}__seed{i}.csv" for i in range(3)]
        if not all(os.path.exists(p) for p in parts):
            continue
        seeds = (derive_seeds(4242, 3) if stem.endswith("__confirm") else
                 derive_seeds(4343, 3) if stem.endswith("__confirm2") else SEEDS[:N_SEEDS])
        frames = [pd.read_csv(p, float_precision="round_trip") for p in parts]   # exact float text round trip
        for i, f in enumerate(frames):
            if len(f) != 1 or int(f["seed"].iloc[0]) != int(seeds[i]):
                raise ValueError(f"{parts[i]}: expected one row with seed {seeds[i]}, got {f['seed'].tolist()}")
        pd.concat(frames, ignore_index=True).to_csv(f"{stem}.csv.tmp", index=False)
        os.replace(f"{stem}.csv.tmp", f"{stem}.csv")
        print(f"[merge] {os.path.basename(stem)}.csv", flush=True)


if __name__ == "__main__":
    main()
