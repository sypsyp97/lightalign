"""Fit the released n=5994, seed-0 weights from a training embedding directory."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fit the released CT-RATE map on all 5994 training pairs, seed 0.",
    )
    parser.add_argument("--train", required=True, help="training embedding directory (cases, image/, text.npy, prompts.npy)")
    parser.add_argument("--out", required=True, help="output .npz path; the .json is written beside it")
    args = parser.parse_args(argv)
    from lightalign.align import fit_release
    fit_release(args.train, args.out)


if __name__ == "__main__":
    main()
