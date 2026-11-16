"""Refit the published pair counts and print the chest-unit table."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "From saved embeddings, refit n = 50, 100, 200, 400, 1600, 5994 "
            "and print mean AUROC, weighted F1, and accuracy on the chest validation volumes."
        ),
    )
    parser.add_argument("--train", required=True, help="training embedding directory")
    parser.add_argument("--valid", required=True, help="validation embedding directory (every reconstruction)")
    parser.add_argument("--out", default=None, help="optional CSV path for the full-precision rows")
    args = parser.parse_args(argv)

    import pandas as pd

    from lightalign.align import reproduction_table

    rows = reproduction_table(args.train, args.valid)
    header = f"{'n':>6} {'seeds':>5} {'volumes':>8} {'reader':<6} {'auroc':>8} {'f1_weighted':>12} {'accuracy':>9}"
    print(header, flush=True)
    for row in rows:
        print(
            f"{row['n']:6d} {row['seeds']:5d} {row['volumes']:8d} {row['reader']:<6} "
            f"{row['auroc']:8.4f} {row['f1_weighted']:12.4f} {row['accuracy']:9.4f}",
            flush=True,
        )
    if args.out:
        folder = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(folder, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
