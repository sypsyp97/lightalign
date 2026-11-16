"""Run the upstream entry point after checking its actual data selections."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__, allow_abbrev=False)
    parser.add_argument("--protocol-source", required=True)
    parser.add_argument("--protocol-audit", required=True)
    options, remaining = parser.parse_known_args()
    source = Path(options.protocol_source)
    sys.path.insert(0, str(source / "src"))
    import data

    original = data.get_supervised_data

    def checked(args):
        result = original(args)
        train = result["train"]
        expected = {"num_samples_paired": args.n_supervised_pairs,
                    "num_samples_unpaired_image": args.n_unsupervised_image,
                    "num_samples_unpaired_text": args.n_unsupervised_text}
        for name, count in expected.items():
            if train.data_info[name] != count:
                raise ValueError(f"Upstream selection changed {name}: {train.data_info[name]} != {count}")
        paths = args.supervised_image_embedding
        if isinstance(paths, str):
            paths = [paths]
        if len(paths) != 1:
            raise ValueError(f"Expected one paired bank, got {paths}")
        with h5py.File(paths[0]) as f:
            ids = f["ids"][:].astype(str)
        indices = np.asarray(train.bimodal_loader.dataset.indices)
        expected_indices = np.random.RandomState(args.seed).permutation(len(ids))[:args.n_supervised_pairs]
        if not np.array_equal(indices, expected_indices):
            raise ValueError("Upstream pair draw differs from the common draw")
        audit = {"seed": args.seed, "n": args.n_supervised_pairs, "paired_bank_count": len(ids),
                 **expected, "pair_draw_sha256": hashlib.sha256(ids[indices].astype("S").tobytes()).hexdigest()}
        path = Path(options.protocol_audit)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit, indent=2) + "\n")
        print("[protocol-data] " + json.dumps(audit), flush=True)
        return result

    data.get_supervised_data = checked
    entry = source / "tools/train.py"
    sys.argv = [str(entry), *remaining]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
