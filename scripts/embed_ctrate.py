"""Build CT-RATE case lists and write image and text embeddings."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _env(name):
    if name in os.environ:
        return os.environ[name]
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Embed CT-RATE train and validation volumes and reports.")
    sub = parser.add_subparsers(dest="step", required=True)

    cases = sub.add_parser("cases", help="write train/ and valid/ case lists")
    cases.add_argument("--ctrate", required=True, help="CT-RATE root, or the directory that contains dataset/")
    cases.add_argument("--out", required=True, help="embedding root; creates train/ and valid/")

    images = sub.add_parser("images", help="write image/<id>.npy block-11 features")
    images.add_argument("--ctrate", required=True)
    images.add_argument("--out", required=True)
    images.add_argument("--split", choices=("train", "valid", "both"), default="both")
    images.add_argument("--voxelfm-code", default=None, help="VoxelFM checkout (default: VOXELFM_CODE)")
    images.add_argument("--voxelfm", default=None, help="weight directory or rmaguado/VoxelFM")
    images.add_argument("--rank", type=int, default=0)
    images.add_argument("--world", type=int, default=1)
    images.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 loads volumes in this process)")

    texts = sub.add_parser("texts", help="write text.npy and prompts.npy")
    texts.add_argument("--out", required=True)
    texts.add_argument("--split", choices=("train", "valid", "both"), default="both")
    texts.add_argument("--text-model", default=None, help="Hugging Face id or a local snapshot")
    texts.add_argument("--text-revision", default=None, help="used when --text-model is a repo id")
    texts.add_argument("--batch", type=int, default=4)

    args = parser.parse_args(argv)

    if args.step == "cases":
        from lightalign.align import build_train_cases, build_valid_cases
        build_train_cases(args.ctrate, os.path.join(args.out, "train"))
        build_valid_cases(args.ctrate, os.path.join(args.out, "valid"))
        return

    splits = ("train", "valid") if args.split == "both" else (args.split,)

    if args.step == "images":
        from lightalign.align import dataset_root
        from lightalign.ct import embed_image_dir
        from lightalign.pins import VOXELFM_REPO
        code = args.voxelfm_code if args.voxelfm_code is not None else _env("VOXELFM_CODE")
        spec = args.voxelfm if args.voxelfm is not None else _env("VOXELFM_WEIGHTS")
        if spec is None:
            spec = VOXELFM_REPO
        root = dataset_root(args.ctrate)
        for split in splits:
            embed_image_dir(
                root, os.path.join(args.out, split), code, spec, args.rank, args.world, args.workers,
            )
        return

    if args.step == "texts":
        from lightalign.align import load_label_meta
        from lightalign.pins import QWEN_REPO, QWEN_REVISION
        from lightalign.text import embed_directory
        model = args.text_model if args.text_model is not None else QWEN_REPO
        revision = args.text_revision if args.text_revision is not None else QWEN_REVISION
        if args.batch < 1:
            raise ValueError("batch must be >= 1")
        for split in splits:
            emb = os.path.join(args.out, split)
            load_label_meta(emb)
            embed_directory(emb, model, revision, args.batch)
        return

    raise RuntimeError(args.step)


if __name__ == "__main__":
    main()
