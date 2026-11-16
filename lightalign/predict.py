"""Score one chest CT NIfTI with the released linear map."""

import argparse
import os


def _env(name):
    if name in os.environ:
        return os.environ[name]
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="lightalign-predict",
        description=(
            "Score 18 CT-RATE findings on one chest CT NIfTI. "
            "TC is the report-classifier decision function. "
            "P is the positive-prompt cosine minus the negative-prompt cosine."
        ),
    )
    parser.add_argument("--ct", required=True, help="NIfTI volume (.nii or .nii.gz)")
    parser.add_argument(
        "--weights",
        required=True,
        help="path to the .npz, a directory with the release files, or a Hugging Face repo id",
    )
    parser.add_argument(
        "--voxelfm-code",
        default=None,
        help="VoxelFM checkout (default: VOXELFM_CODE)",
    )
    parser.add_argument(
        "--voxelfm",
        default=None,
        help="local VoxelFM weight directory or rmaguado/VoxelFM (default: VOXELFM_WEIGHTS or that repo)",
    )
    args = parser.parse_args(argv)
    if not os.path.isfile(args.ct):
        raise FileNotFoundError(args.ct)

    from lightalign.align import FINDINGS, load_release, score_feature
    from lightalign.ct import embed_nifti, load_voxelfm
    from lightalign.pins import VOXELFM_REPO

    code = args.voxelfm_code
    if code is None:
        code = _env("VOXELFM_CODE")
    spec = args.voxelfm
    if spec is None:
        spec = _env("VOXELFM_WEIGHTS")
    if spec is None:
        spec = VOXELFM_REPO

    weights, _meta = load_release(args.weights)
    model = load_voxelfm(code, spec)
    raw = embed_nifti(model, args.ct, check=True)
    tc, prompt = score_feature(raw, weights)
    print("finding,TC,P")
    for name, a, b in zip(FINDINGS, tc, prompt):
        print(f"{name},{a:.6f},{b:.6f}")


if __name__ == "__main__":
    main()
