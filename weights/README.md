---
license: cc-by-nc-sa-4.0
datasets:
- ibrahimhamamci/CT-RATE
base_model:
- rmaguado/VoxelFM
- Qwen/Qwen3.5-9B
tags:
- medical
- ct
---

# LightAlign CT-RATE weights

Linear map and report-side finding classifiers for 18 CT-RATE findings, from the paper "What Matters for Cross-Modal Alignment".

Code: [github.com/sypsyp97/lightalign](https://github.com/sypsyp97/lightalign) · Project page: [sypsyp97.github.io/lightalign](https://sypsyp97.github.io/lightalign/)

The image encoder is frozen VoxelFM ViT-B 3D (`rmaguado/VoxelFM`, revision `7a58beab42c0ba213f40b9dbcaae43d3488416dd`; code commit `fba41a34c3e4dcfebbe902be5e2b2ce26db27d52`). The feature is the CLS token after block 11, after the final layer norm.

The text encoder is frozen Qwen3.5-9B (`Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`). The feature is the mean over tokens of the middle hidden state (`len(hidden_states) // 2`). The vision tower is not used. The public commit is pinned here. The command that produced the original text arrays did not pass a revision.

The map is ridge regression from image features to text features, fit on all 5,994 paired training patients (seed 0). Each modality is L2-normalized, centered by the mean of those unit vectors on the full training pool, and L2-normalized again. Ridge uses `RidgeCV` with alphas `logspace(-3, 3, 13)` and `fit_intercept=False`. There is no intercept in this file.

Each finding then has a `StandardScaler` and a logistic regression fit on the centered report embeddings and the report-derived label. A new scan is mapped into that space. `TC` is the decision function. `P` is the cosine with the positive prompt embedding minus the cosine with the negative prompt embedding. The prompt strings are `IMPRESSION:\n{label} is present.` and `IMPRESSION:\nNo {label.lower()}.`.

Files: `lightalign-ctrate-n5994-seed0.npz` and `lightalign-ctrate-n5994-seed0.json`.

## Intended use

Research on chest CT representation and zero-shot finding readout.

Not for clinical use. Not a medical device. Do not use the scores to diagnose, triage, or treat a patient.

## Training data

CT-RATE (`ibrahimhamamci/CT-RATE`), gated, license CC BY-NC-SA 4.0. Training rows are the hashed 30% of `train_fixed` reconstructions `*_a_1` after the dataset non-chest list is removed (5,994 patients: 2,024 `sel`, 1,945 `train`, 2,025 `test`). The map uses every row. Labels are the 18 report-derived abnormality columns shipped with CT-RATE. They are not a new manual annotation.

These weights inherit CC BY-NC-SA 4.0.

## How to use

Score one NIfTI volume with the `lightalign` package:

```bash
lightalign-predict --ct volume.nii.gz --weights weights/ --voxelfm-code /path/to/VoxelFM
```

Prediction needs a CUDA GPU, the VoxelFM code checkout (`--voxelfm-code` or `VOXELFM_CODE`), the VoxelFM weights (downloaded from `rmaguado/VoxelFM` by default), and this file. It does not need Qwen3.5-9B. The same package rebuilds this file from CT-RATE embeddings. See the package README for the case rules, the pair counts, and the chest-unit table.

## Limitations

- Research only. The scores are not calibrated probabilities. This file does not contain an operating point. Weighted F1 and accuracy from the reproduction script use a threshold chosen on the evaluated volumes.
- Eighteen findings, taken from report text in one public dataset. A finding the report does not name is a negative label in that table. That is not a radiologist saying the finding is absent on the scan.
- The reader expects a 3D chest CT in NIfTI. Voxels are read from `dataobj` with no rescale slope or intercept. The CT-RATE fixed release already stores corrected Hounsfield units. A file that still stores unscaled values in `dataobj` will be wrong.
- The map is linear. It was fit on 5,994 pairs from CT-RATE. Other hospitals, scanners, and reconstructions were not part of the fit.
- A different Qwen3.5-9B snapshot would not match these coefficients.
- Attention uses fused scaled dot-product attention. Each process checks it once against the reference attention on a reduced volume, and stops if the cosine is below 0.9999.

## License

cc-by-nc-sa-4.0
