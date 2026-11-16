"""Training-pool ridge map, report classifiers, and the CT-RATE chest table."""

import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lightalign.pins import (
    QWEN_REPO,
    QWEN_REVISION,
    VOXELFM_COMMIT,
    VOXELFM_FILES,
    VOXELFM_REPO,
    VOXELFM_REVISION,
)
from lightalign.text import finding_prompts, report_text

FINDINGS = (
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
)

SEEDS = range(5)
ALPHAS = np.logspace(-3, 3, 13)
GRID = np.linspace(0, 1, 100)
PAPER_NS = (50, 100, 200, 400, 1600, 5994)
RELEASE_N = 5994
RELEASE_SEED = 0
RELEASE_STEM = "lightalign-ctrate-n5994-seed0"
TRAIN_N = 5994
VALID_N = 3039
CHEST_N = 3002
TRAIN_SPLITS = ("sel", "train", "test")
SPLIT_COUNTS = {"sel": 2024, "train": 1945, "test": 2025}

_REQUIRED = (
    "image_mean",
    "text_mean",
    "ridge_coef",
    "scaler_mean",
    "scaler_scale",
    "logistic_coef",
    "logistic_intercept",
    "prompt_embeddings",
    "finding_names",
)


def normalize(z):
    """Row-wise L2 normalization."""
    n = np.linalg.norm(z, axis=1, keepdims=True)
    if np.any(n == 0) or not np.isfinite(n).all():
        raise ValueError("non-finite or zero-norm row")
    return z / n


def ridge(a, b):
    """Ridge from image rows to text rows. fit_intercept is false."""
    return RidgeCV(alphas=ALPHAS, fit_intercept=False).fit(a, b)


def report_classifiers(text_features, labels):
    """One StandardScaler + logistic regression per finding, on report embeddings.

    The named arguments are the defaults of scikit-learn 1.6.1. Later releases
    move unnamed defaults, so the call states them.
    """
    models = []
    for j in range(labels.shape[1]):
        models.append(make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, fit_intercept=True, solver="lbfgs", max_iter=100, tol=1e-4),
        ).fit(text_features, labels[:, j]))
    return models


def auc_matrix(scores, y):
    """AUC per column via tie-averaged ranks. A single-class column is nan."""
    n = len(y)
    ranks = rankdata(scores, axis=0)
    pos = y.sum(0)
    neg = n - pos
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = ((ranks * y).sum(0) - pos * (pos + 1) / 2) / (pos * neg)
    return np.where((pos > 0) & (neg > 0), auc, np.nan)


def threshold(prob, lab):
    """Grid point closest to the ROC top-left corner. Ties keep the larger threshold."""
    best, best_t = np.inf, 0.0
    pos, neg = lab == 1, lab == 0
    for t in GRID:
        pred = prob > t
        tpr, fpr = (pred & pos).sum() / pos.sum(), (pred & neg).sum() / neg.sum()
        d = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
        if d <= best:
            best, best_t = d, t
    return best_t


def metrics(prob, lab):
    """Mean over findings of weighted F1, accuracy, precision, and recall."""
    rows = []
    for j in range(lab.shape[1]):
        p, l = prob[:, j], lab[:, j]
        pred = (p > threshold(p, l)).astype(int)
        rows.append({
            "f1_weighted": f1_score(l, pred, average="weighted"),
            "accuracy": accuracy_score(l, pred),
            "precision": precision_score(l, pred, zero_division=0),
            "recall": recall_score(l, pred),
        })
    return pd.DataFrame(rows).mean().to_dict()


def minmax(s):
    """Per-finding min-max. A constant finding stays constant (denominator 1)."""
    lo, hi = s.min(0), s.max(0)
    return (s - lo) / np.where(hi > lo, hi - lo, 1)


def dataset_root(root):
    """Directory that contains train_fixed, or its dataset/ child."""
    root = os.path.abspath(root)
    if os.path.isdir(os.path.join(root, "train_fixed")):
        return root
    nested = os.path.join(root, "dataset")
    if os.path.isdir(os.path.join(nested, "train_fixed")):
        return nested
    raise FileNotFoundError(
        "CT-RATE root must contain train_fixed, or dataset/train_fixed. "
        "Row order is the sorted glob order of that directory, and the pair draws depend on it."
    )


def _read_tables(root, subset):
    if subset == "train":
        report_csv = "train_reports.csv"
        label_csv = "train_predicted_labels.csv"
        no_chest_txt = "no_chest_train.txt"
        volume_dir = "train_fixed"
    elif subset == "valid":
        report_csv = "validation_reports.csv"
        label_csv = "valid_predicted_labels.csv"
        no_chest_txt = "no_chest_valid.txt"
        volume_dir = "valid_fixed"
    else:
        raise ValueError(subset)
    report_path = os.path.join(root, "radiology_text_reports", report_csv)
    label_path = os.path.join(root, "multi_abnormality_labels", label_csv)
    chest_path = os.path.join(root, "metadata", no_chest_txt)
    for path in (report_path, label_path, chest_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    reports = pd.read_csv(report_path)
    labels = pd.read_csv(label_path)
    if "VolumeName" not in reports.columns or "VolumeName" not in labels.columns:
        raise KeyError("VolumeName")
    reports = reports.set_index("VolumeName")
    labels = labels.set_index("VolumeName")
    if tuple(labels.columns) != FINDINGS:
        raise ValueError(
            "label columns differ from the CT-RATE 18-finding order: "
            + ", ".join(str(c) for c in labels.columns)
        )
    for col in ("Findings_EN", "Impressions_EN"):
        if col not in reports.columns:
            raise KeyError(col)
    with open(chest_path, encoding="utf-8") as f:
        no_chest = {os.path.basename(p) for p in f.read().split()}
    return reports, labels, no_chest, volume_dir


def _case_text_and_labels(reports, labels, name):
    if name not in reports.index or name not in labels.index:
        raise KeyError(f"{name} has no report or no labels")
    report = reports.loc[name]
    lab = labels.loc[name]
    if not isinstance(report, pd.Series) or not isinstance(lab, pd.Series):
        raise ValueError(f"duplicate table row for {name}")
    text = report_text(report["Findings_EN"], report["Impressions_EN"])
    fields = {}
    for c in labels.columns:
        number = float(lab[c])
        if number not in (0.0, 1.0):
            raise ValueError(f"{name} label {c} is {lab[c]!r}")
        fields[c] = int(number)
    return text, fields


def _write_cases(out, cases, names):
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, "cases.jsonl.tmp")
    cases.to_json(tmp, orient="records", lines=True)
    os.replace(tmp, os.path.join(out, "cases.jsonl"))
    payload = {"labels": list(names), "prompts": finding_prompts(names)}
    tmpj = os.path.join(out, "labels.json.tmp")
    with open(tmpj, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
        f.write("\n")
    os.replace(tmpj, os.path.join(out, "labels.json"))


def _split_counts(cases):
    return {k: int((cases["split"] == k).sum()) for k in TRAIN_SPLITS}


def chest_mask(cases):
    """Volumes absent from the dataset non-chest list. Raises if that column is missing."""
    if "no_chest" not in cases.columns:
        raise KeyError("validation cases have no no_chest column; chest unit is undefined")
    mask = []
    for v in cases["no_chest"].tolist():
        if v == True:
            mask.append(False)
        elif v == False:
            mask.append(True)
        else:
            raise TypeError(f"no_chest value {v!r} is not true or false")
    return np.array(mask, dtype=bool)


def assert_train_pool(cases):
    counts = _split_counts(cases)
    if len(cases) != TRAIN_N or counts != SPLIT_COUNTS:
        raise RuntimeError(
            f"training pool is {len(cases)} rows with splits {counts}; "
            f"expected {TRAIN_N} rows and {SPLIT_COUNTS}"
        )


def assert_valid_pool(cases):
    if len(cases) != VALID_N:
        raise RuntimeError(f"validation set has {len(cases)} volumes, expected {VALID_N}")
    n_chest = int(chest_mask(cases).sum())
    if n_chest != CHEST_N:
        raise RuntimeError(f"chest unit has {n_chest} volumes, expected {CHEST_N}")


def split_of_train(patient):
    """Keep 30 percent of patients. The next hash digit assigns sel, train, or test."""
    h = int(hashlib.sha256(patient.encode()).hexdigest(), 16)
    if h % 10 >= 3:
        return None
    return TRAIN_SPLITS[(h // 10) % 3]


def build_train_cases(root, out):
    """Hashed 30 percent of train_fixed *_a_1 volumes, non-chest names removed."""
    root = dataset_root(root)
    reports, labels, no_chest, volume_dir = _read_tables(root, "train")
    pattern = os.path.join(root, volume_dir, "*", "*_a", "*_a_1.nii.gz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no volumes matched {pattern}")
    rows = []
    for path in files:
        name = os.path.basename(path)
        if name in no_chest:
            continue
        patient = name.split("_")[1]
        split = split_of_train(patient)
        if split is None:
            continue
        text, fields = _case_text_and_labels(reports, labels, name)
        rows.append({
            "id": name.replace(".nii.gz", ""),
            "key": os.path.relpath(path, root).replace(".nii.gz", ""),
            "split": split,
            "report": text,
            **fields,
        })
    cases = pd.DataFrame(rows)
    assert_train_pool(cases)
    _write_cases(out, cases, list(labels.columns))
    print(f"train files={len(files)} kept={len(cases)} splits={_split_counts(cases)}", flush=True)


def build_valid_cases(root, out):
    """Every valid_fixed reconstruction. All rows are split test."""
    root = dataset_root(root)
    reports, labels, no_chest, volume_dir = _read_tables(root, "valid")
    pattern = os.path.join(root, volume_dir, "*", "*", "*.nii.gz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no volumes matched {pattern}")
    rows = []
    for path in files:
        name = os.path.basename(path)
        text, fields = _case_text_and_labels(reports, labels, name)
        _, patient, scan, recon = name.replace(".nii.gz", "").split("_")
        rows.append({
            "id": name.replace(".nii.gz", ""),
            "key": os.path.relpath(path, root).replace(".nii.gz", ""),
            "split": "test",
            "report": text,
            "no_chest": name in no_chest,
            "study": f"{patient}_{scan}",
            "recon": int(recon),
            **fields,
        })
    cases = pd.DataFrame(rows)
    if len(cases) != len(labels):
        raise ValueError(f"{len(cases)} volumes on disk, {len(labels)} in the label table")
    assert_valid_pool(cases)
    _write_cases(out, cases, list(labels.columns))
    print(
        f"valid volumes={len(cases)} no_chest={int(cases.no_chest.sum())} "
        f"studies={cases.study.nunique()}",
        flush=True,
    )


def load_cases(emb):
    path = os.path.join(emb, "cases.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    cases = pd.read_json(path, lines=True)
    for col in ("id", "key", "split", "report"):
        if col not in cases.columns:
            raise KeyError(f"cases.jsonl missing {col}")
    return cases


def load_label_meta(emb):
    path = os.path.join(emb, "labels.json")
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    names = meta["labels"]
    prompts = meta["prompts"]
    if tuple(names) != FINDINGS:
        raise ValueError("finding names differ from the CT-RATE 18-label order")
    if list(prompts) != finding_prompts(names):
        raise ValueError("prompts differ from the positive/negative impression template")
    return names


def load_image_matrix(emb, cases):
    rows = []
    for key in cases["key"].tolist():
        path = os.path.join(emb, "image", os.path.basename(str(key)) + ".npy")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        rows.append(np.load(path))
    x = np.stack(rows).astype(np.float64)
    if x.ndim != 2 or len(x) != len(cases) or not np.isfinite(x).all():
        raise ValueError(f"bad image matrix in {emb} shape={x.shape}")
    return x


def load_matrix(path, n_rows):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    a = np.load(path).astype(np.float64)
    if a.ndim != 2 or len(a) != n_rows or not np.isfinite(a).all():
        raise ValueError(f"bad array {path} shape={a.shape}")
    return a


def label_matrix(cases):
    lab = cases[list(FINDINGS)].to_numpy()
    if lab.shape != (len(cases), len(FINDINGS)):
        raise ValueError(f"label shape {lab.shape}")
    if not np.isin(lab, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    return lab.astype(np.float64)


def _pack_classifiers(models):
    means, scales, coefs, intercepts = [], [], [], []
    for pipe in models:
        scaler = pipe.named_steps["standardscaler"]
        linear = pipe.named_steps["logisticregression"]
        means.append(np.asarray(scaler.mean_, dtype=np.float64))
        scales.append(np.asarray(scaler.scale_, dtype=np.float64))
        coef = np.asarray(linear.coef_, dtype=np.float64)
        if coef.shape[0] != 1:
            raise RuntimeError(f"expected a binary classifier, coef shape {coef.shape}")
        coefs.append(coef.reshape(-1))
        icept = np.asarray(linear.intercept_, dtype=np.float64).reshape(-1)
        if icept.shape != (1,):
            raise RuntimeError(f"expected one intercept, got shape {icept.shape}")
        intercepts.append(float(icept[0]))
    return (
        np.stack(means),
        np.stack(scales),
        np.stack(coefs),
        np.asarray(intercepts, dtype=np.float64),
    )


def _release_meta(alpha):
    import sklearn
    alpha = np.asarray(alpha, dtype=np.float64)
    return {
        "format": RELEASE_STEM,
        "n_pairs": RELEASE_N,
        "seed": RELEASE_SEED,
        "findings": list(FINDINGS),
        "image": {
            "repo": VOXELFM_REPO,
            "revision": VOXELFM_REVISION,
            "code": "https://github.com/rmaguado/VoxelFM",
            "code_commit": VOXELFM_COMMIT,
            "files": list(VOXELFM_FILES),
            "feature": "normed CLS token after block 11",
            "attention": "fused scaled_dot_product_attention",
            "dtype": "bfloat16",
        },
        "text": {
            "repo": QWEN_REPO,
            "revision": QWEN_REVISION,
            "pooling": "mean over tokens of hidden_states[len(hidden_states)//2]",
            "max_length": 1024,
            "dtype": "bfloat16",
            "padding_side": "right",
            "loader": "AutoModelForImageTextToText; vision tower deleted",
        },
        "normalization": (
            "L2-normalize, subtract the mean of the training-pool unit vectors, L2-normalize again"
        ),
        "ridge": {
            "alphas": "logspace(-3, 3, 13)",
            "fit_intercept": False,
            "alpha": alpha.tolist(),
        },
        "logistic_regression": {
            "C": 1.0,
            "fit_intercept": True,
            "solver": "lbfgs",
            "max_iter": 100,
            "tol": 0.0001,
            "scaler": "StandardScaler",
            "note": "explicit scikit-learn 1.6.1 defaults",
        },
        "prompt_embeddings": "centered and L2-normalized; positive then negative per finding",
        "fit_libraries": {
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
    }


def _save_release(npz_path, arrays, meta):
    folder = os.path.dirname(os.path.abspath(npz_path))
    os.makedirs(folder, exist_ok=True)
    tmp = npz_path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, npz_path)
    json_path = npz_path[:-4] + ".json"
    tmpj = json_path + ".tmp"
    with open(tmpj, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    os.replace(tmpj, json_path)


def fit_release(train_dir, npz_path):
    """Fit n=5994, seed 0 on the full training pool and write the release files."""
    if not str(npz_path).endswith(".npz"):
        raise ValueError("out path must end with .npz")
    train = load_cases(train_dir)
    assert_train_pool(train)
    load_label_meta(train_dir)
    image = load_image_matrix(train_dir, train)
    text = load_matrix(os.path.join(train_dir, "text.npy"), len(train))
    prompts = load_matrix(os.path.join(train_dir, "prompts.npy"), len(FINDINGS) * 2)
    labels = label_matrix(train)
    xt = normalize(image)
    y = normalize(text)
    prompts_u = normalize(prompts)
    pool = np.arange(len(xt))
    if len(pool) != RELEASE_N:
        raise RuntimeError(f"release fit requires {RELEASE_N} training rows")
    mx = xt.mean(0)
    my = y.mean(0)

    def fx(v, mean=mx):
        return normalize(v - mean)

    def fy(v, mean=my):
        return normalize(v - mean)

    perm = np.random.default_rng(RELEASE_SEED).permutation(pool)
    model = ridge(fx(xt[perm[:RELEASE_N]]), fy(y[perm[:RELEASE_N]]))
    coef = np.asarray(model.coef_, dtype=np.float64)
    if coef.shape != (text.shape[1], image.shape[1]):
        raise RuntimeError(f"ridge coef shape {coef.shape}")
    clfs = report_classifiers(fy(y), labels)
    scaler_mean, scaler_scale, logistic_coef, logistic_intercept = _pack_classifiers(clfs)
    pr = np.asarray(fy(prompts_u), dtype=np.float64)
    if logistic_coef.shape != (len(FINDINGS), text.shape[1]):
        raise RuntimeError(f"logistic coef shape {logistic_coef.shape}")
    if pr.shape != (len(FINDINGS) * 2, text.shape[1]):
        raise RuntimeError(f"prompt embedding shape {pr.shape}")
    alpha = np.asarray(model.alpha_, dtype=np.float64)
    arrays = {
        "image_mean": np.asarray(mx, dtype=np.float64),
        "text_mean": np.asarray(my, dtype=np.float64),
        "ridge_coef": coef,
        "ridge_alpha": alpha,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "logistic_coef": logistic_coef,
        "logistic_intercept": logistic_intercept,
        "prompt_embeddings": pr,
        "finding_names": np.array(FINDINGS),
    }
    _save_release(npz_path, arrays, _release_meta(alpha))
    print(f"wrote {npz_path}", flush=True)


def _reader_means(scores, labels):
    aucs, f1s, accs = [], [], []
    for s in scores:
        aucs.append(float(np.nanmean(auc_matrix(s, labels))))
        m = metrics(minmax(s), labels)
        f1s.append(m["f1_weighted"])
        accs.append(m["accuracy"])
    return {
        "auroc": float(np.mean(aucs)),
        "f1_weighted": float(np.mean(f1s)),
        "accuracy": float(np.mean(accs)),
    }


def reproduction_table(train_dir, valid_dir):
    """Pair counts 50, 100, 200, 400, 1600, 5994 on the 3002 chest volumes.

    Centering and the report classifiers use the full training pool.
    Only the ridge pairs change with n and the seed.
    """
    train = load_cases(train_dir)
    valid = load_cases(valid_dir)
    assert_train_pool(train)
    assert_valid_pool(valid)
    load_label_meta(train_dir)
    load_label_meta(valid_dir)
    xt = normalize(load_image_matrix(train_dir, train))
    xv = normalize(load_image_matrix(valid_dir, valid))
    y = normalize(load_matrix(os.path.join(train_dir, "text.npy"), len(train)))
    prompts = normalize(load_matrix(os.path.join(train_dir, "prompts.npy"), len(FINDINGS) * 2))
    lt = label_matrix(train)
    lv = label_matrix(valid)
    mask = chest_mask(valid)
    pool = np.arange(len(xt))
    mx = xt.mean(0)
    my = y.mean(0)

    def fx(v, mean=mx):
        return normalize(v - mean)

    def fy(v, mean=my):
        return normalize(v - mean)

    # lbfgs on this matrix does not use the pair seed. The ablation refits it per seed.
    clfs = report_classifiers(fy(y), lt)
    pr = fy(prompts)
    lv_chest = lv[mask]
    rows = []
    for n in PAPER_NS:
        if n > len(pool):
            raise ValueError(f"n = {n} > {len(pool)} pool patients")
        seeds = list(SEEDS if n < len(pool) else [0])
        per_reader = {"P": [], "TC": []}
        for seed in seeds:
            perm = np.random.default_rng(seed).permutation(pool)
            coef = ridge(fx(xt[perm[:n]]), fy(y[perm[:n]])).coef_
            u = fx(xv) @ coef.T
            per_reader["P"].append(normalize(u) @ (pr[0::2] - pr[1::2]).T)
            per_reader["TC"].append(np.stack([c.decision_function(u) for c in clfs], 1))
        for reader in ("TC", "P"):
            chest_scores = [s[mask] for s in per_reader[reader]]
            stats = _reader_means(chest_scores, lv_chest)
            rows.append({
                "n": n,
                "seeds": len(seeds),
                "volumes": int(mask.sum()),
                "reader": reader,
                **stats,
            })
    return rows


def resolve_release(spec):
    """npz path, directory, or Hugging Face repo id. The JSON sits beside the npz."""
    npz_name = RELEASE_STEM + ".npz"
    json_name = RELEASE_STEM + ".json"
    if os.path.isfile(spec):
        if not spec.endswith(".npz"):
            raise ValueError("weights file must be a .npz")
        npz = os.path.abspath(spec)
        js = npz[:-4] + ".json"
    elif os.path.isdir(spec):
        npz = os.path.join(os.path.abspath(spec), npz_name)
        js = os.path.join(os.path.abspath(spec), json_name)
    else:
        from huggingface_hub import hf_hub_download
        npz = hf_hub_download(repo_id=spec, filename=npz_name)
        js = hf_hub_download(repo_id=spec, filename=json_name)
    if not os.path.isfile(npz):
        raise FileNotFoundError(npz)
    if not os.path.isfile(js):
        raise FileNotFoundError(js)
    return npz, js


def load_release(spec):
    npz_path, json_path = resolve_release(spec)
    with open(json_path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta["n_pairs"] != RELEASE_N or meta["seed"] != RELEASE_SEED:
        raise ValueError("weights JSON is not the n=5994 seed=0 release")
    if meta["ridge"]["fit_intercept"] is not False:
        raise ValueError("ridge fit_intercept must be false")
    if tuple(meta["findings"]) != FINDINGS:
        raise ValueError("weights JSON findings do not match")
    z = np.load(npz_path)
    for key in _REQUIRED:
        if key not in z.files:
            raise KeyError(key)
    names = tuple(str(x) for x in z["finding_names"].tolist())
    if names != FINDINGS:
        raise ValueError("weights finding names do not match")
    text_dim = z["ridge_coef"].shape[0]
    image_dim = z["ridge_coef"].shape[1]
    n_find = len(FINDINGS)
    expected = {
        "image_mean": (image_dim,),
        "text_mean": (text_dim,),
        "scaler_mean": (n_find, text_dim),
        "scaler_scale": (n_find, text_dim),
        "logistic_coef": (n_find, text_dim),
        "logistic_intercept": (n_find,),
        "prompt_embeddings": (n_find * 2, text_dim),
    }
    for key, shape in expected.items():
        if z[key].shape != shape:
            raise ValueError(f"{key} shape {z[key].shape} != {shape}")
    return z, meta


def map_image(raw, image_mean, ridge_coef):
    """Unit image, subtract the training mean, unit again, then ridge into text space."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 1:
        raise ValueError(f"expected one feature vector, got shape {raw.shape}")
    if raw.shape != np.asarray(image_mean).shape:
        raise ValueError(
            f"feature dim {raw.shape[0]} does not match image mean {np.asarray(image_mean).shape[0]}"
        )
    unit = normalize(raw.reshape(1, -1))
    mapped = normalize(unit - np.asarray(image_mean, dtype=np.float64)) @ np.asarray(ridge_coef, dtype=np.float64).T
    return mapped[0]


def tc_scores(u, scaler_mean, scaler_scale, coef, intercept):
    """Per-finding StandardScaler + binary decision_function, in numpy."""
    xs = (u - scaler_mean) / scaler_scale
    return (xs * coef).sum(axis=1) + intercept


def prompt_scores(u, prompt_embeddings):
    """Cosine with the positive prompt minus cosine with the negative prompt."""
    norm = np.linalg.norm(u)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("non-finite or zero-norm mapped embedding")
    diff = prompt_embeddings[0::2] - prompt_embeddings[1::2]
    return (u / norm) @ diff.T


def score_feature(raw, weights):
    u = map_image(raw, weights["image_mean"], weights["ridge_coef"])
    tc = tc_scores(
        u, weights["scaler_mean"], weights["scaler_scale"],
        weights["logistic_coef"], weights["logistic_intercept"],
    )
    prompt = prompt_scores(u, weights["prompt_embeddings"])
    if tc.shape != (len(FINDINGS),) or prompt.shape != (len(FINDINGS),):
        raise RuntimeError(f"score shapes {tc.shape} {prompt.shape}")
    if not np.isfinite(tc).all() or not np.isfinite(prompt).all():
        raise RuntimeError("non-finite scores")
    return tc, prompt
