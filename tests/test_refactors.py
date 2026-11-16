"""Independent contract checks for measurements and extracted refactors."""

import nibabel as nib
import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import RidgeCV

from align import anchor_zero, ctrate, measures, t2, zoo
from align.align import normalize, polar
from align.baselines import ALPHAS, auc_matrix, zero_shot_scores
from align.anchor_zero import ranks


@pytest.fixture
def masks():
    total = np.zeros((4, 4, 4), dtype=np.int16)
    total[0, 0, :2] = 1
    total[1, 0, 0] = 3
    total[2, 0, :3] = 51
    nodules = np.zeros_like(total)
    nodules[3, 0, 0] = 1
    for point in [(0, 0, 0), (0, 0, 1), (1, 1, 1), (3, 3, 3)]:
        nodules[point] = 2
    effusion = np.zeros_like(total)
    effusion[0, 1, :2] = 2
    effusion[1, 2, 0] = 3
    effusion[2, 3, 0] = 7
    return {
        "ts_total": nib.Nifti1Image(total, np.diag([2.0, 3.0, 4.0, 1.0])),
        "ts_lung_nodules": nib.Nifti1Image(nodules, np.diag([1.0, 2.0, 5.0, 1.0])),
        "ts_pleural_pericard_effusion": nib.Nifti1Image(effusion, np.diag([0.5, 2.0, 2.0, 1.0])),
    }


def test_measure_masks_label_volumes_and_default_connectivity(masks):
    expected = {
        "key": "case-1", "ts_total_1": 0.048, "ts_total_3": 0.024, "ts_total_51": 0.072,
        "ts_lung_nodules_1": 0.01, "ts_lung_nodules_2": 0.04,
        "ts_lung_nodules_components": 3,
        "ts_pleural_pericard_effusion_2": 0.004,
        "ts_pleural_pericard_effusion_3": 0.002,
        "ts_pleural_pericard_effusion_7": 0.002,
    }
    actual = measures.measure_masks("case-1", masks)
    assert set(actual) == set(expected)
    assert actual.pop("key") == expected.pop("key")
    assert actual == pytest.approx(expected)


def test_measure_masks_background_only():
    empty = nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.int16), np.eye(4))
    assert measures.measure_masks("empty", {task: empty for task in measures.TASKS}) == {
        "key": "empty", "ts_lung_nodules_components": 0,
    }


@pytest.mark.parametrize("task", ["ts_total", "ts_lung_nodules", "ts_pleural_pericard_effusion"])
def test_measure_masks_rejects_noninteger_dtype(masks, task):
    original = masks[task]
    masks[task] = nib.Nifti1Image(np.asanyarray(original.dataobj).astype(np.float32), original.affine)
    with pytest.raises(TypeError):
        measures.measure_masks("bad-dtype", masks)


@pytest.mark.parametrize("invalid_label", [3, 9])
def test_measure_masks_rejects_unknown_lung_nodule_label(masks, invalid_label):
    original = masks["ts_lung_nodules"]
    data = np.asanyarray(original.dataobj).copy()
    data[0, 3, 0] = invalid_label
    masks["ts_lung_nodules"] = nib.Nifti1Image(data, original.affine)
    with pytest.raises(ValueError):
        measures.measure_masks("bad-label", masks)


def test_measure_matches_written_nifti_masks(tmp_path, monkeypatch, masks):
    key = "written-case"
    for task, image in masks.items():
        folder = tmp_path / "ts_seg" / task
        folder.mkdir(parents=True)
        nib.save(image, folder / f"{key}.nii.gz")
    monkeypatch.setattr(measures, "CTRATE", str(tmp_path))
    loaded = {task: nib.load(tmp_path / "ts_seg" / task / f"{key}.nii.gz") for task in measures.TASKS}
    assert measures.measure(key) == measures.measure_masks(key, loaded)


def old_inline_seed(x, y, prompts, img_anchor, idx, train, seed, lab, K):
    """Loop body copied from 26cfc6d:align/anchor_zero.py, through the maps."""
    diag = {}
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train)
    half_a, half_b = perm[:len(perm) // 2], perm[len(perm) // 2:]
    pairs = rng.choice(half_a, K, replace=False)
    mx, my = x[half_a].mean(0), y[half_b].mean(0)
    fx, fy = (lambda v: normalize(v - mx)), (lambda v: normalize(v - my))
    pr = fy(prompts)
    txt_anchor = zero_shot_scores(fy(y[half_b]), pr)[:, idx]
    ra, rb = ranks(img_anchor[half_a]), ranks(txt_anchor)
    cost = ((ra[:, None, :] - rb[None, :, :]) ** 2).sum(-1)
    i, j = linear_sum_assignment(cost)
    img_p, txt_p = half_a[i], half_b[j]
    if seed == 0:   # anchor quality, read on the halves' own labels (diagnostic only, never used in the fit)
        diag["image_anchor_auc_vs_label"] = auc_matrix(img_anchor[half_a], lab[half_a][:, idx]).tolist()
        diag["text_anchor_auc_vs_label"] = auc_matrix(txt_anchor, lab[half_b][:, idx]).tolist()
        diag["pseudo_pair_label_agreement"] = (lab[img_p][:, idx] == lab[txt_p][:, idx]).mean(0).tolist()
        diag["random_pair_label_agreement"] = (lab[img_p][:, idx] == lab[rng.permutation(txt_p)][:, idx]).mean(0).tolist()
    xs, ys = fx(x[img_p]), fy(y[txt_p])
    rand = rng.permutation(len(txt_p))
    maps = {"anchor_ridge": RidgeCV(alphas=ALPHAS, fit_intercept=False).fit(xs, ys).coef_.T,
            "anchor_procrustes": polar(xs.T @ ys),
            "random_ridge": RidgeCV(alphas=ALPHAS, fit_intercept=False).fit(xs, ys[rand]).coef_.T,
            "true100_ridge": RidgeCV(alphas=ALPHAS, fit_intercept=False).fit(fx(x[pairs]), fy(y[pairs])).coef_.T}
    return maps, fx, fy, pr, diag


@pytest.fixture
def seed_inputs(monkeypatch):
    monkeypatch.setattr(t2, "K", 20)
    monkeypatch.setattr(anchor_zero, "K", 20)
    rng = np.random.default_rng(27691)
    raw_x = rng.normal(size=(144, 16))
    x = normalize(raw_x)
    y = normalize(raw_x @ rng.normal(size=(16, 12)) + rng.normal(size=(144, 12)))
    prompts = normalize(rng.normal(size=(12, 12)))
    img_anchor = rng.normal(size=(144, 4))
    idx = [0, 2, 3, 5]
    train = rng.permutation(144)[:120]
    lab = rng.integers(0, 2, size=(144, 6))
    return x, y, prompts, img_anchor, idx, train, lab


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_fit_seed_equals_historical_inline_maps(seed_inputs, seed):
    x, y, prompts, img_anchor, idx, train, lab = seed_inputs
    expected, old_fx, old_fy, old_pr, expected_diag = old_inline_seed(
        x, y, prompts, img_anchor, idx, train, seed, lab, 20,
    )
    diag, calls = {}, []

    def on_pairs(rng, half_a, half_b, txt_anchor, img_p, txt_p):
        calls.append(True)
        diag["image_anchor_auc_vs_label"] = auc_matrix(img_anchor[half_a], lab[half_a][:, idx]).tolist()
        diag["text_anchor_auc_vs_label"] = auc_matrix(txt_anchor, lab[half_b][:, idx]).tolist()
        diag["pseudo_pair_label_agreement"] = (lab[img_p][:, idx] == lab[txt_p][:, idx]).mean(0).tolist()
        diag["random_pair_label_agreement"] = (lab[img_p][:, idx] == lab[rng.permutation(txt_p)][:, idx]).mean(0).tolist()

    actual, fx, fy, pr = anchor_zero.fit_seed(
        x, y, prompts, img_anchor, idx, train, seed, on_pairs=on_pairs if seed == 0 else None,
    )
    assert set(actual) == set(expected)
    for name in expected:
        assert np.array_equal(actual[name], expected[name]), name
    assert np.array_equal(fx(x), old_fx(x))
    assert np.array_equal(fy(y), old_fy(y))
    assert np.array_equal(pr, old_pr)
    assert diag == expected_diag
    assert len(calls) == int(seed == 0)


def test_fit_seed_zero_without_diagnostic_changes_only_random_map(seed_inputs):
    x, y, prompts, img_anchor, idx, train, lab = seed_inputs
    expected, _, _, _, _ = old_inline_seed(x, y, prompts, img_anchor, idx, train, 0, lab, 20)
    actual, _, _, _ = anchor_zero.fit_seed(x, y, prompts, img_anchor, idx, train, 0)
    assert not np.array_equal(actual["random_ridge"], expected["random_ridge"])
    for name in ("anchor_ridge", "anchor_procrustes", "true100_ridge"):
        assert np.array_equal(actual[name], expected[name]), name


@pytest.fixture
def hu_volume():
    hu = torch.full((6, 7, 8), -1000.0)
    hu[1:5, 2:6, 3:7] = torch.arange(64, dtype=torch.float32).reshape(4, 4, 4)
    return hu


@pytest.mark.parametrize("name", ["voxelfm_patch", "ctfm", "dinov3_2d", "unknown_encoder"])
def test_encode_volume_cpu_routing(hu_volume, name):
    seen = []
    feature = torch.tensor([1.25, -2.0, 0.0])

    def encoder(volume):
        assert volume.device.type == "cpu"
        seen.append(volume.clone())
        return feature

    result = ctrate.encode_volume({name: encoder}, hu_volume)
    expected_input = hu_volume if name in ("voxelfm_patch", "ctfm") else zoo.crop_body(hu_volume)
    assert len(seen) == 1
    assert torch.equal(seen[0], expected_input)
    assert set(result) == {name}
    assert torch.equal(result[name], feature)


def test_encode_volume_returns_all_named_features(hu_volume):
    features = {
        "voxelfm_patch": torch.tensor([1.0, 2.0]),
        "ctfm": torch.tensor([3.0, 4.0, 5.0]),
        "other": torch.tensor([-1.0]),
    }
    encoders = {name: (lambda volume, feature=feature: feature) for name, feature in features.items()}
    result = ctrate.encode_volume(encoders, hu_volume)
    assert set(result) == set(features)
    for name, feature in features.items():
        assert torch.equal(result[name], feature)


@pytest.mark.parametrize("name", ["voxelfm_patch", "ctfm", "other"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "posinf", "neginf"])
def test_encode_volume_rejects_nonfinite_features(hu_volume, name, bad_value):
    def encoder(volume):
        return torch.tensor([0.0, bad_value])

    with pytest.raises(RuntimeError):
        ctrate.encode_volume({name: encoder}, hu_volume)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="VoxelFM bf16 autocast requires CUDA")
def test_encode_volume_voxelfm_bf16_autocast(hu_volume):
    hu = hu_volume.cuda()
    calls = []

    class Encoder:
        def prepare(self, volume):
            assert volume is hu
            calls.append("prepare")
            return volume + 1

        def model(self, volume):
            calls.append("model")
            assert torch.equal(volume, hu + 1)
            assert torch.is_autocast_enabled("cuda")
            assert torch.get_autocast_dtype("cuda") == torch.bfloat16
            return {"x_norm_clstoken": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16, device=hu.device)}

    result = ctrate.encode_volume({"voxelfm": Encoder()}, hu)
    assert calls == ["prepare", "model"]
    assert result["voxelfm"].dtype == torch.float32
    assert torch.equal(result["voxelfm"], torch.tensor([1.0, 2.0], device=hu.device))
