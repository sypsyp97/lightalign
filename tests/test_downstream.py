"""Contract tests for the downstream pair-budget experiment.

Expected budgets and criterion signs come from the registered equations. Small
feature dimensions keep the full five-seed, eight-arm run inexpensive; only GW
is replaced, so RidgeCV and all evaluation metrics still run.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score

from align import downstream
from align.baselines import ALPHAS


PAIR_COUNTS = (0, 25, 50, 100, 200, 400, 800, 1600)
SEEDS = (0, 1, 2, 3, 4)
ENCODERS = tuple(f"encoder_{i}" for i in range(5))


def unit_rows(values):
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def centered(values, indices):
    """Common embedding protocol: row L2, training mean, then row L2."""
    values = unit_rows(values)
    return unit_rows(values - values[indices].mean(axis=0))


@pytest.fixture
def embedding_dir(tmp_path):
    def write(n_train=1601):
        emb = tmp_path / f"emb_{n_train}"
        encoder = "synthetic"
        (emb / encoder).mkdir(parents=True)
        rng = np.random.default_rng(761)
        splits = np.array(["train"] * n_train + ["test"] * 12 + ["sel"] * 5)
        rng.shuffle(splits)
        x = rng.normal(size=(len(splits), 5)) + np.array([2, -1, 0.5, 0, 1])
        y = x @ rng.normal(size=(5, 4)) + rng.normal(size=(len(splits), 4))
        prompts = rng.normal(size=(36, 4))
        names = [f"finding_{j:02d}" for j in range(18)]
        labels = (np.arange(len(splits))[:, None] + np.arange(18)) % 2
        test = np.flatnonzero(splits == "test")
        labels[test] = (np.arange(len(test))[:, None] + np.arange(18)) % 2
        keys = [f"volumes/patient_{i:04d}" for i in range(len(splits))]
        rows = [{"key": key, "split": split, **dict(zip(names, row.tolist()))}
                for key, split, row in zip(keys, splits, labels)]
        pd.DataFrame(rows).to_json(emb / "cases.jsonl", orient="records", lines=True)
        (emb / "labels.json").write_text(json.dumps({"labels": names}))
        for key, row in zip(keys, x):
            np.save(emb / encoder / f"{key.rsplit('/', 1)[-1]}.npy", row)
        np.save(emb / "textmid_4B.npy", y)
        np.save(emb / "promptsmid_4B.npy", prompts)
        return SimpleNamespace(
            args=SimpleNamespace(emb=str(emb), out=str(tmp_path / f"out_{n_train}"), encoder=encoder),
            x=x, y=y, prompts=prompts, labels=labels, keys=keys,
            train=np.flatnonzero(splits == "train"), test=test,
        )

    return write


@pytest.fixture
def observed_fits(monkeypatch):
    calls = []
    pending = []
    real_ridge, real_metrics = downstream.ridge, downstream.metrics

    def ridge(x, y):
        model = real_ridge(x, y)
        pending.append({"kind": "ridge", "x": x.copy(), "y": y.copy(), "model": model})
        return model

    def gw(x, y, seed):
        gx = lambda v: unit_rows(v - np.linspace(0.1, 0.5, x.shape[1]))
        gy = lambda v: unit_rows(v + np.linspace(0.2, 0.8, y.shape[1]))
        pending.append({"kind": "gw", "x": x.copy(), "y": y.copy(), "seed": seed, "gx": gx, "gy": gy})
        w = np.random.default_rng(9000 + seed).normal(size=(x.shape[1], y.shape[1]))
        return w, gx, gy

    def metrics(w, xt, yt, prompts, lab):
        fit = pending.pop()
        fit.update(w=w.copy(), xt=xt.copy(), yt=yt.copy(), prompts=prompts.copy(), lab=lab.copy())
        calls.append(fit)
        # Associate output rows with observations without requiring CSV row order.
        return {**real_metrics(w, xt, yt, prompts, lab), "observed_call": len(calls) - 1}

    monkeypatch.setattr(downstream, "ridge", ridge)
    monkeypatch.setattr(downstream, "gromov_wasserstein", gw)
    monkeypatch.setattr(downstream, "metrics", metrics)
    return calls


def read_curve(data):
    return pd.read_csv(f"{data.args.out}/{data.args.encoder}_curve.csv")


def test_run_full_grid_nested_train_pairs_and_train_centering(embedding_dir, observed_fits):
    data = embedding_dir()
    downstream.run(data.args)
    curve = read_curve(data)
    assert len(curve) == 5 * 8
    assert set(zip(curve.seed, curve.n)) == {(seed, n) for seed in SEEDS for n in PAIR_COUNTS}
    assert set(curve.encoder) == {data.args.encoder}
    metric_columns = ["mean_auc", "auc_resid", "foscttm", "r@1", "r@5", "r@10"]
    assert np.isfinite(curve[metric_columns].to_numpy()).all()
    assert curve[metric_columns].ge(0).all().all()
    assert curve[metric_columns].le(1).all().all()

    xc, yc = centered(data.x, data.train), centered(data.y, data.train)
    pc = unit_rows(unit_rows(data.prompts) - unit_rows(data.y)[data.train].mean(axis=0))
    train_tree = cKDTree(xc[data.train])
    seed_draws = []
    for seed in SEEDS:
        previous = np.array([], dtype=int)
        for n in PAIR_COUNTS[1:]:
            row = curve[(curve.seed == seed) & (curve.n == n)].iloc[0]
            fit = observed_fits[int(row.observed_call)]
            assert fit["kind"] == "ridge"
            assert fit["x"].shape == (n, 5)
            distance, local_ids = train_tree.query(fit["x"])
            np.testing.assert_allclose(distance, 0, atol=1e-12)
            ids = data.train[local_ids]
            assert len(set(ids)) == n
            np.testing.assert_array_equal(ids[:len(previous)], previous)
            np.testing.assert_allclose(fit["y"], yc[ids], atol=1e-12)
            np.testing.assert_allclose(fit["xt"], xc[data.test], atol=1e-12)
            np.testing.assert_allclose(fit["yt"], yc[data.test], atol=1e-12)
            np.testing.assert_allclose(fit["prompts"], pc, atol=1e-12)
            np.testing.assert_array_equal(fit["lab"], data.labels[data.test])
            assert isinstance(fit["model"], RidgeCV)
            np.testing.assert_array_equal(fit["model"].alphas, ALPHAS)
            assert fit["model"].fit_intercept is False
            previous = ids
        seed_draws.append(tuple(previous))
    assert len(set(seed_draws)) == 5


def test_zero_pair_arm_uses_disjoint_train_halves_and_aligner_preprocessing(embedding_dir, observed_fits):
    data = embedding_dir()
    downstream.run(data.args)
    curve = read_curve(data)
    for seed in SEEDS:
        row = curve[(curve.seed == seed) & (curve.n == 0)].iloc[0]
        fit = observed_fits[int(row.observed_call)]
        assert fit["kind"] == "gw"
        assert fit["seed"] == seed
        # The seed convention is the common align protocol named by the docstring.
        perm = np.random.default_rng(seed).permutation(data.train)
        half_a, half_b = np.array_split(perm, [len(perm) // 2])
        assert set(half_a).isdisjoint(half_b)
        assert set(half_a) | set(half_b) == set(data.train)
        np.testing.assert_allclose(fit["x"], centered(data.x, half_a)[half_a], atol=1e-12)
        np.testing.assert_allclose(fit["y"], centered(data.y, half_b)[half_b], atol=1e-12)
        np.testing.assert_allclose(fit["xt"], fit["gx"](centered(data.x, half_a)[data.test]), atol=1e-12)
        np.testing.assert_allclose(fit["yt"], fit["gy"](centered(data.y, half_b)[data.test]), atol=1e-12)
        pc = unit_rows(unit_rows(data.prompts) - unit_rows(data.y)[half_b].mean(axis=0))
        np.testing.assert_allclose(fit["prompts"], fit["gy"](pc), atol=1e-12)


def test_test_patients_cannot_change_fits_or_seed_draws(embedding_dir, observed_fits):
    data = embedding_dir()
    downstream.run(data.args)
    first = read_curve(data)
    original_calls = list(observed_fits)
    observed_fits.clear()
    rng = np.random.default_rng(972)
    for index in data.test:
        key = data.keys[index].rsplit("/", 1)[-1]
        np.save(f"{data.args.emb}/{data.args.encoder}/{key}.npy", rng.normal(size=5) * 50 + 100)
    changed_y = data.y.copy()
    changed_y[data.test] = rng.normal(size=(len(data.test), 4)) * 80 - 100
    np.save(f"{data.args.emb}/textmid_4B.npy", changed_y)
    downstream.run(data.args)
    second = read_curve(data)
    for seed in SEEDS:
        for n in PAIR_COUNTS:
            before = first[(first.seed == seed) & (first.n == n)].iloc[0]
            after = second[(second.seed == seed) & (second.n == n)].iloc[0]
            old = original_calls[int(before.observed_call)]
            new = observed_fits[int(after.observed_call)]
            for field in ("x", "y", "w", "prompts"):
                np.testing.assert_allclose(new[field], old[field], rtol=0, atol=1e-12)
            assert not np.allclose(new["xt"], old["xt"])
            assert not np.allclose(new["yt"], old["yt"])


def test_run_rejects_fewer_than_1600_train_patients(embedding_dir, observed_fits):
    # Total patients exceed 1600, but neither test nor sel may fill the pair budget.
    data = embedding_dir(n_train=1599)
    with pytest.raises(ValueError, match="1600"):
        downstream.run(data.args)
    assert not observed_fits


@pytest.mark.parametrize("direction,auc,fos,recall", [(1, 1, 0, 1), (-1, 0, 1, 0)])
def test_metrics_have_correct_classification_and_retrieval_direction(direction, auc, fos, recall):
    rng = np.random.default_rng(182)
    labels = np.zeros((12, 18))
    for j in range(18):
        labels[rng.choice(12, 6, replace=False), j] = 1
    prompts = np.empty((36, 12))
    prompts[0::2], prompts[1::2] = labels.T, 1 - labels.T
    # Continuous scores avoid numerical ties after common-axis regression.
    prompts += rng.normal(scale=0.01, size=prompts.shape)
    result = downstream.metrics(direction * np.eye(12), np.eye(12), np.eye(12), unit_rows(prompts), labels)
    assert result["mean_auc"] == pytest.approx(auc)
    scores = direction * (unit_rows(prompts)[0::2] - unit_rows(prompts)[1::2]).T
    design = np.column_stack((np.ones(len(scores)), scores.mean(axis=1)))
    coefficients = np.linalg.lstsq(design, scores, rcond=None)[0]
    expected_residual_auc = roc_auc_score(labels, scores - design @ coefficients)
    assert result["auc_resid"] == pytest.approx(expected_residual_auc)
    assert result["foscttm"] == pytest.approx(fos)
    for k in (1, 5, 10):
        assert result[f"r@{k}"] == pytest.approx(recall)


def test_residual_removes_a_pure_common_axis():
    rng = np.random.default_rng(1037)
    common = rng.normal(size=4096)
    loading = np.linspace(0.5, 1.5, 18)
    noise = rng.normal(scale=0.15, size=(len(common), len(loading)))
    noise -= noise.mean(axis=1, keepdims=True)
    scores = common[:, None] * loading + noise
    labels = (common[:, None] > np.linspace(-0.5, 0.5, len(loading))).astype(int)
    # Mean loading is one and row-mean noise is zero, so the score mean is common.
    np.testing.assert_allclose(scores.mean(axis=1), common, atol=1e-12)
    assert roc_auc_score(labels, scores) > 0.95

    residual_scores = downstream.residual(scores)

    assert residual_scores.shape == scores.shape
    assert np.isfinite(residual_scores).all()
    assert roc_auc_score(labels, residual_scores) == pytest.approx(0.5, abs=0.03)


def test_residual_keeps_finding_specific_signal():
    rng = np.random.default_rng(2081)
    common = rng.normal(scale=4, size=4096)
    specific = rng.normal(size=(len(common), 18))
    specific -= specific.mean(axis=1, keepdims=True)
    scores = common[:, None] * np.linspace(0.5, 1.5, 18) + specific
    finding = 5
    labels = (specific[:, finding] > 0).astype(int)

    residual_scores = downstream.residual(scores)

    assert roc_auc_score(labels, residual_scores[:, finding]) > 0.95


@pytest.fixture
def summary_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(downstream, "ENCODERS", ENCODERS)
    out, spiked = tmp_path / "curves", tmp_path / "spiked"
    out.mkdir()
    spiked.mkdir()

    def write(*, spectra=None, ns=PAIR_COUNTS, bad_task=None, reverse_counts=(), reverse_one_encoder=False,
              auc_resid=None, encoders=ENCODERS):
        for i, encoder in enumerate(encoders):
            r_hat = [0.6 + i * 0.01, 0.3 + i * 0.01, 0.1 + i * 0.01] if spectra is None else spectra[encoder]
            estimates = {
                "4B/mid/64": {"r_hat": r_hat},
                "4B/last/64": {"r_hat": [0.99] * 9},
                "4B/mid/16": {"r_hat": [0.99] * 7},
                "2B/mid/64": {"r_hat": [0.99] * 5},
            }
            (spiked / f"{encoder}_estimates.json").write_text(json.dumps(estimates))
            rows = []
            for seed in SEEDS:
                for j, n in enumerate(ns):
                    auc_sign = -1 if bad_task == "mean_auc" else 1
                    fos_sign = 1 if bad_task == "foscttm" else -1
                    if n in reverse_counts or (reverse_one_encoder and i == 0):
                        fos_sign *= -1
                    auc = 0.6 + auc_sign * (i + j) * 0.01
                    resid = 0.55 + (i + j) * 0.01 if auc_resid is None else auc_resid[encoder][n]
                    fos = 0.4 + fos_sign * (i + j) * 0.01
                    if n == 0:
                        auc, fos = 0.99, 0.01  # D1 must exclude the zero-pair arm.
                        if auc_resid is None:
                            resid = 0.99  # D3 must also exclude the zero-pair arm.
                    rows.append({"encoder": encoder, "seed": seed, "n": n,
                                 "mean_auc": auc + (seed - 2) * 0.0001,
                                 "auc_resid": resid + (seed - 2) * 0.0001,
                                 "foscttm": fos + (seed - 2) * 0.0001,
                                 "r@1": 0.1, "r@5": 0.3, "r@10": 0.5})
            pd.DataFrame(rows).to_csv(out / f"{encoder}_curve.csv", index=False)
        return SimpleNamespace(out=str(out), spiked=str(spiked))

    return write


def d3_scores(pair_ranks, strength_ranks, *, ns=PAIR_COUNTS, encoders=ENCODERS):
    positive_counts = [n for n in ns if n > 0]
    return {
        encoder: {0: 0.99, **{
            n: 0.5 + 0.01 * pair_rank + 0.02 * strength_rank
            for n, pair_rank in zip(positive_counts, pair_ranks, strict=True)
        }}
        for encoder, strength_rank in zip(encoders, strength_ranks, strict=True)
    }


def assert_d3_summary(result, pair_correlations, strength_correlations, expected_pass):
    # The criterion fixes both medians, whether JSON stores medians or individual correlations.
    for field, expected in (("D3_n", pair_correlations), ("D3_S", strength_correlations)):
        reported = result[field]
        if isinstance(reported, dict):
            assert reported == pytest.approx(expected)
        else:
            assert reported == pytest.approx(np.median(list(expected.values())))
    assert result["D3_pass"] is expected_pass


def test_summary_budget_counts_strengths_and_boundary(summary_inputs, monkeypatch):
    ns = (0, 25, 50, 100, 128, 200, 256, 400, 800, 1600)
    monkeypatch.setattr(downstream, "NS", ns)
    spectra = {e: [0.8 + i * 0.01, 0.5, 1 / 3, 0.1, 0] for i, e in enumerate(ENCODERS)}
    args = summary_inputs(spectra=spectra, ns=ns)
    downstream.summary(args)
    curves = pd.read_csv(f"{args.out}/curves.csv")
    assert len(curves) == len(ENCODERS) * len(ns)
    source = pd.concat([pd.read_csv(f"{args.out}/{e}_curve.csv") for e in ENCODERS])
    expected_residual = source.groupby(["encoder", "n"])["auc_resid"].mean().sort_index()
    actual_residual = curves.set_index(["encoder", "n"])["auc_resid"].sort_index()
    pd.testing.assert_series_equal(actual_residual, expected_residual)
    for row in curves.itertuples():
        # rc(d/n, d/n) = d / (n-d); equality is not detectable.
        visible = [] if row.n <= 128 else [r for r in spectra[row.encoder] if r > 64 / (row.n - 64)]
        assert row.k_pred == len(visible)
        assert row.S == pytest.approx(sum(visible), abs=1e-12)
    at_equality = curves[(curves.encoder == ENCODERS[0]) & (curves.n == 256)].iloc[0]
    assert at_equality.k_pred == 2
    assert at_equality.S == pytest.approx(1.3)


@pytest.mark.parametrize("bad_task", [None, "mean_auc", "foscttm"])
def test_summary_d1_d2_use_both_tasks_with_lower_foscttm_better(summary_inputs, bad_task):
    args = summary_inputs(bad_task=bad_task)
    downstream.summary(args)
    with open(f"{args.out}/summary.json") as handle:
        result = json.load(handle)
    expected = {"auc": -1 if bad_task == "mean_auc" else 1,
                "foscttm": -1 if bad_task == "foscttm" else 1}
    assert set(result["D1"]) == set(ENCODERS)
    assert set(result["D2"]) == {"200", "400", "800", "1600"}
    for criterion in ("D1", "D2"):
        for correlations in result[criterion].values():
            assert correlations == pytest.approx(expected)
        assert result[f"{criterion}_pass"] is (bad_task is None)


def test_d1_requires_every_encoder_to_improve(summary_inputs):
    args = summary_inputs(reverse_one_encoder=True)
    downstream.summary(args)
    with open(f"{args.out}/summary.json") as handle:
        result = json.load(handle)
    assert result["D1"][ENCODERS[0]]["foscttm"] == pytest.approx(-1)
    assert result["D1_pass"] is False


@pytest.mark.parametrize("reverse_counts,expected", [((1600,), True), ((800, 1600), False)])
def test_d2_uses_median_of_the_four_eligible_pair_counts(summary_inputs, reverse_counts, expected):
    args = summary_inputs(reverse_counts=reverse_counts)
    downstream.summary(args)
    with open(f"{args.out}/summary.json") as handle:
        result = json.load(handle)
    assert result["D2_pass"] is expected


@pytest.mark.parametrize("bad_task", [None, "mean_auc", "foscttm"])
@pytest.mark.parametrize("pair_sign,strength_sign,expected", [(1, 1, True), (-1, 1, False), (1, -1, False)])
def test_d3_signs_and_independence_from_raw_auc_and_retrieval(summary_inputs, bad_task,
                                                          pair_sign, strength_sign, expected):
    pair_ranks = tuple(pair_sign * rank for rank in range(1, 8))
    strength_ranks = tuple(strength_sign * rank for rank in range(1, 6))
    args = summary_inputs(bad_task=bad_task, auc_resid=d3_scores(pair_ranks, strength_ranks))
    downstream.summary(args)
    with open(f"{args.out}/summary.json") as handle:
        result = json.load(handle)
    assert_d3_summary(result, {e: pair_sign for e in ENCODERS},
                      {str(n): strength_sign for n in (200, 400, 800, 1600)}, expected)


@pytest.mark.parametrize("pair_ranks,strength_ranks,pair_rho,strength_rho,expected", [
    ((3, 2, 1, 6, 5, 4, 7), (1, 2, 3, 4, 5), 5 / 7, 1, True),
    ((4, 2, 3, 1, 5, 6, 7), (1, 2, 3, 4, 5), 19 / 28, 1, False),
    ((1, 2, 3, 4, 5, 6, 7), (2, 1, 4, 5, 3), 1, 0.6, True),
    ((1, 2, 3, 4, 5, 6, 7), (2, 3, 4, 1, 5), 1, 0.4, False),
])
def test_d3_uses_point_seven_and_point_five_thresholds(summary_inputs, pair_ranks, strength_ranks,
                                                     pair_rho, strength_rho, expected):
    assert spearmanr(range(1, 8), pair_ranks).statistic == pytest.approx(pair_rho)
    assert spearmanr(range(1, 6), strength_ranks).statistic == pytest.approx(strength_rho)
    args = summary_inputs(auc_resid=d3_scores(pair_ranks, strength_ranks))
    downstream.summary(args)
    with open(f"{args.out}/summary.json") as handle:
        result = json.load(handle)
    assert_d3_summary(result, {e: pair_rho for e in ENCODERS},
                      {str(n): strength_rho for n in (200, 400, 800, 1600)}, expected)


def test_d3_thresholds_include_equality(summary_inputs, monkeypatch):
    # Five positive counts and three encoders give exact rank correlations of 0.7 and 0.5.
    ns = (0, 25, 200, 400, 800, 1600)
    encoders = ENCODERS[:3]
    monkeypatch.setattr(downstream, "NS", ns)
    monkeypatch.setattr(downstream, "ENCODERS", encoders)
    scores = d3_scores((1, 2, 4, 5, 3), (1, 3, 2), ns=ns, encoders=encoders)
    args = summary_inputs(ns=ns, encoders=encoders, auc_resid=scores)
    downstream.summary(args)
    with open(f"{args.out}/summary.json") as handle:
        result = json.load(handle)
    assert_d3_summary(result, {e: 0.7 for e in encoders},
                      {str(n): 0.5 for n in (200, 400, 800, 1600)}, True)


@pytest.mark.parametrize("reversed_encoders,expected", [(2, True), (3, False)])
def test_d3_pair_criterion_uses_median_across_encoders(summary_inputs, reversed_encoders, expected):
    signs = [-1 if i < reversed_encoders else 1 for i in range(len(ENCODERS))]
    scores = {
        encoder: {0: 0.99, **{
            n: 0.5 + 0.04 * i + 0.001 * signs[i] * j
            for j, n in enumerate(PAIR_COUNTS[1:], start=1)
        }}
        for i, encoder in enumerate(ENCODERS)
    }
    args = summary_inputs(auc_resid=scores)
    downstream.summary(args)
    with open(f"{args.out}/summary.json") as handle:
        result = json.load(handle)
    assert_d3_summary(result, dict(zip(ENCODERS, signs)),
                      {str(n): 1 for n in (200, 400, 800, 1600)}, expected)


@pytest.mark.parametrize("reverse_counts,expected", [((1600,), True), ((800, 1600), False)])
def test_d3_strength_criterion_uses_median_of_only_eligible_counts(summary_inputs, reverse_counts, expected):
    good_ranks, bad_ranks = (2, 1, 4, 5, 3), (5, 4, 3, 2, 1)
    scores = {
        encoder: {0: 0.99, **{
            n: 0.5 + 0.02 * j + 0.001 * (bad_ranks[i] if n <= 128 or n in reverse_counts else good_ranks[i])
            for j, n in enumerate(PAIR_COUNTS[1:], start=1)
        }}
        for i, encoder in enumerate(ENCODERS)
    }
    args = summary_inputs(auc_resid=scores)
    downstream.summary(args)
    with open(f"{args.out}/summary.json") as handle:
        result = json.load(handle)
    expected_strength = {str(n): -1 if n in reverse_counts else 0.6 for n in (200, 400, 800, 1600)}
    assert_d3_summary(result, {e: 1 for e in ENCODERS}, expected_strength, expected)


def test_summary_allows_no_detectable_shared_components(summary_inputs):
    spectra = {e: [0.6 + i * 0.01, 0.3] for i, e in enumerate(ENCODERS)}
    spectra[ENCODERS[0]] = []
    args = summary_inputs(spectra=spectra)
    downstream.summary(args)
    curves = pd.read_csv(f"{args.out}/curves.csv")
    null_encoder = curves[curves.encoder == ENCODERS[0]]
    assert len(null_encoder) == 8
    assert null_encoder.k_pred.eq(0).all()
    assert null_encoder.S.eq(0).all()
