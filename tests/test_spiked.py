"""Contract tests for squared canonical correlations in the finite-rank model."""

from math import sqrt

import numpy as np
import pytest

import align.spiked as spiked


ASPECT_RATIOS = [(0.1, 0.1), (0.2, 0.3), (0.45, 0.05), (0.49, 0.5)]


@pytest.mark.parametrize(
    "c1,c2,expected",
    [(0.1, 0.1, 1 / 9), (0.2, 0.2, 0.25),
     (0.1, 0.2, 1 / 6), (0.25, 0.5, 1 / sqrt(3))],
)
def test_rc_matches_formula(c1, c2, expected):
    assert spiked.rc(c1, c2) == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize(
    "c1,c2,expected",
    [(0.1, 0.1, 0.36), (0.2, 0.2, 0.64),
     (0.1, 0.2, 0.5), (0.25, 0.5, (2 + sqrt(3)) / 4)],
)
def test_edge_matches_formula(c1, c2, expected):
    assert spiked.edge(c1, c2) == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("function", [spiked.rc, spiked.edge], ids=["rc", "edge"])
@pytest.mark.parametrize(
    "c1,c2",
    [(0, 0.2), (0.2, 0), (-0.1, 0.2), (0.2, -0.1),
     (0.5, 0.5), (0.8, 0.3), (1, 0.1), (0.1, 1),
     (np.nan, 0.2), (0.2, np.nan), (np.inf, 0.2), (0.2, np.inf),
     (-np.inf, 0.2), (0.2, -np.inf)],
)
def test_rc_and_edge_reject_invalid_aspect_ratios(function, c1, c2):
    with pytest.raises(ValueError):
        function(c1, c2)


@pytest.mark.parametrize(
    "r,c1,c2",
    [(0.03, 0.1, 0.1), (0.3, 0.1, 0.1), (0.6, 0.1, 0.1),
     (0.8, 0.2, 0.3), (1, 0.6, 0.1), (1.2, 0.2, 0.3)],
)
def test_gamma_matches_formula(r, c1, c2):
    expected = (
        (1 - c1) * (1 - c2) * r
        + c1 * (1 - c2) + c2 * (1 - c1) + c1 * c2 / r
    )
    assert spiked.gamma(r, c1, c2) == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("c1,c2", ASPECT_RATIOS)
def test_gamma_meets_edge_at_threshold(c1, c2):
    actual = spiked.gamma(spiked.rc(c1, c2), c1, c2)
    assert actual == pytest.approx(spiked.edge(c1, c2), abs=1e-12, rel=0)


@pytest.mark.parametrize("c1,c2", ASPECT_RATIOS)
def test_gamma_is_increasing_above_threshold_and_ends_at_one(c1, c2):
    population = np.linspace(spiked.rc(c1, c2), 1, 41)
    predicted = np.array([spiked.gamma(r, c1, c2) for r in population])
    assert np.all(np.diff(predicted) > 0)
    assert predicted[-1] == pytest.approx(1, abs=1e-14, rel=0)


@pytest.mark.parametrize("c1,c2", ASPECT_RATIOS)
def test_invert_gamma_round_trips_outliers(c1, c2):
    threshold = spiked.rc(c1, c2)
    for fraction in [0.02, 0.2, 0.7, 1]:
        r = threshold + fraction * (1 - threshold)
        recovered = spiked.invert_gamma(spiked.gamma(r, c1, c2), c1, c2)
        assert threshold <= recovered <= 1
        assert recovered == pytest.approx(r, abs=1e-10, rel=0)


@pytest.mark.parametrize("c1,c2", [(0.1, 0.1), (0.2, 0.3)])
@pytest.mark.parametrize("location", ["edge", "below_edge", "zero", "negative", "above_one"])
def test_invert_gamma_rejects_non_outlier_values(c1, c2, location):
    right_edge = spiked.edge(c1, c2)
    lam = {
        "edge": right_edge,
        "below_edge": np.nextafter(right_edge, -np.inf),
        "zero": 0,
        "negative": -0.1,
        "above_one": np.nextafter(1.0, np.inf),
    }[location]
    with pytest.raises(ValueError):
        spiked.invert_gamma(lam, c1, c2)


def test_predict_sorts_outliers_and_pins_other_spikes_to_edge():
    population = np.array([0.05, 0.6, spiked.rc(0.1, 0.1), 0, 0.3, 1, 0.6])
    predicted = spiked.predict(population, 0.1, 0.1)
    expected = np.array([1, 256 / 375, 256 / 375, 1369 / 3000, 0.36, 0.36, 0.36])
    assert isinstance(predicted, np.ndarray)
    assert predicted.shape == population.shape
    np.testing.assert_allclose(predicted, expected, rtol=0, atol=1e-12)


def test_predict_empty_spikes():
    predicted = spiked.predict(np.array([]), 0.1, 0.2)
    assert isinstance(predicted, np.ndarray)
    assert predicted.shape == (0,)


@pytest.mark.parametrize("n,p,q", [(128, 5, 3), (128, 3, 5), (32, 1, 1), (18, 9, 8)])
def test_sample_cc2_matches_centered_covariance_eigenvalues(n, p, q):
    rng = np.random.default_rng(417)
    x = rng.standard_normal((n, p))
    y = x @ rng.standard_normal((p, q)) + rng.standard_normal((n, q))
    x = x + np.arange(1, p + 1) * 7
    y = y - np.arange(1, q + 1) * 11
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    sxx = xc.T @ xc / (n - 1)
    syy = yc.T @ yc / (n - 1)
    sxy = xc.T @ yc / (n - 1)
    eigenvalues = np.linalg.eigvals(np.linalg.solve(sxx, sxy) @ np.linalg.solve(syy, sxy.T))
    np.testing.assert_allclose(eigenvalues.imag, 0, rtol=0, atol=1e-12)
    expected = np.sort(eigenvalues.real)[::-1][:min(p, q)]

    actual = spiked.sample_cc2(x, y)

    assert isinstance(actual, np.ndarray)
    assert actual.shape == (min(p, q),)
    assert np.all(np.isfinite(actual))
    assert np.all((actual >= 0) & (actual <= 1))
    assert np.all(np.diff(actual) <= 0)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.fixture
def paired_samples():
    rng = np.random.default_rng(819)
    x = rng.standard_normal((160, 5))
    y = x @ rng.standard_normal((5, 3)) + rng.standard_normal((160, 3))
    return x, y


def test_sample_cc2_centers_each_column(paired_samples):
    x, y = paired_samples
    expected = spiked.sample_cc2(x, y)
    actual = spiked.sample_cc2(x + np.arange(5) * 13 + 5, y - np.arange(3) * 17 - 9)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("block", ["x", "y", "both"])
def test_sample_cc2_is_invariant_to_invertible_maps(paired_samples, block):
    x, y = paired_samples
    a = np.diag([0.5, 1, 1.5, 2, 3]) + np.diag(np.full(4, 0.4), 1)
    b = np.array([[2, 0.3, -0.5], [0, 0.7, 0.4], [0, 0, 1.5]])
    expected = spiked.sample_cc2(x, y)
    mapped_x = x @ a if block in ("x", "both") else x
    mapped_y = y @ b if block in ("y", "both") else y
    np.testing.assert_allclose(spiked.sample_cc2(mapped_x, mapped_y), expected, rtol=0, atol=1e-11)


@pytest.mark.parametrize("p,q", [(4, 3), (3, 5)])
def test_sample_cc2_detects_exact_linear_column(p, q):
    rng = np.random.default_rng(53)
    x = rng.standard_normal((100, p))
    y = rng.standard_normal((100, q))
    y[:, 0] = x @ np.arange(1, p + 1) + 17

    actual = spiked.sample_cc2(x, y)

    assert actual.shape == (min(p, q),)
    assert np.all((actual >= 0) & (actual <= 1))
    assert np.all(np.diff(actual) <= 0)
    assert actual[0] == pytest.approx(1, abs=1e-12, rel=0)


@pytest.mark.parametrize("r", [np.array([]), np.array([0.09, 0.81, 0.36])], ids=["null", "spiked"])
def test_simulate_has_specified_gaussian_covariance(r):
    n, p, q = 40_000, 5, 4
    x, y = spiked.simulate(r, p, q, n, np.random.default_rng(632))
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert x.shape == (n, p)
    assert y.shape == (n, q)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))

    joint = np.column_stack((x, y))
    expected = np.eye(p + q)
    indices = np.arange(len(r))
    expected[indices, p + indices] = np.sqrt(r)
    expected[p + indices, indices] = np.sqrt(r)
    np.testing.assert_allclose(np.cov(joint, rowvar=False), expected, rtol=0, atol=0.035)

    centered = joint - joint.mean(axis=0)
    standardized = centered / np.sqrt(np.mean(centered ** 2, axis=0))
    np.testing.assert_allclose(np.mean(standardized ** 3, axis=0), 0, rtol=0, atol=0.08)
    np.testing.assert_allclose(np.mean(standardized ** 4, axis=0), 3, rtol=0, atol=0.12)


def test_simulate_uses_supplied_rng_reproducibly():
    r = np.array([0.64, 0.16])
    first = spiked.simulate(r, 3, 4, 50, np.random.default_rng(71))
    replay = spiked.simulate(r, 3, 4, 50, np.random.default_rng(71))
    different = spiked.simulate(r, 3, 4, 50, np.random.default_rng(72))
    for original, repeated, changed in zip(first, replay, different):
        np.testing.assert_array_equal(original, repeated)
        assert not np.array_equal(original, changed)


def test_simulate_perfect_spike_matches_first_coordinate():
    x, y = spiked.simulate(np.array([1.0, 0.0]), 3, 2, 100, np.random.default_rng(941))
    np.testing.assert_allclose(x[:, 0] - x[:, 0].mean(), y[:, 0] - y[:, 0].mean(), rtol=0, atol=1e-12)


def test_simulation_mean_matches_theorem_1_8():
    r = np.array([0.6, 0.3, 0.05])
    rng = np.random.default_rng(170402408)
    leading = []
    for _ in range(5):
        x, y = spiked.simulate(r, 400, 400, 4000, rng)
        leading.append(spiked.sample_cc2(x, y)[:3])
    empirical = np.mean(leading, axis=0)
    predicted = spiked.predict(r, 0.1, 0.1)
    np.testing.assert_allclose(empirical, predicted, rtol=0, atol=0.02)


@pytest.mark.parametrize("r", [0.02, 0.1, 0.3, 0.8, 1.0])
@pytest.mark.parametrize("d", [1, 17, 400])
def test_budget_matches_boundary_and_makes_spike_detectable_above_it(r, d):
    boundary = spiked.budget(r, d)
    assert boundary == pytest.approx(d * (1 + 1 / r), rel=1e-13, abs=1e-12)
    n = boundary * (1 + 1e-6)
    assert spiked.rc(d / n, d / n) < r
