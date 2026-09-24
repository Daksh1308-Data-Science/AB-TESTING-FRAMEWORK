import numpy as np
import pytest

from abtest.sequential import _p_beta_greater, sequential_result


def test_p_beta_greater_matches_monte_carlo():
    exact = _p_beta_greater(5, 95, 8, 92)
    rng = np.random.default_rng(0)
    mc = np.mean(rng.beta(5, 95, size=200_000) > rng.beta(8, 92, size=200_000))
    assert exact == pytest.approx(mc, abs=3e-3)


def test_sequential_stops_when_variant_better():
    sizes = np.arange(50, 1001, 50)
    x_c = np.round(0.02 * sizes).astype(int)
    x_v = np.round(0.15 * sizes).astype(int)
    res = sequential_result(np.vstack([x_c, x_v]), np.vstack([sizes, sizes]), stop_prob_best=0.95)
    assert res["stopped"] is True
    assert res["stopped_decision"] == "variant"
    assert res["obs_at_stop"] < 1000
    assert res["p_best_at_stop"] >= 0.95
    assert res["trajectory"].iloc[-1]["p_variant_best"] >= 0.95


def test_sequential_tied_does_not_stop():
    sizes = np.arange(50, 1001, 50)
    x = np.round(0.05 * sizes).astype(int)
    res = sequential_result(np.vstack([x, x]), np.vstack([sizes, sizes]), stop_prob_best=0.95)
    assert res["stopped"] is False
    assert res["trajectory"].iloc[-1]["p_control_best"] == pytest.approx(0.5, abs=0.01)


def test_sequential_max_obs():
    sizes = np.arange(50, 1001, 50)
    x_c = np.round(0.02 * sizes).astype(int)
    x_v = np.round(0.15 * sizes).astype(int)
    res = sequential_result(
        np.vstack([x_c, x_v]), np.vstack([sizes, sizes]), stop_prob_best=0.95, max_obs=400
    )
    assert res["obs_at_stop"] <= 400
    assert res["trajectory"]["n_per_arm"].max() <= 400


def test_exact_and_mc_agree():
    sizes = np.arange(100, 701, 100)
    x_c = np.round(0.03 * sizes).astype(int)
    x_v = np.round(0.09 * sizes).astype(int)
    e = sequential_result(np.vstack([x_c, x_v]), np.vstack([sizes, sizes]), pbest_method="exact")
    m = sequential_result(
        np.vstack([x_c, x_v]), np.vstack([sizes, sizes]),
        pbest_method="mc", n_pbest_samples=20_000, seed=1,
    )
    assert m["trajectory"].iloc[-1]["p_variant_best"] == pytest.approx(
        e["trajectory"].iloc[-1]["p_variant_best"], abs=0.02
    )