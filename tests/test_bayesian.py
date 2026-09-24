import numpy as np
import pytest

from abtest.bayesian import posterior_samples, prob_best, expected_loss, choose_variant


def test_posterior_matches_conjugate_mean():
    res = posterior_samples(50, 1000, [60], [1000], n_samples=10000, engine="closed_form", seed=3)
    assert res["control"].mean() == pytest.approx(51 / 1002, abs=3e-4)
    assert res["variants"][0].mean() == pytest.approx(61 / 1002, abs=3e-4)
    assert res["engine"] == "closed_form"


def test_posterior_reproducible():
    a = posterior_samples(50, 1000, [60], [1000], n_samples=5000, seed=7)
    b = posterior_samples(50, 1000, [60], [1000], n_samples=5000, seed=7)
    np.testing.assert_array_equal(a["control"], b["control"])


def test_prob_best_strong_signal():
    res = posterior_samples(50, 10_000, [120], [10_000], n_samples=20_000, seed=5)
    pb = prob_best(res)
    assert pb["V1"] > 0.99
    assert sum(pb.values()) == pytest.approx(1.0)


def test_expected_loss_and_decision():
    res = posterior_samples(50, 10_000, [120], [10_000], n_samples=20_000, seed=6)
    loss = expected_loss(res)
    assert loss["V1"] >= 0
    assert loss["control"] > loss["V1"]
    assert choose_variant(loss)["chosen"] == "V1"


def test_control_only():
    res = posterior_samples(50, 1000, n_samples=5000, seed=1)
    assert prob_best(res)["control"] == pytest.approx(1.0)
    assert expected_loss(res)["control"] == 0.0


def test_available_engines():
    res = posterior_samples(50, 1000, n_samples=1000, engine="auto", seed=2)
    assert res["engine"] in {"pymc", "closed_form"}
    try:
        import pymc  # noqa: F401
        pymc_available = True
    except Exception:
        pymc_available = False
    if pymc_available:
        pm_res = posterior_samples(50, 1000, n_samples=1000, engine="pymc", seed=2)
        assert pm_res["control"].mean() == pytest.approx(51 / 1002, abs=1e-2)
    else:
        with pytest.raises(ImportError):
            posterior_samples(50, 1000, n_samples=1000, engine="pymc")