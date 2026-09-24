import pytest

from abtest.power import min_sample_size, power_curve, expected_duration


def test_min_sample_size_statsmodels_example():
    # statsmodels: proportion_effectsize(0.1, 0.15) under NormalIndPower at
    # power=0.8, alpha=0.05 gives n per arm ~= 681 (measured on statsmodels 0.14).
    assert min_sample_size(0.1, 0.05) == pytest.approx(681, abs=5)


def test_min_sample_size_monotonic():
    assert min_sample_size(0.05, 0.02) >= min_sample_size(0.05, 0.05)


def test_power_curve():
    df = power_curve(0.10, 0.05, 3000)
    assert list(df.columns) == ["n", "power"]
    assert df["n"].max() == 3000
    assert 0 <= df["power"].min() and df["power"].max() <= 1
    n_req = min_sample_size(0.10, 0.05)
    at_required = df[df["n"] <= n_req].iloc[-1]
    assert at_required["power"] == pytest.approx(0.8, abs=0.05)


def test_expected_duration():
    assert expected_duration(1000, 200) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        expected_duration(100, 0)