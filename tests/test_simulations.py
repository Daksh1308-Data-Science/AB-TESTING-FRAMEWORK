import pytest

from abtest.simulations import (
    simulate_peeking,
    simulate_power,
    simulate_multiple_testing,
    simulate_time_to_decision,
)


def test_peeking_inflates_false_positives():
    out = simulate_peeking(cr=0.05, n_per_arm=2000, peek_every=100, alpha=0.05, n_trials=300, seed=11)
    assert out["fixed_false_positive_rate"] == pytest.approx(0.05, abs=0.04)
    assert out["peeking_false_positive_rate"] > out["fixed_false_positive_rate"]
    assert out["peeking_false_positive_rate"] > 0.10


def test_power_measured_near_target():
    out = simulate_power(base_rate=0.05, mde=0.01, alpha=0.05, power_target=0.8, n_trials=300, seed=12)
    assert out["measured_power"] == pytest.approx(0.8, abs=0.1)
    assert out["recommended_n_per_arm"] > 0


def test_multiple_testing_reduces_false_positives():
    out = simulate_multiple_testing(k=20, n_per_arm=2000, cr=0.05, alpha=0.05, n_trials=100, seed=13)
    assert out["bonferroni_false_positive_rate"] <= out["raw_false_positive_rate"]
    assert out["fdr_bh_false_positive_rate"] <= out["raw_false_positive_rate"]
    assert out["raw_any_false_positive_rate"] > 0.2


def test_time_to_decision_faster_or_equal():
    out = simulate_time_to_decision(base_rate=0.05, mde=0.01, n_trials=40, seed=14)
    assert out["sequential_observations_to_decision"] <= out["fixed_observations_to_decision"]
    assert out["sequential_observations_to_decision"] > 0
    assert out["pct_faster"] >= 0


def test_all_sims_seeded_and_reproducible():
    a = simulate_peeking(n_trials=50, seed=99)
    b = simulate_peeking(n_trials=50, seed=99)
    assert a["peeking_false_positive_rate"] == b["peeking_false_positive_rate"]