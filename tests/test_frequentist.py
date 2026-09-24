import pytest

from abtest.frequentist import z_test_two_proportion, confidence_interval_lift, variant_result


def test_z_test_hand_case():
    # control 10/100, variant 20/100: pooled z = (0.2-0.1)/sqrt(0.15*0.85*0.02)
    r = z_test_two_proportion(10, 100, 20, 100)
    assert r.z == pytest.approx(1.980295, abs=1e-4)
    assert r.p_value == pytest.approx(0.0476704, abs=1e-4)
    assert "z-test" in r.method


def test_z_test_direction_and_alternatives():
    r2 = z_test_two_proportion(30, 500, 40, 500)  # variant (40) > control (30)
    rl = z_test_two_proportion(30, 500, 40, 500, alternative="larger")
    rs = z_test_two_proportion(40, 500, 30, 500, alternative="larger")
    assert r2.z == pytest.approx(1.239394, abs=1e-4)
    assert rl.p_value == pytest.approx(r2.p_value / 2, abs=1e-6)  # one-sided = two/2 when z>0
    assert rs.p_value > 0.5  # variant is worse


def test_z_test_validation():
    with pytest.raises(ValueError):
        z_test_two_proportion(10, 100, 20, 100, alternative="nonsense")
    with pytest.raises(ValueError):
        z_test_two_proportion(100, 10, 20, 100)  # successes > total


def test_confidence_interval_lift():
    ci = confidence_interval_lift(50, 1000, 60, 1000)
    assert ci["absolute_lift"] == pytest.approx(0.01)
    assert ci["relative_lift"] == pytest.approx(0.2)
    assert ci["ci_low"] < 0.01 < ci["ci_high"]
    assert 0 <= ci["control_ci_low"] < ci["control_ci_high"] <= 1
    assert 0 <= ci["variant_ci_low"] < ci["variant_ci_high"] <= 1


def test_variant_result_keys():
    r = variant_result(50, 1000, 60, 1000, variant="V1")
    for key in [
        "variant", "z", "p_value", "control_rate", "variant_rate",
        "absolute_lift", "relative_lift", "ci_low", "ci_high",
    ]:
        assert key in r
    assert r["variant"] == "V1"