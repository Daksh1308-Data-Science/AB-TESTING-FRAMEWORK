import pytest
import pandas as pd

from abtest.data_generator import generate_conversion_data


def test_shape_and_columns():
    df = generate_conversion_data(1000, 2, 0.05, [0.06, 0.04])
    assert list(df.columns) == ["user_id", "group", "converted"]
    assert set(df["group"].unique()) == {"control", "V1", "V2"}
    assert len(df) == 3000
    assert df["group"].value_counts()["control"] == 1000
    assert df["converted"].isin([0, 1]).all()


def test_seeded_reproducible():
    a = generate_conversion_data(500, 1, 0.05, 0.07, seed=7)
    b = generate_conversion_data(500, 1, 0.05, 0.07, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_scalar_variant_rate_accepted():
    df = generate_conversion_data(100, 1, 0.05, 0.06, seed=1)
    assert set(df["group"].unique()) == {"control", "V1"}


def test_rates_close_to_truth():
    df = generate_conversion_data(20000, 1, 0.05, 0.07, seed=1)
    rates = df.groupby("group")["converted"].mean()
    assert abs(rates["control"] - 0.05) < 0.01
    assert abs(rates["V1"] - 0.07) < 0.01


def test_validation():
    with pytest.raises(ValueError):
        generate_conversion_data(100, 2, 0.05, [0.06])  # wrong number of variant rates
    with pytest.raises(ValueError):
        generate_conversion_data(0, 1, 0.05, 0.06)
    with pytest.raises(ValueError):
        generate_conversion_data(100, 1, 1.5, 0.06)