"""Known-value tests for the real Udacity e-commerce A/B dataset.

Numerical expectations are pinned against the released data file
(data/ab_data.csv), which is committed to the repo.
"""

import numpy as np
import pandas as pd
import pytest

from abtest.real_data import (
    load_ab_data,
    clean_ab_data,
    srm_check,
    chronological_split,
    load_countries,
)
from abtest.frequentist import z_test_two_proportion
from abtest.bayesian import posterior_samples, prob_best, expected_loss, choose_variant

RAW_ROWS = 294_478
UNIQUE_USERS = 290_584
CLEAN_ROWS = 286_690  # 290,584 unique users minus 3,894 who saw both pages


@pytest.fixture(scope="module")
def clean_df():
    return clean_ab_data(load_ab_data())


def test_load_ab_data_shape():
    df = load_ab_data()
    assert df.shape == (RAW_ROWS, 5)
    assert set(df.columns) == {"user_id", "timestamp", "group", "landing_page", "converted"}
    assert df.isna().sum().sum() == 0
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_unique_users_and_multi_page_users():
    df = load_ab_data()
    assert df["user_id"].nunique() == UNIQUE_USERS
    counts = df["user_id"].value_counts()
    assert (counts > 1).sum() == 3894


def test_clean_removes_multi_page_users():
    clean = clean_ab_data(load_ab_data())
    assert len(clean) == CLEAN_ROWS
    assert clean["user_id"].is_unique
    # no group/landing_page mismatch remains after cleaning
    bad = ((clean["group"] == "control") & (clean["landing_page"] == "new_page")) | (
        (clean["group"] == "treatment") & (clean["landing_page"] == "old_page")
    )
    assert bad.sum() == 0
    # chronological ordering for sequential replay
    assert clean["timestamp"].is_monotonic_increasing


def test_srm_check_passes(clean_df):
    srm = srm_check(clean_df)
    assert srm["passes"] is True
    assert srm["control_share"] == pytest.approx(0.49980, abs=1e-3)
    assert srm["ci_low"] <= 0.5 <= srm["ci_high"]
    assert srm["n"] == CLEAN_ROWS


def test_observed_conversion_rates_known(clean_df):
    cr = clean_df.groupby("group")["converted"].mean()
    assert cr["control"] == pytest.approx(0.120173, abs=1e-4)
    assert cr["treatment"] == pytest.approx(0.118726, abs=1e-4)


def test_null_result_on_real_data(clean_df):
    xc = int(clean_df[clean_df.group == "control"].converted.sum())
    nc = int((clean_df.group == "control").sum())
    xt = int(clean_df[clean_df.group == "treatment"].converted.sum())
    nt = int((clean_df.group == "treatment").sum())
    zr = z_test_two_proportion(xc, nc, xt, nt)
    assert zr.z == pytest.approx(-1.1945, abs=1e-2)
    assert zr.p_value == pytest.approx(0.2323, abs=1e-2)

    res = posterior_samples(xc, nc, [xt], [nt], n_samples=20_000, seed=1)
    pb = prob_best(res)
    assert pb["control"] > 0.999  # huge n → posterior is essentially certain
    loss = expected_loss(res)
    assert loss["control"] < loss["V1"]
    assert choose_variant(loss)["chosen"] == "control"


def test_chronological_split_endpoints(clean_df):
    out = chronological_split(clean_df, checkpoints=50)
    assert out["successes"].shape == (2, 50)
    assert out["totals"].shape == (2, 50)
    assert np.all(np.diff(out["totals"][0]) >= 0)
    assert out["totals"][0][-1] == 143_293
    assert out["totals"][1][-1] == 143_397
    assert out["successes"][0][-1] == 17_220
    assert out["successes"][1][-1] == 17_025


def test_countries_join():
    ct = load_countries()
    assert ct.shape == (UNIQUE_USERS, 2)
    assert set(ct.columns) == {"user_id", "country"}
    assert ct["user_id"].is_unique


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_ab_data("does/not/exist.csv")