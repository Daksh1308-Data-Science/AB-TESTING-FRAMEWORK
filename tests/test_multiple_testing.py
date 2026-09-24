import numpy as np
import pandas as pd
import pytest

from abtest.multiple_testing import correct_pvalues, decide_variants
from abtest.frequentist import variant_result


def test_bonferroni_documented_example():
    out = correct_pvalues([0.05, 0.05, 0.05], method="bonferroni")
    np.testing.assert_allclose(out["p_corrected"], [0.15, 0.15, 0.15])
    assert not out["reject"].any()
    assert out["method"] == "bonferroni"


def test_fdr_bh_documented_example():
    out = correct_pvalues([0.05, 0.05, 0.05], method="fdr_bh")
    np.testing.assert_allclose(out["p_corrected"], [0.05, 0.05, 0.05])
    assert out["reject"].all()


def test_decide_variants():
    rows = [
        variant_result(100, 5000, 130, 5000, variant="V1"),
        variant_result(100, 5000, 90, 5000, variant="V2"),
    ]
    df = decide_variants(rows, method="bonferroni", alpha=0.05)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [
        "variant", "p_value", "p_corrected", "reject",
        "absolute_lift", "relative_lift", "ci_low", "ci_high",
    ]
    assert len(df) == 2
    assert df.iloc[0]["variant"] == "V1"
    assert df["reject"].dtype == bool


def test_invalid_method():
    with pytest.raises(ValueError):
        correct_pvalues([0.05], method="holm")