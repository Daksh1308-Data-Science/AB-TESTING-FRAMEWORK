"""Frequentist two-proportion tests and confidence intervals.

Thin, typed wrappers over statsmodels/scipy so callers get a consistent API.
The variant is always group 1 and the control group 2, so a positive z with
``alternative="larger"`` means the variant converted better.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

_ALTERNATIVES = {"two-sided", "larger", "smaller"}


@dataclass
class ZTestResult:
    """Result of a two-proportion z-test."""

    z: float
    p_value: float
    alternative: str
    method: str = "two-proportion z-test (statsmodels, pooled variance)"


def z_test_two_proportion(
    control_successes,
    control_total,
    variant_successes,
    variant_total,
    alternative: str = "two-sided",
) -> ZTestResult:
    """Two-proportion z-test, variant vs. control.

    Parameters are scalars (ints or floats). ``alternative`` is one of
    ``"two-sided"``, ``"larger"`` (variant > control), ``"smaller"``
    (variant < control).
    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {sorted(_ALTERNATIVES)}")
    if control_total <= 0 or variant_total <= 0:
        raise ValueError("totals must be > 0")
    count = np.array([variant_successes, control_successes], dtype=float)
    nobs = np.array([variant_total, control_total], dtype=float)
    if np.any(count > nobs) or np.any(count < 0):
        raise ValueError("successes must be between 0 and total")

    z, p = proportions_ztest(count, nobs, alternative=alternative)
    return ZTestResult(z=float(z), p_value=float(p), alternative=alternative)


def confidence_interval_lift(
    control_successes,
    control_total,
    variant_successes,
    variant_total,
    alpha: float = 0.05,
) -> dict:
    """Rates, lift, and confidence intervals for a variant vs. control.

    Returns a plain dict with:
    ``control_rate``, ``variant_rate``, ``absolute_lift`` (variant - control),
    ``relative_lift`` (absolute lift / control rate), Wilson CIs per arm
    (``control_ci_low/high``, ``variant_ci_low/high``), a normal-approximation
    CI on the absolute lift (``ci_low`` / ``ci_high``), and metadata.
    """
    n_c, n_v = float(control_total), float(variant_total)
    x_c, x_v = float(control_successes), float(variant_successes)
    if n_c <= 0 or n_v <= 0:
        raise ValueError("totals must be > 0")
    p_c, p_v = x_c / n_c, x_v / n_v

    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    c_lo, c_hi = proportion_confint(int(x_c), int(n_c), alpha=alpha, method="wilson")
    v_lo, v_hi = proportion_confint(int(x_v), int(n_v), alpha=alpha, method="wilson")
    se_lift = float(np.sqrt(p_c * (1 - p_c) / n_c + p_v * (1 - p_v) / n_v))

    return {
        "control_rate": p_c,
        "variant_rate": p_v,
        "absolute_lift": p_v - p_c,
        "relative_lift": (p_v - p_c) / p_c if p_c > 0 else np.nan,
        "control_ci_low": float(c_lo),
        "control_ci_high": float(c_hi),
        "variant_ci_low": float(v_lo),
        "variant_ci_high": float(v_hi),
        "ci_low": (p_v - p_c) - z_crit * se_lift,
        "ci_high": (p_v - p_c) + z_crit * se_lift,
        "alpha": alpha,
        "method": "Wilson CI per arm; normal-approx CI on absolute lift",
    }


def variant_result(
    control_successes,
    control_total,
    variant_successes,
    variant_total,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    variant=None,
) -> dict:
    """Combined z-test + CI result for one variant, as a plain dict.

    Convenience for feeding ``multiple_testing.decide_variants`` and the
    dashboard. ``variant`` is a label such as ``"V1"``.
    """
    zr = z_test_two_proportion(
        control_successes, control_total, variant_successes, variant_total, alternative=alternative
    )
    ci = confidence_interval_lift(
        control_successes, control_total, variant_successes, variant_total, alpha=alpha
    )
    out = {"variant": variant or "V1", "z": zr.z, "p_value": zr.p_value, "alternative": alternative}
    out.update(ci)
    return out