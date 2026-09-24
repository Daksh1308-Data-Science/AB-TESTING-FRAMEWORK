"""Power analysis and sample-size calculation.

Uses statsmodels' normal-approximation power analysis
(``NormalIndPower`` + ``proportion_effectsize``) for two-proportion tests with
balanced arms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

_ALTERNATIVES = {"two-sided", "larger", "smaller"}


def _effect_size(base_rate: float, mde: float) -> float:
    target = base_rate + mde
    if not (0 < base_rate < 1) or not (0 < target < 1):
        raise ValueError("base_rate and base_rate + mde must be strictly between 0 and 1")
    if mde <= 0:
        raise ValueError("mde must be > 0")
    return float(proportion_effectsize(base_rate, target))


def min_sample_size(
    base_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    alternative: str = "two-sided",
) -> int:
    """Per-arm sample size for a two-proportion test (balanced arms).

    ``mde`` is the minimum detectable effect as an *absolute* lift in
    conversion rate (e.g. ``0.01`` for +1 pp).
    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {sorted(_ALTERNATIVES)}")
    effect = _effect_size(base_rate, mde)
    n = NormalIndPower().solve_power(
        effect_size=effect, alpha=alpha, power=power, ratio=1.0, alternative=alternative
    )
    return int(np.ceil(n))


def power_curve(
    base_rate: float,
    mde: float,
    max_n: int,
    alpha: float = 0.05,
    power_target: float = 0.8,
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """Power as a function of per-arm sample size, on a log-spaced grid.

    Returns a DataFrame with columns ``n`` and ``power``. ``power_target`` is
    accepted for API compat with ``min_sample_size`` (informational only).
    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {sorted(_ALTERNATIVES)}")
    max_n = max(2, int(max_n))
    ns = np.unique(np.round(np.geomspace(1, max_n, min(300, max_n))).astype(int))
    ns = np.unique(np.append(ns, max_n))
    effect = _effect_size(base_rate, mde)
    powers = np.asarray(
        NormalIndPower().power(
            effect_size=effect, nobs1=ns, alpha=alpha, ratio=1.0, alternative=alternative
        ),
        dtype=float,
    )
    return pd.DataFrame({"n": ns, "power": powers})


def expected_duration(n_per_arm: int | float, daily_traffic_per_arm: int | float) -> float:
    """Expected test duration in days = per-arm sample size / daily traffic."""
    if daily_traffic_per_arm <= 0:
        raise ValueError("daily_traffic_per_arm must be > 0")
    return float(n_per_arm) / float(daily_traffic_per_arm)