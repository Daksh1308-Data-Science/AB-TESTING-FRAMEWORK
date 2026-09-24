"""Synthetic conversion data with known ground-truth rates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_conversion_data(
    n_controls: int,
    n_variants: int,
    cr_control: float,
    cr_variants,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate a balanced, seeded conversion dataset.

    Parameters
    ----------
    n_controls : int
        Number of users per arm (control *and* every variant get this many).
    n_variants : int
        Number of variant arms (an A/B/n test).
    cr_control : float
        True control conversion rate, in [0, 1].
    cr_variants : float or iterable
        True conversion rate(s), one per variant (a scalar is accepted for a
        single variant).
    seed : int, optional
        RNG seed; defaults to 42 for reproducibility.

    Returns
    -------
    pandas.DataFrame with columns ``user_id``, ``group`` (``control`` /
    ``V1`` / ``V2`` / ...), ``converted`` (0/1). Rows are grouped by arm and
    ordered by arrival within each arm, so ``cumsum`` within a group gives the
    arrival-time conversion trajectory.
    """
    n_controls = int(n_controls)
    n_variants = int(n_variants)
    if n_controls <= 0:
        raise ValueError("n_controls must be > 0")
    if n_variants < 1:
        raise ValueError("n_variants must be >= 1")

    variant_rates = np.atleast_1d(np.asarray(cr_variants, dtype=float)).ravel()
    if len(variant_rates) != n_variants:
        raise ValueError(f"expected {n_variants} variant rate(s), got {len(variant_rates)}")

    groups = ["control"] + [f"V{i + 1}" for i in range(n_variants)]
    rates = np.concatenate([[float(cr_control)], variant_rates])
    if np.any((rates < 0) | (rates > 1)):
        raise ValueError("conversion rates must be in [0, 1]")

    rng = np.random.default_rng(seed)
    frames = []
    uid = 0
    for group, rate in zip(groups, rates):
        converted = (rng.random(n_controls) < rate).astype(np.int8)
        frame = pd.DataFrame(
            {
                "user_id": np.arange(uid, uid + n_controls, dtype=np.int64),
                "group": group,
                "converted": converted,
            }
        )
        frames.append(frame)
        uid += n_controls
    return pd.concat(frames, ignore_index=True)