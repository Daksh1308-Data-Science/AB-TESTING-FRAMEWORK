"""Real-world A/B test data: loading, cleaning, and sanity checks.

The canonical Udacity e-commerce A/B experiment (also on Kaggle as "A/B
testing"): a real test of a new landing page vs. the old one, with binary
conversion (0/1 = paid / did not pay). Released columns:

    user_id, timestamp, group (control|treatment),
    landing_page (old_page|new_page), converted (0|1)

The dataset has genuine data-quality quirks — ~3.9k users experienced *both*
pages and there are group/landing_page mismatches — which makes it a good
real-world stress test for the framework's sanity checks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = _REPO_ROOT / "data" / "ab_data.csv"
DEFAULT_COUNTRIES_PATH = _REPO_ROOT / "data" / "countries.csv"

_EXPECTED_COLUMNS = {"user_id", "timestamp", "group", "landing_page", "converted"}


def load_ab_data(path=None) -> pd.DataFrame:
    """Load the raw ``ab_data.csv`` and parse timestamps.

    Raises ``FileNotFoundError`` if the file is missing and ``ValueError`` if
    the expected schema is not present.
    """
    path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"ab_data.csv not found at {path}")
    df = pd.read_csv(path)
    missing = _EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"missing expected columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def clean_ab_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop users who experienced both pages; keep one row per user.

    The standard Udacity cleaning: any user with more than one row saw both
    the old and new page, which contaminates the group assignment — drop them.
    On the released data this also removes every group/landing_page mismatch.
    Rows are returned sorted chronologically (by timestamp, then user_id) so
    the sequential monitor can replay the experiment in arrival order.
    """
    counts = df["user_id"].value_counts()
    dup_users = counts[counts > 1].index
    clean = df[~df["user_id"].isin(dup_users)].copy()
    clean = clean.sort_values(["timestamp", "user_id"]).reset_index(drop=True)
    return clean


def load_countries(path=None) -> pd.DataFrame:
    """Load ``countries.csv`` (user_id, country) — optional feature data."""
    path = Path(path) if path is not None else DEFAULT_COUNTRIES_PATH
    if not path.exists():
        raise FileNotFoundError(f"countries.csv not found at {path}")
    df = pd.read_csv(path)
    if not {"user_id", "country"} <= set(df.columns):
        raise ValueError("countries.csv must have user_id and country columns")
    return df


def srm_check(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Sample Ratio Mismatch check: is the control share consistent with 50/50?

    Sanity check that randomization actually split users evenly. A failing
    SRM means the contrast may be confounded (e.g. caches, time-zone or
    device artifacts) before any analysis can be trusted.

    Returns a dict with ``control_share``, ``ci_low``/``ci_high`` (Wilson-free
    normal CI on a mean), ``n``, ``expected_share`` (0.5), ``passes``, and a
    ``verdict`` string.
    """
    n = int(len(df))
    if n == 0:
        raise ValueError("empty data")
    x = int((df["group"] == "control").sum())
    share = x / n
    z = float(stats.norm.ppf(1 - alpha / 2))
    se = float(np.sqrt(share * (1 - share) / n))
    ci_low, ci_high = share - z * se, share + z * se
    passes = ci_low <= 0.5 <= ci_high
    return {
        "control_share": share,
        "n": n,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "expected_share": 0.5,
        "passes": passes,
        "verdict": "OK (50/50 split consistent)" if passes else "FAILING (sample ratio mismatch)",
    }


def chronological_split(df: pd.DataFrame, checkpoints: int = 100) -> dict:
    """Cumulative per-arm counts along arrival order, for sequential monitoring.

    ``df`` is expected to be cleaned (one row per user) but is sorted by
    timestamp + user_id here regardless. Returns a dict with ``successes`` and
    ``totals`` arrays, each of shape ``(2, checkpoints)`` — row 0 = control,
    row 1 = treatment — ready for ``sequential.sequential_result``.
    """
    checkpoints = int(checkpoints)
    if checkpoints < 2:
        raise ValueError("checkpoints must be >= 2")
    ordered = df.sort_values(["timestamp", "user_id"]).reset_index(drop=True)
    n = len(ordered)
    c_mask = (ordered["group"] == "control").to_numpy()
    t_mask = (ordered["group"] == "treatment").to_numpy()
    converted = (ordered["converted"] == 1).to_numpy()

    pos = np.linspace(0, n - 1, num=checkpoints).astype(int)
    c_succ = (c_mask & converted).cumsum()
    t_succ = (t_mask & converted).cumsum()
    c_tot = c_mask.cumsum()
    t_tot = t_mask.cumsum()
    return {
        "successes": np.vstack([c_succ[pos], t_succ[pos]]),
        "totals": np.vstack([c_tot[pos], t_tot[pos]]),
        "n": int(n),
        "checkpoints": checkpoints,
    }