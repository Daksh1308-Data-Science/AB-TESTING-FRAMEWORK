"""Multiple-testing correction across variant-vs-control comparisons.

Wraps ``statsmodels.stats.multitest.multipletests`` with a uniform API.
Bonferroni controls the family-wise error rate (conservative); Benjamini-
Hochberg (``fdr_bh``) controls the false discovery rate and is the usual
default when comparing several variants against one control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

_METHODS = {"bonferroni", "fdr_bh"}


def correct_pvalues(p_values, method: str = "fdr_bh", alpha: float = 0.05) -> dict:
    """Correct a list of p-values (one per variant-vs-control comparison).

    Returns a dict with ``p_values``, ``p_corrected``, ``reject`` (bool array
    at the given alpha), ``alpha``, and ``method``.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {sorted(_METHODS)}")
    pvals = np.asarray(p_values, dtype=float)
    if np.any((pvals < 0) | (pvals > 1)):
        raise ValueError("p-values must be in [0, 1]")
    reject, p_corrected, _, _ = multipletests(pvals, alpha=alpha, method=method)
    return {
        "p_values": pvals,
        "p_corrected": p_corrected,
        "reject": reject,
        "alpha": alpha,
        "method": method,
    }


def decide_variants(z_results, method: str = "fdr_bh", alpha: float = 0.05) -> pd.DataFrame:
    """Turn per-variant analysis dicts into a corrected decision table.

    ``z_results`` is a list of dicts with at least ``p_value``; ideally
    produced by ``frequentist.variant_result`` (keys ``variant``,
    ``absolute_lift``, ``relative_lift``, ``ci_low``, ``ci_high`` are used
    when present).

    Returns a DataFrame with one row per variant and columns ``variant``,
    ``p_value``, ``p_corrected``, ``reject``, ``absolute_lift``,
    ``relative_lift``, ``ci_low``, ``ci_high``.
    """
    pvals = [float(row["p_value"]) for row in z_results]
    corr = correct_pvalues(pvals, method=method, alpha=alpha)
    rows = []
    for i, row in enumerate(z_results):
        label = row.get("variant") or f"V{i + 1}"
        rows.append(
            {
                "variant": label,
                "p_value": float(row["p_value"]),
                "p_corrected": float(corr["p_corrected"][i]),
                "reject": bool(corr["reject"][i]),
                "absolute_lift": row.get("absolute_lift"),
                "relative_lift": row.get("relative_lift"),
                "ci_low": row.get("ci_low"),
                "ci_high": row.get("ci_high"),
            }
        )
    return pd.DataFrame(rows, columns=list(rows[0].keys()) if rows else None)