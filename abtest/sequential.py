"""Bayesian sequential monitoring for two-arm A/B tests.

Tracks P(control best) and P(variant best) at each checkpoint as data arrives,
using the exact Beta-Bernoulli posterior with uniform priors. A test may stop
early once either arm's probability of being best reaches ``stop_prob_best`` —
Bayesian updating is always valid, so the *stopping rule* (not the data
peeking) is what is controlled here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import betaln


def _p_beta_greater(a1: float, b1: float, a2: float, b2: float) -> float:
    """Exact P(X > Y) for X ~ Beta(a1, b1), Y ~ Beta(a2, b2).

    Uses the classic partial-sum identity (computed in log space for
    stability)::

        P(X > Y) = sum_{j=a2}^{a2+b2-1} B(a1+j, b1+b2)
                                       / ((b2+j) * B(1+j, b2) * B(a1, b1))
    """
    total = 0.0
    lB_a1_b1 = betaln(a1, b1)
    for j in range(int(a2), int(a2 + b2)):
        total += np.exp(
            betaln(a1 + j, b1 + b2) - np.log(b2 + j) - betaln(1 + j, b2) - lB_a1_b1
        )
    return float(min(1.0, max(0.0, total)))


def _posterior_p_greater(x_c, n_c, x_v, n_v, method: str, n_samples: int, seed: int):
    """P(control best), P(variant best) given cumulative counts."""
    if method == "exact":
        p_control_best = _p_beta_greater(x_c + 1, n_c - x_c + 1, x_v + 1, n_v - x_v + 1)
        return p_control_best, 1.0 - p_control_best
    if method != "mc":
        raise ValueError("pbest_method must be 'exact' or 'mc'")
    rng = np.random.default_rng(seed)
    dr_c = rng.beta(x_c + 1, n_c - x_c + 1, size=n_samples)
    dr_v = rng.beta(x_v + 1, n_v - x_v + 1, size=n_samples)
    p_control_best = float(np.mean(dr_c > dr_v))
    return p_control_best, 1.0 - p_control_best


def sequential_result(
    successes,
    totals,
    stop_prob_best: float = 0.95,
    max_obs=None,
    pbest_method: str = "exact",
    n_pbest_samples: int = 8000,
    seed: int = 42,
) -> dict:
    """Bayesian sequential monitoring of a two-arm test.

    Parameters
    ----------
    successes : array-like, shape (2, T)
        Cumulative successes per checkpoint; row 0 = control, row 1 = variant.
    totals : array-like, shape (2, T)
        Cumulative users per checkpoint (per arm).
    stop_prob_best : float
        Stop when max(P(control best), P(variant best)) >= this.
    max_obs : int or None
        Only consider checkpoints up to this many users per arm.
    pbest_method : ``"exact"`` (partial-sum formula) or ``"mc"`` (Monte Carlo
        draws; faster for long trajectories, slightly noisy).
    n_pbest_samples : draws per checkpoint when ``pbest_method="mc"``.

    Returns
    -------
    dict with ``stopped`` (bool), ``obs_at_stop`` (users per arm at stop or
    None), ``p_best_at_stop``, ``stopped_decision`` (``"control"`` /
    ``"variant"`` / None), ``stop_prob_best``, and ``trajectory`` (DataFrame:
    step, n_per_arm, control_successes, variant_successes, p_control_best,
    p_variant_best).
    """
    s = np.asarray(successes, dtype=float)
    t = np.asarray(totals, dtype=float)
    if s.ndim != 2 or t.ndim != 2 or s.shape[0] != 2 or t.shape[0] != 2 or s.shape[1] != t.shape[1]:
        raise ValueError("successes and totals must both have shape (2, T)")
    if not (0 < stop_prob_best < 1):
        raise ValueError("stop_prob_best must be in (0, 1)")

    if max_obs is not None:
        keep = np.where(t[1] <= float(max_obs))[0]
        if len(keep) == 0:
            raise ValueError("max_obs is smaller than the first checkpoint")
        s, t = s[:, keep], t[:, keep]

    trajectory = []
    stopped = False
    obs_at_stop, p_best_at_stop, stopped_decision = None, None, None
    for i in range(s.shape[1]):
        n_c, n_v = float(t[0, i]), float(t[1, i])
        x_c, x_v = float(s[0, i]), float(s[1, i])
        p_c_best, p_v_best = _posterior_p_greater(
            x_c, n_c, x_v, n_v, pbest_method, int(n_pbest_samples), seed + i
        )
        trajectory.append(
            {
                "step": i + 1,
                "n_per_arm": int(n_c),
                "control_successes": int(x_c),
                "variant_successes": int(x_v),
                "p_control_best": p_c_best,
                "p_variant_best": p_v_best,
            }
        )
        if not stopped and max(p_c_best, p_v_best) >= stop_prob_best:
            stopped = True
            obs_at_stop = int(n_c)
            p_best_at_stop = max(p_c_best, p_v_best)
            stopped_decision = "control" if p_c_best >= p_v_best else "variant"

    return {
        "stopped": stopped,
        "obs_at_stop": obs_at_stop,
        "p_best_at_stop": p_best_at_stop,
        "stopped_decision": stopped_decision,
        "stop_prob_best": stop_prob_best,
        "trajectory": pd.DataFrame(trajectory),
    }