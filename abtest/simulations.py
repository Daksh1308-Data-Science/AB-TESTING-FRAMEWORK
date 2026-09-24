"""Simulations that MEASURE the framework's properties.

Every headline number the project claims — the inflated false-positive rate
from peeking, the controlled rate of the planned design, the realized power of
the recommended sample size, the effect of multiple-testing correction, and
the time saved by Bayesian early stopping — is produced here, never hardcoded.
All simulations are seeded (default 42) and return plain dicts.
"""

from __future__ import annotations

import numpy as np

from .frequentist import z_test_two_proportion
from .power import min_sample_size
from .multiple_testing import correct_pvalues
from .sequential import sequential_result


def simulate_peeking(
    cr: float = 0.05,
    n_per_arm: int = 5000,
    peek_every: int = 100,
    alpha: float = 0.05,
    n_trials: int = 500,
    seed: int = 42,
) -> dict:
    """False-positive rate of peeking vs. a fixed design, under the null.

    Runs ``n_trials`` true-null two-arm tests. The *peeking* scenario re-tests
    after every ``peek_every`` users per arm and declares significance the
    first time p < alpha; the *fixed* scenario tests once at the full sample.
    """
    rng = np.random.default_rng(seed)
    peeks = np.arange(peek_every, n_per_arm + 1, peek_every)
    n_peeks = len(peeks)
    fp_peek = 0
    fp_fixed = 0
    for _ in range(int(n_trials)):
        x_c = int(rng.binomial(int(peeks[-1]), cr))
        x_v = int(rng.binomial(int(peeks[-1]), cr))
        if z_test_two_proportion(x_c, int(peeks[-1]), x_v, int(peeks[-1])).p_value < alpha:
            fp_fixed += 1
        for n in peeks:
            n = int(n)
            x_c = int(rng.binomial(n, cr))
            x_v = int(rng.binomial(n, cr))
            if z_test_two_proportion(x_c, n, x_v, n).p_value < alpha:
                fp_peek += 1
                break
    return {
        "peeking_false_positive_rate": fp_peek / n_trials,
        "fixed_false_positive_rate": fp_fixed / n_trials,
        "n_trials": int(n_trials),
        "n_per_arm": int(n_per_arm),
        "peek_every": int(peek_every),
        "n_peeks": int(n_peeks),
        "alpha": alpha,
        "cr": cr,
    }


def simulate_power(
    base_rate: float = 0.05,
    mde: float = 0.01,
    alpha: float = 0.05,
    power_target: float = 0.8,
    n_trials: int = 500,
    seed: int = 42,
) -> dict:
    """Measured power of the framework's recommended sample size.

    Computes ``n_arm = min_sample_size(base_rate, mde, alpha, power_target)``
    (the normal approximation), then measures the actual rejection rate at
    that sample size under the alternative — the empirical power.
    """
    n_arm = min_sample_size(base_rate, mde, alpha, power_target)
    rng = np.random.default_rng(seed)
    true_rate = base_rate + mde
    hits = 0
    for _ in range(int(n_trials)):
        x_c = int(rng.binomial(n_arm, base_rate))
        x_v = int(rng.binomial(n_arm, true_rate))
        if z_test_two_proportion(x_c, n_arm, x_v, n_arm).p_value < alpha:
            hits += 1
    return {
        "recommended_n_per_arm": n_arm,
        "measured_power": hits / n_trials,
        "power_target": power_target,
        "base_rate": base_rate,
        "mde": mde,
        "alpha": alpha,
        "n_trials": int(n_trials),
    }


def simulate_multiple_testing(
    k: int = 20,
    n_per_arm: int = 2000,
    cr: float = 0.05,
    alpha: float = 0.05,
    n_trials: int = 200,
    seed: int = 42,
) -> dict:
    """False positives across k true-null variants: raw vs. corrections.

    Each trial: one control and ``k`` variants, all at the true rate ``cr``.
    Reports the mean per-experiment false-positive count/rate for raw p-values,
    Bonferroni, and Benjamini-Hochberg, plus the rate of *at least one* false
    positive (the "any" rate) per scenario.
    """
    rng = np.random.default_rng(seed)
    raw_total = bonf_total = bh_total = 0
    raw_any = bonf_any = bh_any = 0
    for _ in range(int(n_trials)):
        x_c = int(rng.binomial(n_per_arm, cr))
        pvals = []
        for _ in range(int(k)):
            x_v = int(rng.binomial(n_per_arm, cr))
            pvals.append(z_test_two_proportion(x_c, n_per_arm, x_v, n_per_arm).p_value)
        raw_flags = [p < alpha for p in pvals]
        bonf = correct_pvalues(pvals, method="bonferroni", alpha=alpha)["reject"]
        bh = correct_pvalues(pvals, method="fdr_bh", alpha=alpha)["reject"]
        raw_total += sum(raw_flags)
        bonf_total += int(bonf.sum())
        bh_total += int(bh.sum())
        raw_any += any(raw_flags)
        bonf_any += bool(bonf.any())
        bh_any += bool(bh.any())
    n_trials = int(n_trials)
    k = int(k)
    return {
        "raw_false_positive_rate": raw_total / (n_trials * k),
        "bonferroni_false_positive_rate": bonf_total / (n_trials * k),
        "fdr_bh_false_positive_rate": bh_total / (n_trials * k),
        "raw_any_false_positive_rate": raw_any / n_trials,
        "bonferroni_any_false_positive_rate": bonf_any / n_trials,
        "fdr_bh_any_false_positive_rate": bh_any / n_trials,
        "k": k,
        "n_per_arm": int(n_per_arm),
        "alpha": alpha,
        "n_trials": n_trials,
    }


def simulate_time_to_decision(
    base_rate: float = 0.05,
    mde: float = 0.01,
    alpha: float = 0.05,
    power_target: float = 0.8,
    stop_prob_best: float = 0.95,
    check_every=None,
    n_pbest_samples: int = 6000,
    n_trials: int = 150,
    seed: int = 42,
    max_obs=None,
) -> dict:
    """Time to decision: fixed-sample vs. Bayesian sequential.

    Fixed-sample observations to decision = the planned per-arm sample size.
    Sequential: monitor at checkpoints and stop when P(best) reaches
    ``stop_prob_best``; trials that never stop are assigned the full planned
    sample. ``pct_faster = 100 * (1 - sequential / fixed)``.
    """
    n_arm = min_sample_size(base_rate, mde, alpha, power_target)
    if max_obs is not None:
        n_arm = min(n_arm, int(max_obs))
    check_every = check_every or max(25, n_arm // 40)
    sizes = np.arange(check_every, n_arm + 1, check_every)
    if sizes[-1] != n_arm:
        sizes = np.append(sizes, n_arm)

    rng = np.random.default_rng(seed)
    true_rate = base_rate + mde
    obs_to_decision = []
    for _ in range(int(n_trials)):
        x_c = rng.binomial(sizes, base_rate)
        x_v = rng.binomial(sizes, true_rate)
        res = sequential_result(
            np.vstack([x_c, x_v]),
            np.vstack([sizes, sizes]),
            stop_prob_best=stop_prob_best,
            pbest_method="mc",
            n_pbest_samples=n_pbest_samples,
            seed=seed,
        )
        obs_to_decision.append(res["obs_at_stop"] if res["stopped"] else n_arm)
    obs_to_decision = np.asarray(obs_to_decision, dtype=float)
    fixed = float(sizes[-1])
    seq_mean = float(obs_to_decision.mean())
    return {
        "fixed_observations_to_decision": fixed,
        "sequential_observations_to_decision": seq_mean,
        "pct_faster": (1.0 - seq_mean / fixed) * 100.0,
        "stopped_early_fraction": float(np.mean(obs_to_decision < fixed)),
        "stop_prob_best": stop_prob_best,
        "base_rate": base_rate,
        "mde": mde,
        "n_arm_horizon": float(sizes[-1]),
        "n_trials": int(n_trials),
    }