"""Bayesian A/B testing with a Beta-Bernoulli model.

Posterior draws for the conversion rate of each arm. With the default uniform
Beta(1, 1) prior, the Beta-Bernoulli posterior is exactly
Beta(x + 1, n - x + 1), so Monte Carlo draws from it are exact. The optional
PyMC path (``engine="pymc"`` / ``"auto"``) demonstrates MCMC inference over
the same model; both engines produce the same posterior.
"""

from __future__ import annotations

import numpy as np


def _resolve_engine(engine: str) -> str:
    if engine == "closed_form":
        return "closed_form"
    if engine == "pymc":
        try:
            import pymc  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError("engine='pymc' requested but PyMC is not installed") from exc
        return "pymc"
    # engine == "auto"
    try:
        import pymc  # noqa: F401
        return "pymc"
    except Exception:
        return "closed_form"


def _beta_conjugate(successes: float, total: float, n_samples: int, seed: int) -> np.ndarray:
    """Draws from Beta(successes + 1, total - successes + 1)."""
    rng = np.random.default_rng(seed)
    return rng.beta(successes + 1.0, total - successes + 1.0, size=n_samples)


def _posterior_closed_form(control_successes, control_total, vs, vt, n_samples, seed):
    control = _beta_conjugate(control_successes, control_total, n_samples, seed)
    if len(vs):
        variants = np.array([_beta_conjugate(s, t, n_samples, seed) for s, t in zip(vs, vt)])
    else:
        variants = np.empty((0, n_samples))
    return {
        "control": control,
        "variants": variants,
        "variant_labels": [f"V{i + 1}" for i in range(len(vs))],
        "engine": "closed_form",
        "n_samples": n_samples,
        "note": "exact Beta-Bernoulli posterior (Beta(1,1) prior), closed form",
    }


def _posterior_pymc(control_successes, control_total, vs, vt, n_samples, seed):
    import pymc as pm

    draws_per_chain = max(1, int(np.ceil(n_samples / 2)))
    with pm.Model():
        p_c = pm.Beta("control", alpha=1, beta=1)
        pm.Binomial("obs_c", n=int(control_total), p=p_c, observed=int(control_successes))
        p_vars = []
        for i, (s, t) in enumerate(zip(vs, vt)):
            pv = pm.Beta(f"variant_{i}", alpha=1, beta=1)
            pm.Binomial(f"obs_v_{i}", n=int(t), p=pv, observed=int(s))
            p_vars.append(pv)
        trace = pm.sample(
            draws=draws_per_chain,
            chains=2,
            cores=1,
            tune=1000,
            progressbar=False,
            random_seed=seed,
            step=pm.Metropolis(),
            compute_convergence_checks=False,
        )
    control = np.asarray(trace.posterior["control"].to_numpy()).ravel()[:n_samples]
    variants = np.array(
        [np.asarray(trace.posterior[f"variant_{i}"].to_numpy()).ravel()[:n_samples] for i in range(len(vs))]
    )
    return {
        "control": control,
        "variants": variants,
        "variant_labels": [f"V{i + 1}" for i in range(len(vs))],
        "engine": "pymc",
        "n_samples": len(control),
        "note": "PyMC MCMC (Metropolis), Beta(1,1) prior",
    }


def posterior_samples(
    control_successes,
    control_total,
    variant_successes=None,
    variant_total=None,
    n_samples: int = 20_000,
    engine: str = "auto",
    seed: int = 42,
) -> dict:
    """Draw posterior samples for the control and optional variant arms.

    Parameters
    ----------
    control_successes, control_total : scalars for the control arm.
    variant_successes, variant_total : scalars or iterables, one per variant.
        Omit (or pass ``None``) for control-only analysis.
    n_samples : number of posterior draws per arm.
    engine : ``"auto"`` (PyMC if importable, else closed-form Beta),
        ``"closed_form"``, or ``"pymc"``.
    seed : int, for reproducibility.

    Returns
    -------
    dict with ``control`` (ndarray), ``variants`` (ndarray, shape
    ``(n_variants, n_samples)``), ``variant_labels``, ``engine``,
    ``n_samples``, ``note``.
    """
    if control_total <= 0:
        raise ValueError("control_total must be > 0")
    if not 0 <= float(control_successes) <= float(control_total):
        raise ValueError("control_successes must be between 0 and control_total")

    n_samples = int(n_samples)
    seed = int(seed)
    engine = _resolve_engine(engine)

    if variant_successes is None:
        vs, vt = np.array([], dtype=float), np.array([], dtype=float)
    else:
        vs = np.atleast_1d(np.asarray(variant_successes, dtype=float)).ravel()
        vt = np.atleast_1d(np.asarray(variant_total, dtype=float)).ravel()
        if len(vs) != len(vt):
            raise ValueError("variant_successes and variant_total must be the same length")
        if np.any((vs < 0) | (vs > vt)) or np.any(vt <= 0):
            raise ValueError("each variant's successes must be between 0 and its total (>0)")

    if engine == "pymc":
        return _posterior_pymc(control_successes, control_total, vs, vt, n_samples, seed)
    return _posterior_closed_form(control_successes, control_total, vs, vt, n_samples, seed)


def _stack_arms(results: dict) -> tuple:
    control = np.asarray(results["control"])
    variants = np.asarray(results["variants"])
    labels = ["control"] + list(
        results.get("variant_labels") or [f"V{i + 1}" for i in range(variants.shape[0])]
    )
    if variants.size:
        stack = np.vstack([control[None, :], variants])
    else:
        stack = control[None, :]
    return stack, labels


def prob_best(results: dict) -> dict:
    """Probability that each arm is the best, via Monte Carlo over draws.

    ``results`` is the dict returned by :func:`posterior_samples`. Returns
    ``{label: probability}`` summing to 1.
    """
    stack, labels = _stack_arms(results)
    best = np.argmax(stack, axis=0)
    counts = np.bincount(best, minlength=stack.shape[0])
    return dict(zip(labels, (counts / stack.shape[1]).tolist()))


def expected_loss(results: dict) -> dict:
    """Expected loss (regret) of choosing each arm, in conversion-rate units.

    For each arm i: ``E[max_j p_j - p_i]`` estimated over posterior draws —
    the conversion-rate cost (per user) of committing to arm i instead of the
    true best arm.
    """
    stack, labels = _stack_arms(results)
    best = stack.max(axis=0)
    return {label: float(np.mean(best - stack[i])) for i, label in enumerate(labels)}


def choose_variant(losses: dict, control_label: str = "control", cost_ratio: float = 1.0) -> dict:
    """Decide which arm to ship by minimizing expected loss.

    ``cost_ratio`` (> 1) penalizes shipping any variant, modeling
    implementation cost vs. staying with control (it scales variant losses).
    Returns ``{"chosen": label, "scaled_losses": {...}, "rule": ...}``.
    """
    if cost_ratio < 0:
        raise ValueError("cost_ratio must be >= 0")
    scaled = {
        label: (loss * cost_ratio if label != control_label else loss)
        for label, loss in losses.items()
    }
    chosen = min(scaled, key=scaled.get)
    return {
        "chosen": chosen,
        "scaled_losses": scaled,
        "rule": "minimize expected loss (incl. implementation cost)",
    }