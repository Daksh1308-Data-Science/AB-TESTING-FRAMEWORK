"""A/B testing framework for binary conversion metrics.

Frequentist (two-proportion z-test, power analysis, multiple-testing
correction), Bayesian (Beta-Bernoulli posterior, P(best), expected loss),
Bayesian sequential monitoring, and simulation-based measurement of the
framework's properties.
"""

__version__ = "0.1.0"

from .data_generator import generate_conversion_data
from .frequentist import ZTestResult, z_test_two_proportion, confidence_interval_lift, variant_result
from .power import min_sample_size, power_curve, expected_duration
from .multiple_testing import correct_pvalues, decide_variants
from .bayesian import posterior_samples, prob_best, expected_loss, choose_variant
from .sequential import sequential_result
from .simulations import (
    simulate_peeking,
    simulate_power,
    simulate_multiple_testing,
    simulate_time_to_decision,
)

__all__ = [
    "generate_conversion_data",
    "ZTestResult",
    "z_test_two_proportion",
    "confidence_interval_lift",
    "variant_result",
    "min_sample_size",
    "power_curve",
    "expected_duration",
    "correct_pvalues",
    "decide_variants",
    "posterior_samples",
    "prob_best",
    "expected_loss",
    "choose_variant",
    "sequential_result",
    "simulate_peeking",
    "simulate_power",
    "simulate_multiple_testing",
    "simulate_time_to_decision",
]