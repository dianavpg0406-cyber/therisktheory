"""Unit tests and mathematical validation tests for collective risk safety equation module,
integrating Adrián García's severity models (PR #10)[cite: 1]."""
import pytest
import numpy as np
from therisktheory.collective_risk import (
    adjustment_coefficient,
    lundberg_equation,
    full_retention_adjustment,
)
from therisktheory.severity_aggregate import lognormal_severity_moments

def test_adjustment_coefficient_with_garcia_lognormal():
    """Test full retention approximation using García's lognormal severity moments from PR #10[cite: 1]."""
    moments = lognormal_severity_moments(mu=1.0, sigma=0.5)
    mean_x = float(moments["mean"])
    second_moment_x = float(moments["second_raw_moment"])
    safety_loading = 0.2
    approx_r = full_retention_adjustment((mean_x, second_moment_x), safety_loading)
    assert approx_r > 0.0

def test_adjustment_coefficient_exponential_analytical() -> None:
    """Test adjustment coefficient for Exponential severity against exact analytical solution."""
    mu = 2.0
    mean_x = 1.0 / mu
    safety_loading = 0.2

    def mgf_exp(r: float) -> float:
        if r >= mu:
            return float("inf")
        return mu / (mu - r)

    expected_r = (safety_loading * mu) / (1.0 + safety_loading)
    computed_r = adjustment_coefficient(mgf_exp, mean_x, safety_loading, r_max=mu - 1e-3)
    assert np.isclose(computed_r, expected_r, rtol=1e-5)

def test_lundberg_equation_at_root() -> None:
    """Verify that Lundberg equation residual equals zero at computed root."""
    mu = 1.0
    mean_x = 1.0
    safety_loading = 0.3

    def mgf_exp(r: float) -> float:
        return mu / (mu - r)

    r_sol = adjustment_coefficient(mgf_exp, mean_x, safety_loading, r_max=0.99)
    residual = lundberg_equation(r_sol, mgf_exp, mean_x, safety_loading)
    assert np.isclose(residual, 0.0, atol=1e-7)
