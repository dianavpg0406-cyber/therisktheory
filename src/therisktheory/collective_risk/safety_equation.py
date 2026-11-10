"""Safety equation and Lundberg fundamental equation models in collective risk theory.

This module implements the calculation of the adjustment coefficient R (Lundberg's exponent),
evaluates Lundberg's fundamental equation, and assesses the safety equation under full retention,
integrating severity moments from García's models (PR #10)[cite: 1] and frequency parameters.

Reference:
    Bühlmann, H. (1970). Mathematical Methods in Risk Theory. Springer.
    Beard, R., Pentikäinen, T., & Pesonen, E. (1984). Risk Theory. Chapman and Hall.
"""
from __future__ import annotations
import numpy as np
from scipy import optimize
from typing import Callable

def lundberg_equation(
    r: float,
    mgf_severity: Callable[[float], float],
    mean_severity: float,
    safety_loading: float,
) -> float:
    """Evaluate Lundberg's fundamental equation residual for a given adjustment coefficient R.

    Lundberg's equation for a Poisson collective risk model with safety loading lambda is:
        M_X(r) - 1 - (1 + lambda) * E[X] * r = 0

    Args:
        r (float): Trial value for the adjustment coefficient (within the convergence domain where M_X(r) < infinity).
        mgf_severity (Callable[[float], float]): Moment generating function M_X(r) of claim severity.
        mean_severity (float): Expected value E[X] of claim severity (> 0).
        safety_loading (float): Relative safety loading lambda (> 0).

    Returns:
        float: The residual value of the Lundberg equation.
    """
    return mgf_severity(r) - 1.0 - (1.0 + safety_loading) * mean_severity * r

def adjustment_coefficient(
    mgf_severity: Callable[[float], float],
    mean_severity: float,
    safety_loading: float,
    r_max: float = 5.0,
) -> float:
    """Calculate the adjustment coefficient R (Lundberg exponent) by solving Lundberg's equation.

    The adjustment coefficient R is the unique positive root of Lundberg's fundamental equation
    within the domain of existence of the MGF (where M_X(r) < infinity and r < r_0).

    Args:
        mgf_severity (Callable[[float], float]): Function returning M_X(r) for severity X.
        mean_severity (float): Expected claim severity E[X] (> 0).
        safety_loading (float): Safety loading factor lambda (> 0).
        r_max (float, optional): Upper bound search limit for root finding within the MGF domain of convergence. Defaults to 5.0.

    Returns:
        float: The unique positive adjustment coefficient R.

    Raises:
        ValueError: If mean_severity <= 0 or safety_loading <= 0.
        RuntimeError: If root finding fails to converge within the specified range.
    """
    if mean_severity <= 0:
        raise ValueError("mean_severity must be strictly positive.")
    if safety_loading <= 0:
        raise ValueError("safety_loading must be strictly positive.")

    def obj(r: float) -> float:
        return lundberg_equation(r, mgf_severity, mean_severity, safety_loading)

    try:
        # Bracket search interval [1e-6, r_max] within the domain of convergence where M_X(r) exists.
        sol = optimize.root_scalar(obj, bracket=[1e-6, r_max], method="brentq")
        if not sol.converged:
            raise RuntimeError("Root finding algorithm did not converge.")
        return float(sol.root)
    except Exception as err:
        raise RuntimeError(f"Failed to find adjustment coefficient: {err}") from err

def full_retention_adjustment(
    claim_moments: tuple[float, float],
    safety_loading: float,
) -> float:
    """Calculate the introductory linear approximation of the adjustment coefficient under full retention.

    Using a Taylor expansion of the MGF (small R approximation):
        R ≈ (2 * lambda * E[X]) / E[X^2]

    Args:
        claim_moments (tuple[float, float]): Tuple (E[X], E[X^2]) representing first and second moments of severity.
        safety_loading (float): Safety loading factor lambda (> 0).

    Returns:
        float: Approximate adjustment coefficient R under full retention.
    """
    mean_x, second_moment_x = claim_moments
    if mean_x <= 0 or second_moment_x <= 0:
        raise ValueError("Claim moments E[X] and E[X^2] must be strictly positive.")
    if second_moment_x < mean_x**2:
        raise ValueError("Invalid moments: E[X^2] cannot be strictly less than E[X]^2.")
    if safety_loading <= 0:
        raise ValueError("safety_loading must be strictly positive.")
    return (2.0 * safety_loading * mean_x) / second_moment_x
