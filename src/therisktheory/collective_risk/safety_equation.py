"""
Safety equation and Lundberg fundamental equation models in collective risk theory.

This module implements the calculation of the adjustment coefficient R (Lundberg's exponent),
evaluates Lundberg's fundamental equation, and assesses the safety equation under full retention,
integrating severity moments from Garcia's models and frequency parameters from Bracamonte's processes.
"""

import numpy as np
from scipy import optimize
from typing import Callable, Union


def lundberg_equation(
    r: float,
    mgf_severity: Callable[[float], float],
    mean_severity: float,
    safety_loading: float,
) -> float:
    """
    Evaluate Lundberg's fundamental equation residual for a given adjustment coefficient R.

    Lundberg's equation for a Poisson collective risk model with safety loading lambda is given by:
        M_X(r) - 1 - (1 + lambda) * E[X] * r = 0

    Args:
        r (float): Trial value for the adjustment coefficient (r > 0).
        mgf_severity (Callable[[float], float]): Moment generating function M_X(r) of claim severity.
        mean_severity (float): Expected value E[X] of claim severity.
        safety_loading (float): Relative safety loading lambda (lambda > 0).

    Returns:
        float: The residual value of the Lundberg equation (should be 0 at the true adjustment coefficient).
    """
    return mgf_severity(r) - 1.0 - (1.0 + safety_loading) * mean_severity * r


def adjustment_coefficient(
    mgf_severity: Callable[[float], float],
    mean_severity: float,
    safety_loading: float,
    r_max: float = 5.0,
) -> float:
    """
    Calculate the adjustment coefficient R (Lundberg exponent) by solving Lundberg's equation.

    The adjustment coefficient R is the unique positive root of Lundberg's fundamental equation:
        M_X(R) - 1 - (1 + lambda) * E[X] * R = 0

    Args:
        mgf_severity (Callable[[float], float]): Function returning M_X(r) for severity X.
        mean_severity (float): Expected claim severity E[X] (> 0).
        safety_loading (float): Safety loading factor lambda (> 0).
        r_max (float, optional): Upper bound search limit for root finding. Defaults to 5.0.

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
    """
    Calculate the introductory linear approximation of the adjustment coefficient under full retention.

    Using a Taylor expansion of the MGF (small R approximation):
        R ≈ (2 * lambda * E[X]) / E[X^2]

    Args:
        claim_moments (tuple[float, float]): Tuple (E[X], E[X^2]) representing first and second moments of severity.
        safety_loading (float): Safety loading factor lambda (> 0).

    Returns:
        float: Approximate adjustment coefficient R under full retention.

    Raises:
        ValueError: If moments or safety loading are non-positive or mathematically invalid.
    """
    mean_x, second_moment_x = claim_moments

    if mean_x <= 0 or second_moment_x <= 0:
        raise ValueError("Claim moments E[X] and E[X^2] must be strictly positive.")
    if second_moment_x < mean_x**2:
        raise ValueError("Invalid moments: E[X^2] cannot be strictly less than E[X]^2.")
    if safety_loading <= 0:
        raise ValueError("safety_loading must be strictly positive.")

    return (2.0 * safety_loading * mean_x) / second_moment_x
