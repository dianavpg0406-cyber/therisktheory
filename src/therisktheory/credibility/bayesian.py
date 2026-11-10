"""Bayesian credibility models and actuarial premium estimation.

This module implements exact Bayesian credibility formulas using conjugate priors
(specifically the Poisson-Gamma model for claim frequencies) following actuarial theory principles.
"""
from __future__ import annotations
import numpy as np
from typing import Union
from numpy.typing import ArrayLike, NDArray

def bayesian_credibility_premium(
    alpha_prior: float,
    beta_prior: float,
    sample_sums: ArrayLike,
    sample_sizes: ArrayLike,
) -> float | NDArray[np.float64]:
    """Calculate the exact Bayesian credibility premium using a Gamma-Poisson conjugate model.

    Assuming a Gamma prior for the risk parameter lambda with shape parameter alpha
    and rate parameter beta (where beta acts as the scale denominator in exposure),
    and Poisson-distributed claims per exposure unit, the posterior distribution is
    also Gamma with updated parameters.

    Args:
        alpha_prior (float): Prior shape parameter (alpha > 0).
        beta_prior (float): Prior rate parameter (beta > 0).
        sample_sums (ArrayLike): Sum of observed claims or counts across exposures (must be >= 0 and finite).
        sample_sizes (ArrayLike): Exposure units or number of periods observed (must be > 0 and finite).

    Returns:
        float | NDArray[np.float64]: The posterior expected claim rate (Bayesian premium per unit exposure).

    Raises:
        ValueError: If parameters are non-positive, non-finite, or if sample sizes/claims violate domain rules.
        TypeError: If inputs cannot be converted to real numeric arrays.
    """
    if not isinstance(alpha_prior, (int, float, np.number)) or not isinstance(beta_prior, (int, float, np.number)):
        raise TypeError("Prior parameters alpha and beta must be numeric.")
    
    if alpha_prior <= 0 or beta_prior <= 0 or not np.isfinite(alpha_prior) or not np.isfinite(beta_prior):
        raise ValueError("Prior parameters alpha and beta must be strictly positive and finite.")

    try:
        sums_arr = np.asarray(sample_sums, dtype=np.float64)
        sizes_arr = np.asarray(sample_sizes, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError("Sample sums and sizes must be convertible to real numeric arrays.") from e

    if not np.all(np.isfinite(sums_arr)) or not np.all(np.isfinite(sizes_arr)):
        raise ValueError("Sample sums and exposure sizes must contain only finite real numbers.")

    if np.any(sums_arr < 0):
        raise ValueError("Observed claim counts / sample sums cannot be negative.")

    if np.any(sizes_arr <= 0):
        raise ValueError("Sample sizes / exposure units must be strictly positive.")

    # Posterior parameters for Gamma-Poisson conjugacy:
    # alpha_post = alpha_prior + sum(claims)
    # beta_post = beta_prior + sum(exposures)
    alpha_post = alpha_prior + sums_arr
    beta_post = beta_prior + sizes_arr

    result = alpha_post / beta_post
    return result if isinstance(sample_sums, (int, float, np.number)) and isinstance(sample_sizes, (int, float, np.number)) else np.asarray(result, dtype=np.float64)

def posterior_predictive_mean(
    alpha_prior: float,
    beta_prior: float,
    sample_sums: ArrayLike,
    sample_sizes: ArrayLike,
) -> float | NDArray[np.float64]:
    """Calculate the posterior predictive mean for future observations in a Gamma-Poisson model.

    Under the Gamma-Poisson conjugate model, the posterior predictive mean per unit exposure
    equals the posterior expectation of the parameter lambda.

    Args:
        alpha_prior (float): Prior shape parameter (alpha > 0).
        beta_prior (float): Prior rate parameter (beta > 0).
        sample_sums (ArrayLike): Sum of observed claims.
        sample_sizes (ArrayLike): Exposure units observed.

    Returns:
        float | NDArray[np.float64]: The posterior predictive mean per exposure unit.
    """
    return bayesian_credibility_premium(alpha_prior, beta_prior, sample_sums, sample_sizes)
