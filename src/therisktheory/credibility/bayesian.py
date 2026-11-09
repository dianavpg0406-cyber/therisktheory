"""
Bayesian credibility models and actuarial premium estimation.

This module implements exact Bayesian credibility formulas using conjugate priors
(specifically the Poisson-Gamma model for claim frequencies) following actuarial theory principles.
"""

import numpy as np
from scipy import stats
from typing import Union


def bayesian_credibility_premium(
    alpha_prior: float,
    beta_prior: float,
    sample_sums: Union[int, float, np.ndarray],
    sample_sizes: Union[int, float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate the exact Bayesian credibility premium using a Gamma-Poisson conjugate model.

    Assuming a Gamma prior for the risk parameter lambda with shape parameter alpha 
    and scale/rate parameter beta, and Poisson-distributed claims per exposure unit, 
    the posterior distribution is also Gamma with updated parameters. The Bayesian 
    credibility premium is the posterior expectation of the risk rate.

    Args:
        alpha_prior (float): Prior shape parameter (alpha > 0).
        beta_prior (float): Prior scale/rate parameter (beta > 0).
        sample_sums (Union[int, float, np.ndarray]): Sum of observed claims or counts across exposures.
        sample_sizes (Union[int, float, np.ndarray]): Exposure units or number of periods observed.

    Returns:
        Union[float, np.ndarray]: The posterior expected claim rate (Bayesian premium per unit exposure).

    Raises:
        ValueError: If alpha_prior <= 0, beta_prior <= 0, or if sample sizes are non-positive.
    """
    if alpha_prior <= 0 or beta_prior <= 0:
        raise ValueError("Prior parameters alpha and beta must be strictly positive.")
    
    sums_arr = np.asarray(sample_sums, dtype=float)
    sizes_arr = np.asarray(sample_sizes, dtype=float)

    if np.any(sizes_arr <= 0):
        raise ValueError("Sample sizes/exposure units must be strictly positive.")

    # Posterior parameters for Gamma-Poisson conjugacy:
    # alpha_post = alpha_prior + sum(claims)
    # beta_post = beta_prior + sum(exposures)
    alpha_post = alpha_prior + sums_arr
    beta_post = beta_prior + sizes_arr

    # Posterior expectation E[lambda | data] = alpha_post / beta_post
    return alpha_post / beta_post


def posterior_predictive_mean(
    alpha_prior: float,
    beta_prior: float,
    sample_sums: Union[int, float, np.ndarray],
    sample_sizes: Union[int, float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate the posterior predictive mean for future observations in a Gamma-Poisson model.

    The posterior predictive distribution represents the expected value of future claims 
    given the observed data, incorporating both parameter uncertainty and process variance.
    Under the Gamma-Poisson model, the posterior predictive mean per unit exposure 
    equals the posterior mean of the parameter lambda.

    Args:
        alpha_prior (float): Prior shape parameter (alpha > 0).
        beta_prior (float): Prior scale/rate parameter (beta > 0).
        sample_sums (Union[int, float, np.ndarray]): Sum of observed claims.
        sample_sizes (Union[int, float, np.ndarray]): Exposure units observed.

    Returns:
        Union[float, np.ndarray]: The posterior predictive mean per exposure unit.

    Raises:
        ValueError: If parameters are outside valid domains.
    """
    return bayesian_credibility_premium(alpha_prior, beta_prior, sample_sums, sample_sizes)
