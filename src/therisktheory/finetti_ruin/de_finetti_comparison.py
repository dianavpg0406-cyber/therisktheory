"""Comparative analysis between classical risk theory and De Finetti's framework."""

from __future__ import annotations
import numpy as np
import numpy.typing as npt


def elementary_insurance_case(
    initial_capital: float | int | npt.ArrayLike,
    premium: float | int | npt.ArrayLike,
    claim_prob: float | int | npt.ArrayLike,
    claim_amount: float | int | npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Evaluates an elementary insurance case under De Finetti's framework.

    Args:
        initial_capital: Starting reserve or capital amount(s).
        premium: Periodic premium collected.
        claim_prob: Probability of claim occurrence in [0, 1].
        claim_amount: Contractual sum insured or claim payout.

    Returns:
        Expected terminal position as a float64 NumPy array:
        E[U_1] = U_0 + P - (p * S)
    """
    try:
        cap_arr = np.asarray(initial_capital, dtype=np.float64)
        prem_arr = np.asarray(premium, dtype=np.float64)
        prob_arr = np.asarray(claim_prob, dtype=np.float64)
        claim_arr = np.asarray(claim_amount, dtype=np.float64)
    except (ValueError, TypeError) as err:
        raise TypeError("Inputs must be real numbers or numeric arrays.") from err

    if (
        not np.all(np.isfinite(cap_arr))
        or not np.all(np.isfinite(prem_arr))
        or not np.all(np.isfinite(prob_arr))
        or not np.all(np.isfinite(claim_arr))
    ):
        raise ValueError("Inputs must contain only finite values.")

    if np.any(cap_arr < 0) or np.any(prem_arr < 0) or np.any(claim_arr < 0):
        raise ValueError("Initial capital, premium, and claim amount must be non-negative.")

    if np.any(prob_arr < 0.0) or np.any(prob_arr > 1.0):
        raise ValueError("Claim probability must be bounded between 0.0 and 1.0.")

    return np.asarray(cap_arr + prem_arr - (prob_arr * claim_arr), dtype=np.float64)


def compare_classical_de_finetti(
    classical_ruin_prob: float | int | npt.ArrayLike,
    de_finetti_ruin_prob: float | int | npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Computes absolute deviations between classical and De Finetti ruin estimates.

    Args:
        classical_ruin_prob: Ruin probabilities from classical Cramér-Lundberg models.
        de_finetti_ruin_prob: Ruin probabilities from De Finetti's discrete game models.

    Returns:
        Array of absolute differences for model comparison and criticism.
    """
    try:
        c_arr = np.asarray(classical_ruin_prob, dtype=np.float64)
        df_arr = np.asarray(de_finetti_ruin_prob, dtype=np.float64)
    except (ValueError, TypeError) as err:
        raise TypeError("Probabilities must be real numbers or numeric arrays.") from err

    if not np.all(np.isfinite(c_arr)) or not np.all(np.isfinite(df_arr)):
        raise ValueError("Probabilities must contain only finite values.")

    if np.any(c_arr < 0.0) or np.any(c_arr > 1.0) or np.any(df_arr < 0.0) or np.any(df_arr > 1.0):
        raise ValueError("Probabilities must be bounded between 0.0 and 1.0.")

    return np.abs(c_arr - df_arr)
