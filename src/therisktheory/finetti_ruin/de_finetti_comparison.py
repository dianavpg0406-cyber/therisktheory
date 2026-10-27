"""Comparative analysis and applications between classical risk theory and De Finetti's framework."""

from __future__ import annotations
from typing import Union
import numpy as np
import numpy.typing as npt

def elementary_insurance_case(
    initial_capital: Union[float, int, npt.ArrayLike],
    premium: Union[float, int, npt.ArrayLike],
    claim_prob: Union[float, int, npt.ArrayLike]
) -> npt.NDArray[np.float64]:
    """Evaluates an elementary insurance case under De Finetti's framework.

    Args:
        initial_capital: Starting reserve or capital amount(s).
        premium: Periodic premium collected.
        claim_prob: Subjective or objective probability of claim occurrence in [0, 1].

    Returns:
        Expected terminal position as a float64 NumPy array.
    """
    try:
        cap_arr = np.asarray(initial_capital, dtype=np.float64)
        prem_arr = np.asarray(premium, dtype=np.float64)
        prob_arr = np.asarray(claim_prob, dtype=np.float64)
    except (ValueError, TypeError) as err:
        raise TypeError("Inputs must be real numbers or numeric arrays.") from err

    if not np.all(np.isfinite(cap_arr)) or not np.all(np.isfinite(prem_arr)) or not np.all(np.isfinite(prob_arr)):
        raise ValueError("Inputs must contain only finite values.")

    if np.any(cap_arr < 0) or np.any(prem_arr < 0):
        raise ValueError("Initial capital and premium must be non-negative.")

    if np.any(prob_arr < 0.0) or np.any(prob_arr > 1.0):
        raise ValueError("Claim probability must be bounded between 0.0 and 1.0.")

    try:
        return cap_arr + prem_arr - (prob_arr * prem_arr)
    except OverflowError as err:
        raise OverflowError("Numerical overflow occurred in elementary_insurance_case.") from err


def compare_classical_de_finetti(
    classical_ruin_prob: Union[float, int, npt.ArrayLike],
    de_finetti_ruin_prob: Union[float, int, npt.ArrayLike]
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

    try:
        return np.abs(c_arr - df_arr)
    except OverflowError as err:
        raise OverflowError("Numerical overflow occurred in compare_classical_de_finetti.") from err
