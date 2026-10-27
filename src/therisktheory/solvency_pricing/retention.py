"""Risk retention, retention capacity, and inflation adjustment functions.

Reference:
    Beard, R. E., Pentikäinen, T., and Pesonen, E. (1984). Risk Theory.
    Panjer, H. H. and Willmot, G. E. (1992). Insurance Risk Models.
"""

from __future__ import annotations
from typing import Union
import numpy as np
from numpy.typing import ArrayLike, NDArray
from therisktheory import _validation

def retained_loss(
    loss: ArrayLike,
    retention_limit: float
) -> NDArray[np.float64]:
    """Calculate the retained loss by an insurer given a retention limit.

    Args:
        loss (ArrayLike): Loss amount(s) incurred.
        retention_limit (float): Maximum retention limit per risk or event.

    Returns:
        NDArray[np.float64]: The portion of the loss retained by the insurer.
    """
    loss_arr = np.asarray(loss, dtype=np.float64)
    if retention_limit < 0:
        raise ValueError("Retention limit must be non-negative.")
    if np.any(loss_arr < 0):
        raise ValueError("Loss amounts cannot be negative.")
    return np.minimum(loss_arr, retention_limit)

def retention_capacity(
    surplus: float,
    maximum_retention_ratio: float
) -> float:
    """Determine the financial retention capacity based on insurer surplus.

    Args:
        surplus (float): Insurer's available solvency margin or surplus.
        maximum_retention_ratio (float): Maximum proportion of surplus allowed for retention.

    Returns:
        float: Maximum retention capacity amount.
    """
    if surplus < 0 or not (0.0 <= maximum_retention_ratio <= 1.0):
        raise ValueError("Invalid surplus or retention ratio.")
    return float(surplus * maximum_retention_ratio)

def inflate_losses(
    loss: ArrayLike,
    inflation_rate: float,
    periods: int = 1
) -> NDArray[np.float64]:
    """Project the impact of inflation on risk losses over time.

    Args:
        loss (ArrayLike): Base loss amount(s).
        inflation_rate (float): Periodic inflation rate (e.g., 0.05 for 5%).
        periods (int, optional): Number of periods. Defaults to 1.

    Returns:
        NDArray[np.float64]: Inflated loss amount(s).
    """
    loss_arr = np.asarray(loss, dtype=np.float64)
    if inflation_rate < -1.0 or periods < 0:
        raise ValueError("Invalid inflation rate or periods.")
    return loss_arr * ((1.0 + inflation_rate) ** periods)

def deflate_losses(
    inflated_loss: ArrayLike,
    inflation_rate: float,
    periods: int = 1
) -> NDArray[np.float64]:
    """Discount the effect of inflation to bring losses back to present value.

    Args:
        inflated_loss (ArrayLike): Inflated loss amount(s).
        inflation_rate (float): Periodic inflation rate used for discounting.
        periods (int, optional): Number of periods. Defaults to 1.

    Returns:
        NDArray[np.float64]: Deflated (present value) loss amount(s).
    """
    loss_arr = np.asarray(inflated_loss, dtype=np.float64)
    if inflation_rate == -1.0 and periods > 0:
        raise ValueError("Division by zero: inflation rate cannot be -1.0 with positive periods.")
    if inflation_rate < -1.0 or periods < 0:
        raise ValueError("Invalid inflation rate or periods.")
    return loss_arr / ((1.0 + inflation_rate) ** periods)
