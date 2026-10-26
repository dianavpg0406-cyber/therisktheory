import numpy as np
from typing import Union

def retained_loss(loss: Union[float, np.ndarray], retention_limit: float) -> Union[float, np.ndarray]:
    """Calculate the retained loss by an insurer given a retention limit.

    Args:
        loss (float or np.ndarray): Loss amount(s) incurred.
        retention_limit (float): Maximum retention limit per risk or event.

    Returns:
        float or np.ndarray: The portion of the loss retained by the insurer.
    """
    loss_arr = np.asarray(loss)
    if retention_limit < 0:
        raise ValueError("Retention limit must be non-negative.")
    return np.minimum(loss_arr, retention_limit)

def retention_capacity(surplus: float, maximum_retention_ratio: float) -> float:
    """Determine the financial retention capacity based on insurer surplus.

    Args:
        surplus (float): Insurer's available solvency margin or surplus.
        maximum_retention_ratio (float): Maximum proportion of surplus allowed for retention.

    Returns:
        float: Maximum retention capacity amount.
    """
    if surplus < 0 or not (0.0 <= maximum_retention_ratio <= 1.0):
        raise ValueError("Invalid surplus or retention ratio.")
    return surplus * maximum_retention_ratio

def inflate_losses(loss: Union[float, np.ndarray], inflation_rate: float, periods: int = 1) -> Union[float, np.ndarray]:
    """Project the impact of inflation on risk losses over time.

    Args:
        loss (float or np.ndarray): Base loss amount(s).
        inflation_rate (float): Periodic inflation rate (e.g., 0.05 for 5%).
        periods (int, optional): Number of periods. Defaults to 1.

    Returns:
        float or np.ndarray: Inflated loss amount(s).
    """
    loss_arr = np.asarray(loss)
    if inflation_rate < -1.0 or periods < 0:
        raise ValueError("Invalid inflation rate or periods.")
    return loss_arr * ((1.0 + inflation_rate) ** periods)

def deflate_losses(inflated_loss: Union[float, np.ndarray], inflation_rate: float, periods: int = 1) -> Union[float, np.ndarray]:
    """Discount the effect of inflation to bring losses back to present value.

    Args:
        inflated_loss (float or np.ndarray): Inflated loss amount(s).
        inflation_rate (float): Periodic inflation rate used for discounting.
        periods (int, optional): Number of periods. Defaults to 1.

    Returns:
        float or np.ndarray: Deflated (present value) loss amount(s).
    """
    loss_arr = np.asarray(inflated_loss)
    if inflation_rate < -1.0 or periods < 0:
        raise ValueError("Invalid inflation rate or periods.")
    return loss_arr / ((1.0 + inflation_rate) ** periods)
