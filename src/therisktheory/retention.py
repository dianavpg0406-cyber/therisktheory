"""Risk retention, division, and inflation adjustment models in actuarial science."""

from __future__ import annotations
from typing import Union
import numpy as np
import numpy.typing as npt

NUMERIC_KINDS = {"i", "u", "f"}

def _validate_numeric_array(value: Union[float, int, npt.ArrayLike], name: str) -> npt.NDArray[np.float64]:
    """Validate that input is a finite, non-negative numeric value or array."""
    try:
        array = np.asarray(value)
    except (ValueError, TypeError) as error:
        raise TypeError(f"{name} must be a real number or array") from error

    if array.dtype.kind not in NUMERIC_KINDS:
        raise TypeError(f"{name} must be a numeric type, got dtype {array.dtype}")

    array = array.astype(np.float64, copy=False)

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values (no NaN or inf)")

    if np.any(array < 0):
        raise ValueError(f"{name} must be non-negative")

    return array

def retained_loss(loss: Union[float, int, npt.ArrayLike], retention_limit: Union[float, int, npt.ArrayLike]) -> npt.NDArray[np.float64]:
    """Computes the retained loss R(X) = min(X, M) under a retention limit M."""
    l_arr = _validate_numeric_array(loss, "loss")
    m_arr = _validate_numeric_array(retention_limit, "retention_limit")
    try:
        return np.minimum(l_arr, m_arr)
    except OverflowError as err:
        raise OverflowError("Numerical overflow occurred in retained_loss computation.") from err

def retention_capacity(sur
