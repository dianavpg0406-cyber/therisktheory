import numpy as np
import pytest
from therisktheory.retention import (
    retained_loss,
    retention_capacity,
    inflate_losses,
    deflate_losses,
)

def test_retained_loss_boundaries_and_types() -> None:
    assert retained_loss(0.0, 1000.0) == 0.0
    assert retained_loss(1500.0, 1000.0) == 1000.0
    assert retained_loss(500.0, 1000.0) == 500.0
    
    losses = np.array([200.0, 1500.0])
    res = retained_loss(losses, 1000.0)
    np.testing.assert_allclose(res, [200.0, 1000.0])

def test_retention_capacity() -> None:
    assert retention_capacity(1000.0, 0.25) == 250.0

def test_inflation_deflation_inversion() -> None:
    """Demostración formal de inversión entre inflación y deflación."""
    base = 1000.0
    rate = 0.045
    periods = 3
    inflated = inflate_losses(base, rate, periods)
    deflated = deflate_losses(inflated, rate, periods)
    np.testing.assert_allclose(deflated, base)

def test_inflation_edge_cases_and_division_by_zero() -> None:
    with pytest.raises(ValueError):
        deflate_losses(100.0, -1.0, periods=2)

def test_invalid_inputs_and_exceptions() -> None:
    with pytest.raises(ValueError):
        retained_loss(-100.0, 500.0)
    with pytest.raises(ValueError):
        retained_loss(np.nan, 500.0)
    with pytest.raises(TypeError):
        retained_loss("100.0", 500.0)
    with pytest.raises(ValueError):
        inflate_losses(100.0, -1.5)
    with pytest.raises(TypeError):
        inflate_losses(100.0, 0.05, periods=1.5)
    with pytest.raises(ValueError):
        retention_capacity(1000.0, 1.5)
