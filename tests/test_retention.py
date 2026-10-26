import pytest
import numpy as np
from therisktheory.retention import retained_loss, retention_capacity, inflate_losses, deflate_losses

def test_retained_loss_scalar_and_array():
    # Test scalar
    assert retained_loss(150.0, 100.0) == 100.0
    assert retained_loss(50.0, 100.0) == 50.0
    
    # Test array (vectorization)
    losses = np.array([50.0, 150.0, 200.0])
    limit = 100.0
    expected = np.array([50.0, 100.0, 100.0])
    np.testing.assert_array_equal(retained_loss(losses, limit), expected)

def test_retained_loss_invalid():
    with pytest.raises(ValueError):
        retained_loss(100.0, -10.0)

def test_retention_capacity():
    surplus = 1000.0
    ratio = 0.25
    assert retention_capacity(surplus, ratio) == 250.0

    with pytest.raises(ValueError):
        retention_capacity(-100.0, 0.2)
    with pytest.raises(ValueError):
        retention_capacity(1000.0, 1.5)

def test_inflation_functions():
    base = 100.0
    rate = 0.05
    periods = 2
    
    inflated = inflate_losses(base, rate, periods)
    expected_inflated = 100.0 * (1.05 ** 2)
    assert np.isclose(inflated, expected_inflated)
    
    deflated = deflate_losses(inflated, rate, periods)
    assert np.isclose(deflated, base)

def test_inflation_invalid():
    with pytest.raises(ValueError):
        inflate_losses(100.0, -1.5)
