"""Unit tests for De Finetti framework comparison and elementary insurance cases."""

from __future__ import annotations
import numpy as np
import pytest
from therisktheory.de_finetti_comparison import (
    elementary_insurance_case,
    compare_classical_de_finetti,
)

def test_elementary_insurance_case_boundaries() -> None:
    """Pruebas de valores estándar y fronteras para el caso elemental de seguros."""
    # Con valores escalares
    assert elementary_insurance_case(100.0, 50.0, 0.2) == 100.0 + 50.0 - (0.2 * 50.0)
    
    # Con probabilidades límite (0.0 y 1.0)
    assert elementary_insurance_case(100.0, 50.0, 0.0) == 150.0
    assert elementary_insurance_case(100.0, 50.0, 1.0) == 100.0
    
    # Verificación vectorizada con NumPy
    capitals = np.array([100.0, 200.0])
    premiums = np.array([50.0, 50.0])
    probs = np.array([0.1, 0.5])
    res = elementary_insurance_case(capitals, premiums, probs)
    np.testing.assert_allclose(res, [145.0, 225.0])

def test_compare_classical_de_finetti() -> None:
    """Prueba del cálculo de desviaciones absolutas entre modelos de ruina."""
    classical = np.array([0.05, 0.15])
    de_finetti = np.array([0.07, 0.12])
    diffs = compare_classical_de_finetti(classical, de_finetti)
    np.testing.assert_allclose(diffs, [0.02, 0.03])

def test_invalid_types_and_values() -> None:
    """Pruebas de manejo de excepciones para NaN, Inf, probabilidades fuera de rango y negativos."""
    # Probabilidades fuera de rango [0, 1]
    with pytest.raises(ValueError):
        elementary_insurance_case(100.0, 50.0, 1.5)
    with pytest.raises(ValueError):
        elementary_insurance_case(100.0, 50.0, -0.1)
        
    # Capital o primas negativas
    with pytest.raises(ValueError):
        elementary_insurance_case(-10.0, 50.0, 0.2)
    with pytest.raises(ValueError):
        elementary_insurance_case(100.0, -50.0, 0.2)
        
    # NaN o Infinitos
    with pytest.raises(ValueError):
        elementary_insurance_case(np.nan, 50.0, 0.2)
    with pytest.raises(ValueError):
        compare_classical_de_finetti(float('inf'), 0.1)
        
    # Probabilidades de ruina inválidas en la comparación
    with pytest.raises(ValueError):
        compare_classical_de_finetti(-0.05, 0.1)
    with pytest.raises(TypeError):
        compare_classical_de_finetti("0.05", 0.1)
