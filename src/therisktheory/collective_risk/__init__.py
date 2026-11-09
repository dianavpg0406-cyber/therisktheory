"""
Collective risk theory, Lundberg fundamental equation, and safety equation subpackage for TheRiskTheory.
Integrated with frequency (Bracamonte) and severity (Garcia) modules.
"""

from .safety_equation import (
    adjustment_coefficient,
    lundberg_equation,
    full_retention_adjustment,
)

__all__ = [
    "adjustment_coefficient",
    "lundberg_equation",
    "full_retention_adjustment",
]
