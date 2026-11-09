"""
Example script demonstrating Lundberg fundamental equation and adjustment coefficient calculation,
integrating Bracamonte's frequency intensity and Garcia's severity moments.
"""

import numpy as np
from therisktheory.collective_risk import (
    adjustment_coefficient,
    lundberg_equation,
    full_retention_adjustment,
)
from therisktheory.severity_aggregation.lognormal import lognormal_mean, lognormal_second_moment


def run_example():
    print("=== TheRiskTheory: Topic 5 - Collective Risk & Safety Equation Example ===")
    print("Integrating Garcia's severity models and Bracamonte's collective framework...\n")

    mu_log = 1.0
    sigma_log = 0.5
    mean_severity = lognormal_mean(mu_log, sigma_log)
    second_moment = lognormal_second_moment(mu_log, sigma_log)
    safety_loading = 0.20

    print(f"Severity Model (Garcia): Lognormal(mu = {mu_log}, sigma = {sigma_log})")
    print(f"Mean Severity E[X]: {mean_severity:.4f}")
    print(f"Second Moment E[X^2]: {second_moment:.4f}")
    print(f"Safety Loading lambda: {safety_loading:.2%}\n")

    r_approx = full_retention_adjustment((mean_severity, second_moment), safety_loading)

    print("Results:")
    print(f" - Full Retention Adjustment Coefficient Approximation (R): {r_approx:.6f}")


if __name__ == "__main__":
    run_example()
