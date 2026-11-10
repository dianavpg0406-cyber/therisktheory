"""Example script demonstrating Lundberg fundamental equation and adjustment coefficient calculation,
integrating Adrián García's severity models (PR #10)[cite: 1]."""
import numpy as np
from therisktheory.collective_risk import (
    adjustment_coefficient,
    lundberg_equation,
    full_retention_adjustment,
)
from therisktheory.severity_aggregate import lognormal_severity_moments

def run_example() -> None:
    print("=== TheRiskTheory: Topic 5 - Collective Risk & Safety Equation Example ===")
    print("Integrating Adrián García's severity models (PR #10)[cite: 1] and collective framework...\n")
    
    mu_log = 1.0
    sigma_log = 0.5
    moments = lognormal_severity_moments(mu=mu_log, sigma=sigma_log)
    
    mean_severity = float(moments["mean"])
    second_moment = float(moments["second_raw_moment"])
    safety_loading = 0.20

    print(f"Severity Model (García - Lognormal): mu = {mu_log}, sigma = {sigma_log}")
    print(f"Mean Severity E[X]: {mean_severity:.4f}")
    print(f"Second Moment E[X^2]: {second_moment:.4f}")
    print(f"Safety Loading lambda: {safety_loading:.2%}\n")

    r_approx = full_retention_adjustment((mean_severity, second_moment), safety_loading)
    print("Results:")
    print(f" - Full Retention Adjustment Coefficient Approximation (R): {r_approx:.6f}")

    # Exact Lundberg solution with exponential severity for verification
    mu = 1.0 / mean_severity
    def mgf_exp(r: float) -> float:
        if r >= mu:
            return float("inf")
        return mu / (mu - r)

    r_exact = adjustment_coefficient(mgf_exp, mean_severity, safety_loading, r_max=mu - 1e-3)
    print(f" - Exact Lundberg Adjustment Coefficient (Exponential): {r_exact:.6f}")

if __name__ == "__main__":
    run_example()


if __name__ == "__main__":
    run_example()
