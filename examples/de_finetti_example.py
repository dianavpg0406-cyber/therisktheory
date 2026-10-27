"""Example script demonstrating De Finetti comparative analysis and elementary insurance."""

import numpy as np
from therisktheory.de_finetti_comparison import (
    elementary_insurance_case,
    compare_classical_de_finetti,
)

def main() -> None:
    # 1. Elementary insurance case evaluation
    initial_capital = 1000.0
    premium = 200.0
    claim_prob = 0.15
    
    terminal_pos = elementary_insurance_case(initial_capital, premium, claim_prob)
    print(f"Elementary Insurance Terminal Position: {terminal_pos}")

    # 2. Comparative analysis between classical and De Finetti ruin probabilities
    classical_ruin = np.array([0.02, 0.05, 0.10])
    de_finetti_ruin = np.array([0.03, 0.06, 0.09])
    
    deviations = compare_classical_de_finetti(classical_ruin, de_finetti_ruin)
    print(f"Ruin Probability Absolute Deviations: {deviations}")

if __name__ == "__main__":
    main()
