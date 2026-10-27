# Topic 1: Risk Division, Retention, and Inflation Models

## Mathematical concept implemented
Principles of risk division, financial retention capacity, and the effects of inflation on a risk portfolio (`retained_loss`, `retention_capacity`, `inflate_losses`, `deflate_losses`).

## Definitions and assumptions
- **Retained Loss**: Models the portion of loss retained by the insurer under a retention limit or deductible: $R(X) = \min(X, M)$.
- **Retention Capacity**: Defines financial retention capacity based on insurer surplus and retention ratio: $C = S \times r$.
- **Inflation & Deflation**: Projects compound inflation over discrete time periods and discounts losses back to present value: $X_t = X_0 (1 + i)^t$.
 
## Bibliographic references
- Bühlmann, H. (1970). *Mathematical Methods in Risk Theory*. Springer.
- Beard, R. E., Pentikäinen, T., & Pesonen, E. (1984). *Risk Theory*. Chapman and Hall.
- Panjer, H., & Willmot, G. (1992). *Insurance Risk Models*. Society of Actuaries.
