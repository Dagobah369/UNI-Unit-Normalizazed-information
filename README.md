This proposal is based on the arithmetic framework UNI (Unity Normalization Interface) in which the unit 1 is decomposed into five fundamental dimensions A, B, C, D, E satisfying five independent constraints:
A + B + C = 1
A = 2B + 3C
(A + B)^D = 1/2
E[C₁₀] = 9/10
C = 1/(2N) - 1/N³, with N = 10

The unique solution of this system gives the quintuplet:
(A, B, C, D, E) = (0.683, 0.268, 0.049, 13.8, 181.014)
This quintuplet results from the arithmetic constraints. The resulting structure is closed, self-coherent, and reversible. The fundamental invariant C_n · D_n → ln(2) links the kernel to the propagation and constitutes the conservation structure of the system 1=1.

This arithmetic framework alone suffices to autonomously generate three fundamental objects:
The spectrum Z(t) = Σ w_n · e^{-i t D_n} whose minima coincide with the non-trivial zeros of the Riemann zeta function, with 100% coverage and a correlation of 1.000000;
The ntural integers \mathbb{N}, reconstructed by exact inversion n = C / (1 - exp(ln(1/2)/D));
The prime numbers \mathbb{P}, selected by the UNI product table, a direct consequence of the composition structure C_n = (C_i · C_j)/C ↔ n = i × j.
Reproducible results can be obtained via two approaches with a bounded window:

The arithmetic approach (ARI.PY): based on the spectrum Z(t), it achieves fine local precision (median gap 0.15%) over a window of 6,784 zeros.
The analytic approach (ANA.PY): based on the density ρ_UNI(m) = (U / 2π) * ln(mU / 2π), it extends to 2,001,052 zeros (data Odlyzko) and reconstructs 80,057 integers and 1,229 primes.
Both approaches verify the closure of the cycle:
P --UNI table--> Z(t) --minima--> positions --inversion--> N --UNI table--> P

All information is available in the document UNI (Unity Normalization Interface)
Part I: Arithmetic basis of UNI
Part II: Application of UNI to natural numbers, prime numbers, and Riemann zeros

All results presented are fully reproducible. The Python script is documented and allows any reader to reproduce the calculations, modify parameters, and independently verify the results. The document UNI (Unity Normalization Interface) and the Python scripts (ARI.py, ANA.py) are available on GitHub at the following address:
https://github.com/Dagobah369/UNI-Unit-Normalizazed-information 

It should be noted that the zeros6.txt file (Odlyzko) serves only as an independent external comparison and that no external information affects the autonomous generation.
https://www-users.cse.umn.edu/~odlyzko/zeta_tables/ 

Results Table
ARI.py (arithmetic)
•	Principle: Minima of |Z(t)|
•	Zeros generated: 6,784
•	Integers reconstructed: 499 (up to 500)
•	Primes reconstructed: 95 (up to 500)
•	Coverage ℕ: 100% (within the bounded window)
•	Coverage ℙ: 100% (within the bounded window)
•	Mean error on γ: 0.001365
•	Median gap: 0.15%
•	Correlation: 1.000000

ANA.py (analytic)
•	Principle: Recurrence ∫ρ = 1
•	Zeros generated: 2,001,052
•	Integers reconstructed: 80,057 (up to 80,058)
•	Primes reconstructed: 1,229 (up to 10,000)
•	Coverage ℕ: 100% (within the bounded range)
•	Coverage ℙ: 100% (within the bounded range)
•	Mean error on γ: 0.184
•	Median gap: 28.3%
•	Correlation: 1.000000

https://doi.org/10.5281/zenodo.19363036

