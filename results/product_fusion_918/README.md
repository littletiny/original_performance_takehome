# Nonlinear product fusion templates

The target is the exact composition of original hash stages H3, H4 and H5.
The current implementation uses two MADDs, XOR and a final MADD. The new
search checks eleven alternatives with at most three operations, including:

- `(x XOR k)*(a*x+b)+c`, with AND/OR alternatives in place of XOR.
- `x*((a*x+b) XOR k)+c`, with AND/OR alternatives.
- `x*((a*x+b) XOR x)+c`.
- `x*(a*x+b)+(x XOR k)`.
- `c*((a*x+b) XOR k)+x` and `c*((a*x+b) XOR x)+x`.
- `x*(a*x+b)+c`, which also contains all products of two affine terms plus
  a constant, because their product expands to a quadratic polynomial.

All coefficients are arbitrary bit-vector constants. These families allow
intermediate-dependent multiplication, beyond the previously checked affine
XOR pairs and constant-multiplier chains. No candidate changes the fixed hash.

Every family is UNSAT on a necessary set of inputs in a 9-bit or 12-bit
projection. Because these expressions contain only prefix-compatible operations
(addition, multiplication, bitwise operations and the target's fixed left shift),
a complete 32-bit identity would satisfy those projected equations. UNSAT
therefore rejects each listed full-width template. It does not establish global
optimality or exclude unlisted expressions.

Eight 9-bit systems have universally verified models, but the next projection
rejects them. They are not complete hash rewrites. `verification.json` records
independent exhaustive Python checks of each model over all 512 inputs and of
the target against the original three hash stages at both 9 and 12 bits.
The other JSON files retain the necessary inputs, coefficient models and status.

```sh
.solver-venv/bin/python synthesize_product_fusion.py
```
