# Wider XOR fusion templates

`synthesize_xor_pairs.py` checks wider constant-coefficient rewrites than the
single-affine replacements in `xor_audit.py`. It preserves the original hash
and uses necessary low-bit projections to reject complete 32-bit identities.

The first family is `(a*x+b) XOR (d*x+e)`, with arbitrary coefficients. Moving
H2's XOR constant into this form of the existing H3/H4 fusion is inconsistent
already at nine bits. Combining the fused H3/H4 and H5 into this three-operation
form is inconsistent at twelve bits.

The second family is `p*((a*x+b) XOR selector)+q`, with selector either `x` or
any constant. Both twelve-bit systems are inconsistent. The odd outer
multiplier is parameterized by its inverse, turning necessary sample equations
into `(a*x+b) XOR selector = d*target(x)+e`. This avoids products between
unknown coefficients. Oddness follows from the target's varying low bit.

The nine-bit versions of the H3/H4/H5 target admit models because the shifted
branch vanishes at that precision. They are not valid 32-bit hash rewrites;
the next projection rejects them. UNSAT on a necessary subset of inputs is a
rejection proof, while a SAT sample fit is checked separately for universal
equivalence at its stated width. No 32-bit candidate was found.

The reports contain coefficients where applicable, exact necessary inputs,
counterexamples, bit widths, and solver status. These results cover only the
named templates and do not prove global optimality or rule out other rewrites.

```sh
python3 synthesize_xor_pairs.py
```
