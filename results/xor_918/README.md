# Cross-stage XOR audit of the 918-cycle checkpoint

The hash function is unchanged. `xor_audit.py` checks 32-bit identities with
Z3 and audits the saved graph in `results/compact_918/`. Reproduce with:

```sh
python3 xor_audit.py
```

## Cancellation already used

Let `C6 = 0xb55a4f09` and let `u` be H5's result. At a round boundary,

```text
(u XOR (u >> 16) XOR C6) XOR node
  = (u XOR (u >> 16)) XOR (node XOR C6).
```

A cached node can include `XOR C6` once, while each input retains H6's biased
representation. Equivalently, the matching value and node biases cancel:

```text
(value XOR C6) XOR (node XOR C6) = value XOR node.
```

The submitted graph defers H6's constant in 384 of its 512 eight-lane hash
groups. The remaining 128 are in zero-based rounds 7, 8, 9, and 15. This count
is a gross hash-operation saving, not a net cycle claim: biased tree setup
and restoration also cost instructions. The graph currently has 272 scalar
lane equivalents of setup/restoration XOR work.

## Additional templates checked

For `T_s(x) = x XOR (x >> s)`, both `T_16` and `T_19` are involutions on
32-bit words. A constant may be transported through either transform:

```text
T_s(x XOR T_s(C)) = T_s(x) XOR C.
```

This suggests eliminating H2's fixed XOR by producing a biased H1 result,
or eliminating H6's fixed XOR by producing a biased H5 result. The required
biases are respectively `0xc761dad0` and `0xb55afa53`.

Neither biased result is a single affine MADD `a*x+b`, even with arbitrary
constant multiplier and addend. The stronger template `a*(x XOR k)+b` also
fails, so allowing an additional fixed bias in cached tree nodes does not
make this particular H1 rewrite work. The H5 check likewise permits a free
fixed input bias.

The solver returns UNSAT for necessary equations in 12-bit or 9-bit
projections. Since a valid 32-bit identity must satisfy those equations,
these are rejection proofs for the named templates, not random testing.
The report distinguishes exhaustive projection constraints from smaller
inconsistent sets of necessary constraints.

Pushing H2's constant through the existing two-MADD H4 fusion also fails:
the shifted branch has zero low nine bits, and the unshifted branch would
need an affine replacement of `33*(x XOR C2)+C3+C4` in those bits.

`audit.json` includes all statuses, projection widths, constraints' input
values, the required biases, and a concrete counterexample to commuting a
fixed XOR mask through H1's multiply-add. It also proves the existing H1,
H5, and fused H3/H4 MADD forms equivalent to the original stages.

These results do not prove global optimality or rule out more general
multi-instruction, representation, caching, or scheduling changes. The
submitted kernel remains 918 cycles and 10,811 static bundles; <900 is unmet.
