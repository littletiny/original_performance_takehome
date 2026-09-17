# Additional four-operation hash templates

Let F be the original composition H3/H4/H5, C2 the H2 XOR constant, and
`D = C6 XOR (C6 >> 16) = 0xb55afa53`. Since `D XOR (D >> 16) = C6`, producing
`F(x) XOR D` could absorb H6's constant. The three synthesis targets are
`F(x XOR C2)`, `F(x) XOR D`, and `F(x XOR C2) XOR D`.

The existing implementation of F uses four operations. Each proposed form
also uses four operations, so a successful boundary-constant absorption could
remove a separate XOR. All coefficients in the following families are free:

| Family | Result for the three targets |
| --- | --- |
| `p*((a*x+b) XOR (d*x+e))+q` | H2/H6 rejected at 12 bits; both at 9 bits |
| `p*((a*x+b) XOR (d*x+e))+x` | All rejected at 9 bits |
| `p*((a*x+b) XOR (d*x+e))+(a*x+b)` | All rejected at 9 bits |
| Replace XOR in the three rows above with AND or OR | All 18 target/family combinations rejected at 9 bits |

The constant-addend XOR search uses the inverse odd outer multiplier. Output
parity forces oddness and, for XOR, exactly one inner multiplier to be odd;
term exchange and simultaneous top-bit flips choose equivalent representatives.
The AND/OR search does not apply those XOR-specific inner restrictions.

For the add-x XOR family, the value at x=0 bounds the power of two dividing p.
Every allowed power is checked, using an inverse for the odd part. The add-A
family's final search leaves p completely free; the older finite-multiplier
screen is redundant and is not needed for the rejection claim.

UNSAT of necessary low-bit equations rejects the named full-width expression:
the targets and candidate operations use addition, multiplication, bitwise
operations and fixed left shifts, all compatible with low-bit projection.
This is not a global hash lower bound. There is no full 32-bit rewrite candidate.

The two outer-XOR models at nine bits are projection models only. Independent
Python evaluation verifies both over all 512 inputs and compares all three
targets with the original hash stages at 9/12 bits and three high prefixes:
41,472 original-hash evaluations. Evidence is in `verification.json`.

```sh
.solver-venv/bin/python synthesize_affine_outer.py
.solver-venv/bin/python synthesize_affine_feedback.py --families add_x add_A_free \
  --output results/affine_feedback_free_917
.solver-venv/bin/python synthesize_bitwise_outer.py
python3 verify_affine_projections.py
```

Necessary inputs and solver results are retained here and in
`../affine_feedback_free_917/` and `../bitwise_outer_917/`.
