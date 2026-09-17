# Form initial PC-offset vectors arithmetically

`pc_offset_madd` constructs an initial offset vector from an existing address
bank when its lane order and stride match. For an anchor bank with values
`anchor + 8*position[j]`, coefficient `cases*span/8` and addend
`table + 14 - coefficient*anchor`, the result is the exact absolute PC vector.
The fourteen-word fixed table origin is unchanged. Other shapes keep literal
PC loads. Numeric values used elsewhere are not relocated as PC literals.

The basic source replaces eight LOADs with two VALU operations (+16 W), using
anchor 14, coefficient 8 and the already-required scalar 2462. The released
LOAD capacity supports additional memory broadcasts. Seventeen configurations
receive reference seeds 0/1 and 32 trials of 750 iterations.

Candidate 002 is frozen-verified at **914 cycles / 10,807 bundles**, W=54,492.
It passes seeds 0–9, retained checkpoints, every PC mapping, outputs, all
non-output memory and exact standalone expansion. The addend's own memory
broadcast is evaluated separately in `../pc_delta_memory_914/`.
Candidate 009 is also frozen-verified at 918 cycles, exercising the generated
PC MADD together with globally permuted address banks and early child rows.

A real-machine test checks normal/permuted address banks, a wrapped negative
addend, three-cycle handlers and runtime case choices. It exposed a general
lowerer boundary bug: a last handler at the last logical cycle returned to the
first table instead of program end. That return now targets program end. The
previous production program is unaffected; the fix does not change the ISA
or cycle accounting.
