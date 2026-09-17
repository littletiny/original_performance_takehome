# Joint early/late rows, STORE spans and PC offsets

The 128-case inventory combines both row placements, six/eleven long grandchild
handlers, retained originals, dense/fixed table banks, PC-offset MADDs, scalar
hash constants and FLOW targets. Ninety-six graphs fit 12,000 bundles at a
hypothetical 899-cycle schedule. Their best resource-window bound is **901**.

The lowest arithmetic work is **53,698**, corresponding to a count-only
arithmetic bound of **895**. Its full resource-window bound is **923**, so it
does not satisfy the necessary sub-900 conditions. This is a concrete example
of why the arithmetic capacity bound cannot substitute for all port windows.

Eight selected graphs pass both reference seeds before 32 trials of 800
iterations. Only two retain allocated screens: 1,001 and 1,051 cycles; those
scores are not independently frozen-verified here. `100000` denotes no retained
allocated result. Earlier node loads are tested in `../joint_prefetch_914/`.
