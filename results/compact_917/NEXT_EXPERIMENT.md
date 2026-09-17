# Late child-pair layout — implemented and measured in research

The proposal below has now been implemented in `optimize.py` and measured in
`results/late_pair_rows_917/`. Its best screen is 919 cycles; the seven-pair
layout saves 94 weighted arithmetic slots but executes in 927 cycles. It has
not replaced the submitted kernel. See that directory's README for validation,
padding rules, reuse ordering and follow-up screens. The remainder records the
original proposal and its verification requirements.

The seven paired depth-3 regions in round 14 still copy the second stream's
two children with ALU (16 copies per region). Unlike round 3, their outgoing
parity has no live coordinate-update consumer after dead-code elimination:
round 15 only needs that parity to choose its final node. This may permit
a different temporary layout without also converting an index vector.

## Candidate layout

Use one 48-word buffer in the unused index area, starting at 2054. Stream A
uses a row beginning at 2054, stream B at 2078. In handler j, each stream does
one real `vstore` at `row_base + 2*j`, starting at the chosen adjacent child
pair in the mirrored depth-4 scratch cache. The first two source words are
useful. The other six words are padding; later stores overwrite any such
padding inside the final 16-word payload. Stores run in increasing j order.
After all handlers, each row contains `L0,R0,L1,R1,...,L7,R7`.

Load each row as two eight-word blocks. For each four-input block, spread
the four parity bits into the even scratch lanes, perform one `vselect`
between the loaded block and its one-word-shifted view, and consume only the
even results with the existing scalar node-mix XORs. The final hash input
must be assembled as the usual contiguous eight-word vector. Force the
round-14 bit masks and the round-15 mix XORs to scalar form without changing
their weighted arithmetic count; exclude these forced packs from migration.

This uses four LOADs per paired region instead of two and four node FLOW
selections instead of two. It removes 16 scalar child copies per region.
Across all seven pairs, gross changes are -112 ALU copies, +14 LOAD and +14
FLOW. New row pointers, clears, lifetimes, and engine rebalancing still need
to be counted. This is not a net-saving or cycle claim. Additional dispatch
chains would be required before the extra FLOW count can fit a sub-900 graph.

## Requirements before treating a candidate as valid

- Existing `pt.V` virtual addresses have stride eight. Merely setting a size
  above eight would alias another virtual value. Use an isolated wider
  research reference type (a subclass accepted by the existing lowerers),
  and have `lane()` preserve its type. Leave ordinary graphs' representation
  unchanged. The native allocator must accept a per-value word stride.
- A child source at offset six needs eight physically in-bounds readable
  words. Reserve a 14-word span for the affected biased depth-4 blocks.
  Loaded pair blocks need a nine-word span for shifted vector reads.
- Only padding source words and odd selection lanes may be unobserved.
  Prove that they have no path to any hash, branch, required memory or output.
  Every actual vector output word still needs a write lifetime so unused
  outputs cannot clobber other live scratch values. No machine rule changes.
- Implement padding semantics explicitly in the research interpreter;
  do not globally replace missing-value errors with zero.
- Preserve all 20,480 hash checkpoints, all PC mappings, outputs and all
  non-output memory. Clear the complete 48-word temporary buffer afterward.
- A block-zero load must follow all stores overlapping words 0..7; a
  block-one load must follow all stores overlapping words 8..15. Reusing
  the buffer requires the prior block-zero reads before next handler 0's
  stores, and prior block-one reads before next handler 1's stores.
  Same-cycle reads precede writes. Merging regions must include these loads
  in the compound unit to avoid a cyclic scheduling graph.
- Add a frozen-machine microbenchmark that poisons padding, exercises all
  eight child-pair choices and buffer reuse, and verifies the payload plus
  restoration. Require full frozen execution before reporting performance.

The source checkpoint is `results/compact_917`. This proposal is available as
a research configuration and is absent from the submitted kernel.
