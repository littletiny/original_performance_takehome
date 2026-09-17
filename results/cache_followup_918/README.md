# Cache, dependency, and hash follow-up from 918 cycles

The submitted kernel remains **918 cycles / 10,811 static bundles**. None of
these experiments improved it. Its expanded program was compared in full with
the rebuilt default graph after the development-tool changes.

## Wide stores of grandchild rows

`grand_row_groups` moves selected four-node caches into one to three reusable
72-word buffers in the index area. A real `vstore` copies an eight-node source
block. For an odd quartet choice, the destination moves back four words; the
useful quartet therefore always appears in the same four positions of an
eight-word row. Writes to neighboring rows overlap only their unused padding.
The selected node is later loaded from an address chosen by the existing path
bits. All modified index words are restored to zero.

Each row store consumes one of the two STORE slots, so one previously stored
child is copied with ALU instead. Across all eleven existing grandchild groups,
the two-buffer graph removes 231 ALU instructions, adds seven VALU instructions,
and adds 77 LOADs before rebalancing. Lower arithmetic work did not produce a
faster kernel; the 48-configuration screen's best allocated result was 930.

`lookup_vstore` exists only in the development IR and export descriptors. The
lowerer and standalone decoder expand it into the existing machine's `vstore`.
No ISA or machine limit was changed.

Partial reuse now preserves the last reader of every untouched buffer field.
`test_memory_order.py` delays an old field-one load past a later field-zero
write and checks the reference result. Removing the necessary dependency causes
a hash-stage mismatch. The semantic interpreter uses a dependency-valid logical
timeline when row reuse crosses construction order, with reads preceding all
same-cycle writes. Performance still requires allocated frozen-machine runs.

## Moving lookups to depth four

`dispatch4_groups` and `gather_dispatch4` let selected groups gather at depth
three, dispatch at depth four, and prepare the two depth-five children with
STORE slots. Fetch plans now include only streams that consume prefetched
children. `search_depth4.py` preserves the existing hash instruction choices
and estimates a memory-buffer order from measured region times.

All 64 configurations that fit the screening size bound passed reference
semantics before scheduling. Their best allocated result was 948 cycles.

## Restore ranges and root reuse

`precise_restore` waits only for biased levels that overlap a restored block.
Fourteen restore/buffer variants preserved 918 but did not improve it. Failure
to allocate some tested schedules is not a proof of globally impossible
allocations.

`header_root` reuses the runtime root value already read at memory word seven
by the header `vload`. Four variants remained at 918. `scalar_root_groups`
replaces selected root broadcasts/mixes with scalar XORs while preserving the
hash-choice counter. Forced scalar root packs are excluded from automatic
scalar/vector migration. Sixteen variants also failed to improve 918; the
variant making every root mix scalar reached 922 with a header root.

## Frozen execution evidence

Each listed probe passed ten seeds, all 20,480 retained hash-stage checkpoints
per seed, the complete PC-to-cycle mapping, final outputs, and preservation of
non-output memory. The exported expansion also matched the complete program.

| Saved probe | Cycles | Static bundles |
| --- | ---: | ---: |
| `grand_rows_918/candidate_025` | 930 | 10,823 |
| `grand_rows_918/candidate_032` | 957 | 10,850 |
| `grand_row_pair_probe/candidate_000` | 922 | 10,815 |
| `depth4_918/candidate_027` | 948 | 11,865 |
| `scalar_roots_918/candidate_002` | 922 | 10,815 |
| `header_root_918/candidate_003` | 918 | 10,811 |

The wide-store regression also executes every quartet choice on the unchanged
frozen machine, checking both destination alignments and neighboring rows.
The allocator, scheduler, and memory-order regression suites pass 12 tests.

Reproduce the screens from the repository root:

```sh
python3 search_grand_rows.py results/compact_918 results/grand_rows_918
python3 search_depth4.py results/compact_918 results/depth4_918
PYTHONPATH=. python3 results/precise_restore_918/run.py
PYTHONPATH=. python3 results/header_root_918/run.py
PYTHONPATH=. python3 results/scalar_roots_918/run.py
python3 -m unittest test_lane_allocate test_schedule test_memory_order -v
```

Saved search summaries use 100000 to indicate that no allocated candidate was
retained. Those rows are not measured execution scores.
