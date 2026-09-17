# Verified 917-cycle checkpoint

The submitted `perf_takehome.KernelBuilder` now executes the standard shape
in **917 cycles / 10,810 static VLIW bundles**, within the 12,000-bundle cap.
The **<900** target remains unmet.

This checkpoint starts from the 918-cycle constant-load alternative and folds
the final path bit of group 22 into the fixed PC offset vector. Addresses
2318, 2326, ..., 2374 form the zero choices; 2319, 2327, ..., 2375 form the
one choices. All but the last of those new choices are scalar pointers already
used by the temporary child buffers. With q2 retained from the previous level,
the target is `2*q2 + selected_offset`, equivalent to `offset + 2*q2 + bit`.
The handlers still use the complete three-bit choice for node lookup.

The PC fold saves one VALU operation, adds one ALU constant and one FLOW
selection: seven weighted arithmetic slots net. The candidate also uses the
runtime root already loaded in the header and retains original depth-4 nodes,
avoiding 16 scalar XORs during restoration. The 48 scalar constants transferred
to LOAD remain from the preceding alternative.

| Metric | Submitted 918 | Submitted 917 |
| --- | ---: | ---: |
| Dynamic cycles | 918 | 917 |
| Static bundles | 10,811 | 10,810 |
| Static engine-slot operations | 230,118 | 232,189 |
| ALU slots | 10,947 | 10,932 |
| VALU slots | 5,472 | 5,465 |
| LOAD slots | 1,712 | 1,760 |
| STORE slots | 839 | 839 |
| Executed FLOW slots, including bootstrap/pause | 875 | 876 |
| Weighted arithmetic, ALU + 8*VALU | 54,723 | 54,652 |
| Compressed payload bytes | 151,558 | 151,870 |
| Scratch address span | 1,536 | 1,536 |
| Live-word peak | 1,336 | 1,357 |

The static layout consists of 557 main bundles, 9,600 case bundles and 653
padding bundles. All are counted. The graph's resource-window lower bound is
913, not a global algorithmic bound. Counts alone still require at least
`54,652 - 899*60 = 712` fewer weighted arithmetic slots for a strict sub-900
result, before dependency and other-port constraints.

Verification completed:

- All nine unchanged submission tests and all three native tests pass at 917.
- The 12 allocator, scheduler and memory-order regressions pass.
- Seeds 0–9 pass all 20,480 hash checkpoints per seed, every executed PC mapping,
  final values and preservation of all non-output memory.
- The actual submitted KernelBuilder passes the same checks on seeds 901 and
  12345; its complete expanded program equals the verified lowered program.
- The standalone embedded decoder expands to the exact verified program.
- Seed 0 and seed 1 take different PC paths with identical per-cycle port counts.
- The initial pause shares logical cycle 1; the frozen ISA and cycle accounting
  are unchanged. No test or problem source was modified.

The latest measured port plot is `port_utilization.png`, with a JSON summary
and per-cycle CSV beside it. It was visually inspected. The earlier root-level
918 profile is preserved, including its pre-existing local timestamp edit.

```sh
python3 export_candidate.py results/compact_917 --write
python3 tests/submission_tests.py
python3 -m unittest perf_takehome test_lane_allocate test_schedule test_memory_order -v
python3 plot_utilization.py --output results/compact_917/port_utilization.png
python3 resource_bounds.py results/compact_917 --output results/compact_917/bounds.json
```
