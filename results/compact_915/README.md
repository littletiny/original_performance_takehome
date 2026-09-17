# Verified 915-cycle checkpoint

The submitted kernel now executes in **915 cycles / 10,808 static VLIW bundles**.
It remains within the latest 12,000-bundle cap. The strict **<900** goal is unmet.

Sixteen even-numbered input groups use VLOAD at a one-based heap coordinate
and consume lane six for depths 8–10. The implicit lane offset removes the
six-word address compensation without an additional LOAD. This releases 32
FLOW selections; four MADD node selections at depth four move to FLOW. Selected
late broadcasts use STORE fills plus VLOAD, with complete index-row restoration.

The full derivation and search evidence are in `../overfetch_917/README.md` and
`../memory_vectors_917/README.md`. The hash algorithm and every retained hash
stage are unchanged. Tree/input data remain runtime values.

| Metric | Previous 917 | Current 915 |
| --- | ---: | ---: |
| Dynamic cycles | 917 | 915 |
| Static VLIW bundles | 10,810 | 10,808 |
| Static engine-slot operations | 232,189 | 231,286 |
| ALU operations | 10,932 | 10,913 |
| VALU operations | 5,465 | 5,458 |
| LOAD operations | 1,760 | 1,769 |
| STORE operations | 839 | 912 |
| Executed FLOW, including bootstrap/pause | 876 | 848 |
| W = ALU + 8*VALU | 54,652 | 54,577 |
| Compressed payload bytes | 151,870 | 148,327 |
| Scratch address span | 1,536 | 1,536 |
| Live-word peak | 1,357 | 1,394 |

The new layout has 555 main bundles, 9,600 case bundles and 653 padding bundles;
all 10,808 are counted. Scalar/vector rebalancing and earlier consumption of
overfetch words are part of the saved schedule. The allocator uses policy
`lanes-6`; unused VLOAD outputs still reserve actual write lifetimes.

Verification completed:

- Nine unchanged submission tests and three native tests pass at 915.
- Seventeen allocator, scheduling, memory-order and padding regressions pass.
- Seeds 0–9 pass all 20,480 hash checkpoints per seed, every executed PC mapping,
  final outputs and preservation of all non-output memory.
- The actual submitted KernelBuilder passes the same full checks on seeds 901
  and 12345; its complete expanded program equals the verified lowered program.
- The embedded standalone decoder expands to the exact verified program.
- The latest port utilization plot is `port_utilization.png`, with its per-cycle
  CSV and source-bound JSON. It was visually inspected.

W is 75 lower, and runtime FLOW is 28 lower, than the previous production
checkpoint. At 915 cycles the arithmetic ports are still about 99.4% occupied.
The fixed graph's resource-window lower bound is 912. Strict sub-900 still
requires at least `54,577 - 899*60 = 637` fewer weighted arithmetic slots before
startup, tail and other-port constraints; this is not a global algorithm bound.

The source is `../overfetch_refine_917/candidate_003`. Broader overfetch/path
transfers reduce W further but did not beat this verified cycle count. Future
work can use the freed FLOW capacity, with explicit accounting of allocation,
other port windows and the static bundle cap.

```sh
python3 export_candidate.py results/compact_915 --write
python3 tests/submission_tests.py
python3 -m unittest test_lane_allocate test_schedule test_memory_order test_late_pair_rows test_memory_vectors test_overfetch -v
python3 resource_bounds.py results/compact_915 --output results/compact_915/bounds.json
python3 plot_utilization.py --output results/compact_915/port_utilization.png
```
