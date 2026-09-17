# Verified 914-cycle checkpoint

The submitted kernel executes in **914 cycles / 10,807 static VLIW bundles**.
This improves the preceding 915-cycle checkpoint by one cycle and one bundle.
The strict **<900** target remains unmet; the bundle limit remains **12,000**.

Two input groups form the depth-5 sibling address prefix after round 3 and add
the final branch bit after round 4. Ninety-eight scalar-constant expressions
use operands that were already ready in the source schedule. Eight round-12
shallow path updates use two FLOW selections each instead of a MADD. The final
scalar/vector balance and scratch allocation are part of the saved schedule.

| Metric | Previous 915 | Current 914 |
| --- | ---: | ---: |
| Dynamic cycles | 915 | 914 |
| Static bundles | 10,808 | 10,807 |
| Static engine-slot operations | 231,286 | 231,289 |
| ALU | 10,913 | 10,904 |
| VALU | 5,458 | 5,454 |
| LOAD | 1,769 | 1,769 |
| STORE | 912 | 912 |
| Executed FLOW, including bootstrap/pause | 848 | 864 |
| W = ALU + 8*VALU | 54,577 | 54,536 |
| Resource-window lower bound | 912 | 911 |
| Scratch live-word peak | 1,394 | 1,410 |

The final layout contains 554 main bundles, 9,600 case bundles and 653 padding
bundles. All are counted. The standalone compressed payload is 158,118 bytes.
Scratch addresses remain within 1,536 words; the allocator uses `lanes-9`.
The hash, runtime tree/input data, ISA and machine accounting are unchanged.

Validation:

- Nine submission tests and three native tests pass at 914 cycles.
- Twenty-four allocator, scheduling, memory-order and constant-rewrite
  regressions pass.
- Seeds 0–9 pass all 20,480 retained hash checkpoints per seed, every executed
  PC mapping, final outputs and preservation of all non-output memory.
- Seeds 901 and 12345 repeat those checks through the actual submitted
  KernelBuilder, whose entire expanded program equals the verified lowerer.
- The standalone decoder expands to the exact verified program.
- `perf_takehome.py` changes are confined to the generated block.
- The latest port plot was visually inspected. Seeds 0 and 1 use different
  dispatch PCs but identical per-cycle port counts; their profiles are bound
  to the promoted source SHA-256.

The final source is `../transfer_refine_914/candidate_014`. The first 914 result,
`../scheduled_constants_915/candidate_008`, had W=54,592; the 24-case follow-up
retained 914 while reducing work by 56. Relative to production 915, the net
work reduction is 41. This is a measured improvement, not a sub-900 result.
At least `54,536 - 899*60 = 596` further W must disappear under the optimistic
count-only arithmetic test; timing windows and scratch still constrain it.

```sh
python3 export_candidate.py results/compact_914 --write
python3 tests/submission_tests.py
python3 perf_takehome.py
python3 verify_actual_checkpoint.py results/compact_914
python3 resource_bounds.py results/compact_914 --output results/compact_914/bounds.json
python3 plot_utilization.py --output results/compact_914/port_utilization.png
python3 results/compact_914/report.py
```
