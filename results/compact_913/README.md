# Verified 913-cycle checkpoint

This first 913 checkpoint takes **913 cycles / 11,049 static VLIW bundles**.
The current submission is the smaller equivalent layout in `../compact_913_packed/`.
The static cap is **12,000**. The **<900-cycle goal remains unmet**.

Eight singleton dispatch tables interleave cases so their PC offset vectors
can reuse scalar child-buffer addresses. A short initial prefix occupies
otherwise empty positions before the fixed tables. FLOW materializes constant
19, allowing a 913-cycle schedule. Thirteen scalar constants then move from ALU
to measured FLOW holes, one root-mix pack moves from VALU to ALU, and eight more
constants use measured LOAD holes. All retained hash computations remain intact.

| Metric | Previous 914 | Current 913 |
| --- | ---: | ---: |
| Dynamic cycles | 914 | 913 |
| Static bundles | 10,807 | 11,049 |
| ALU | 10,892 | 10,870 |
| VALU | 5,449 | 5,442 |
| LOAD | 1,771 | 1,778 |
| STORE | 984 | 984 |
| Runtime FLOW, including pause/bootstrap | 864 | 881 |
| W = ALU + 8*VALU | 54,484 | 54,406 |
| Fixed-graph resource-window bound | 910 | 909 |
| Scratch live-word peak | 1,388 | 1,322 |
| Static engine-slot operations | 231,948 | 231,523 |

The physical layout is 553 main, 9,600 case and 896 padding bundles. The
bootstrap executes in cycle 6 and the initial pause in cycle 1. Graph FLOW
includes that pause but excludes the bootstrap, hence 880 versus 881 at runtime.
Scratch span remains 1,536 words. The compressed payload is 162,335 bytes.

Validation completed:

- Nine submission tests, three native tests and 38 research regressions pass.
- Seeds 0–9 pass 20,480 retained hash checkpoints each, executed-PC mappings,
  outputs and preservation of every non-output memory word.
- The actual KernelBuilder passes the same checks on seeds 901 and 12345.
  Its complete program and the standalone expansion equal the verified lowerer.
- Only the generated production block changes. The hash, machine and original
  tests are unchanged; tree and input values remain runtime data.
- The port plot is bound to the source SHA and visually inspected. Two seeds
  take different PC paths with identical per-cycle port counts.

The source is `../load_holes_913/candidate_002`. Four longer independent searches
do not improve 913; they do not prove optimality. Count-only arithmetic still
requires at least `54,406 - 899*60 = 466` fewer W. Other ports, dependency windows
and allocation must also fit.

```sh
python3 export_candidate.py results/compact_913 --write
python3 tests/submission_tests.py
python3 perf_takehome.py
python3 verify_actual_checkpoint.py results/compact_913
python3 plot_utilization.py --output results/compact_913/port_utilization.png
python3 results/compact_913/report.py
```
