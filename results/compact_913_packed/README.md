# Verified 913-cycle / 10,537-bundle submission

The current submitted kernel takes **913 cycles**, with **10,537 static bundles**
under the 12,000 cap. Its weighted arithmetic is **54,406**, 78 below the preceding
914 checkpoint. The **<900-cycle goal remains unmet**.

Interleaved PC tables reuse child-buffer scalar addresses. A short prefix frees
early FLOW slots for constants, and measured FLOW/LOAD holes remove scalar ALU
constant work. Finally, two PC offset rewrites place four tables in existing
holes, removing 512 padding bundles while retaining every scheduled operation
time and engine count from the first 913 checkpoint.

| Metric | Previous 914 | Current 913 |
| --- | ---: | ---: |
| Dynamic cycles | 914 | 913 |
| Static bundles | 10,807 | 10,537 |
| W = ALU + 8*VALU | 54,484 | 54,406 |
| Resource-window lower bound | 910 | 909 |
| Scratch live-word peak | 1,388 | 1,322 |
| Static engine-slot operations | 231,948 | 231,523 |

Executed slots are ALU 10,870, VALU 5,442, LOAD 1,778, STORE 984 and FLOW 881.
Graph FLOW is 880 because it includes the pause but excludes the bootstrap.
Scratch span remains 1,536. The layout is 553 main, 9,600 case and 384 padding
bundles; bootstrap is cycle 6 and pause cycle 1. The payload is 162,359 bytes.

Nine submission tests, three native tests and 38 research regressions pass.
Frozen seeds 0–9 and actual-builder seeds 901/12345 pass all 20,480 retained hash
checkpoints per seed, executed-PC mapping, outputs and all non-output memory.
The complete actual builder and standalone expansion equal the verified lowerer.
Only the generated production block changes. The hash, machine and original
tests are unchanged; all tree and input values remain runtime data.

The source-bound port plot is visually checked. Two seeds have different PC
paths and identical per-cycle port counts. Count-only arithmetic still requires
at least `54,406 - 899*60 = 466` fewer W; timing and other resources also constrain
the target. The current search results do not establish optimality.

```sh
python3 export_candidate.py results/compact_913_packed --write
python3 tests/submission_tests.py
python3 perf_takehome.py
python3 verify_actual_checkpoint.py results/compact_913_packed
python3 plot_utilization.py --output results/compact_913_packed/port_utilization.png
python3 results/compact_913_packed/report.py
```
