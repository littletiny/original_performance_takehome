# Verified 914-cycle checkpoint with less arithmetic

The submitted kernel remains at **914 cycles / 10,807 static bundles** and now
uses **54,484 weighted arithmetic slots**, 52 fewer than the preceding 914
checkpoint. Its fixed-graph resource-window bound is **910**, down from 911.
The **<900** goal is still unmet; the static limit remains **12,000 bundles**.

The first paired dispatch's eight PC-offset constants are replaced by
`8 * address_vector_14 + 2462`. The vector's lanes contain `14 + 8*j`, so the
result is the required absolute PC vector `2574 + 64*j`. The scalar 2462 is
already an input pointer. Its uniform vector and eight additional setup
vectors use ordered STORE fills followed by VLOAD in the existing cleared row.
One scalar hash-shift pack moves into a measured VALU hole.

| Metric | Previous 914 | Current 914 |
| --- | ---: | ---: |
| ALU | 10,904 | 10,892 |
| VALU | 5,454 | 5,449 |
| LOAD | 1,769 | 1,771 |
| STORE | 912 | 984 |
| Executed FLOW, including bootstrap/pause | 864 | 864 |
| W = ALU + 8*VALU | 54,536 | 54,484 |
| Resource-window lower bound | 911 | 910 |
| Scratch live-word peak | 1,410 | 1,388 |
| Static engine-slot operations | 231,289 | 231,948 |

The layout is still 554 main, 9,600 case and 653 padding bundles. The standalone
payload is 162,972 bytes. The scratch address span is 1,536 words, using
allocator policy `lanes-3`. The hash algorithm and every retained checkpoint
are unchanged. All tree/input values remain runtime data.

Validation completed:

- Nine submission tests, three native tests and 29 research regressions pass.
- Seeds 0–9 pass all 20,480 retained hash checkpoints per seed, every executed
  PC mapping, final outputs and preservation of all non-output memory.
- The actual submitted builder passes those checks on seeds 901 and 12345;
  its complete program equals the verified lowerer and standalone expansion.
- Source changes outside the generated production block are absent.
- The new port plot was visually inspected. Seeds 0 and 1 take different
  dispatch PCs but have identical per-cycle port counts.

The source is `../pc_delta_holes_914/candidate_000`. Eight hole-transfer
variants and four independent searches of 100 trials / 1,200 iterations do not
beat 914. The observed four-cycle gap above the bound is not proven unavoidable.
Count-only arithmetic still requires at least `54,484 - 899*60 = 544` fewer W;
startup, tails, other port windows and allocation also need to fit.

For a strict 899-cycle budget, a resource-window bound of 895 leaves four
cycles of scheduling margin; 890 leaves nine. Those are useful engineering
targets, not necessary or sufficient conditions. The current measured gap is
four, but the verified `../bound899_915/candidate_009` has bound 899 and takes
967 cycles. The gap includes constraints omitted by the bound and any search
shortfall; it is not proven unavoidable. Furthermore, the joint STORE screen
in `../pair_store_joint_914/` reaches arithmetic bound 895 while its stronger
resource-window bound is 923. Reducing arithmetic counts alone is insufficient.

```sh
python3 export_candidate.py results/compact_914_pc --write
python3 tests/submission_tests.py
python3 perf_takehome.py
python3 verify_actual_checkpoint.py results/compact_914_pc
python3 resource_bounds.py results/compact_914_pc --output results/compact_914_pc/bounds.json
python3 plot_utilization.py --output results/compact_914_pc/port_utilization.png
python3 results/compact_914_pc/report.py
```
