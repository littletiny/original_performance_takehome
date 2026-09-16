# Sub-900 optimization checkpoint

The current verified entry point is **928 cycles with 9,776 static bundles**,
down from the initial 980 cycles.
The requested **<900-cycle target has not been reached**.

The user revised the static VLIW instruction-bundle limit to **10,000**
(`len(KernelBuilder.instrs)`). Both the exporter and the standalone decoder
enforce this limit. The current program contains **220,244 individual
engine-slot operations**, including the initial pause; those are a different
quantity from bundles.

The preceding 932-cycle checkpoint used 477,092 static bundles. The current
version reduces that count by **97.95%** and uses 4 fewer dynamic cycles. Its
verified artifacts remain at `results/portfolio_1/`, and its submitted source
is preserved in commit `e2c302f`; it is no longer the default implementation.
The first compressed checkpoint, 971 cycles and 9,819 bundles, remains in
`results/compact_971/` and commit `8b4ec3a`.
The depth-5 prefetch checkpoint remains in `results/compact_939/` and commit
`5954122` (939 cycles, 9,787 bundles).
The address-chain checkpoint remains in `results/compact_936/` and commit
`1d667e6` (936 cycles, 9,784 bundles).
The first lane-allocation checkpoint remains in `results/compact_934/` and
commit `188b459` (934 cycles, 9,782 bundles).

## Verified checkpoint

Source configuration and schedule: `results/compact_928/config.json` and
`results/compact_928/best.npz`. Frozen verification and export metadata:
`results/compact_928/verification.json`. Fixed-graph resource counts and
100-cycle utilization windows are in `results/compact_928/analysis.json`.

The submitted `perf_takehome.py` is self-contained: its standard-shape builder
expands an embedded logical schedule and dense dispatch descriptors with Python
standard-library modules. It does not import the optimization tools or read the
saved artifacts. Other shapes retain the previous implementation.

Validation completed:

- The unchanged `python3 tests/submission_tests.py` passes all nine tests at
  928 cycles.
- The three native tests in `python3 perf_takehome.py` also pass.
- Five machine-level allocator tests in `test_lane_allocate.py` pass, covering
  progressive in-place vector updates, later reads, simultaneous writes,
  initial-zero reuse, and rejection of same-cycle RAW dependencies.
- Two scheduler regression tests in `test_schedule.py` pass. They cover
  preserving a feasible warm schedule with negative inter-unit lags and
  rejecting infeasible hints as incumbents.
- Seeds 0–9 each pass 20,480 retained hash-stage checkpoints and final output
  comparisons in `tests/frozen_problem.py`.
- The verifier checks every executed PC against its logical cycle, and checks
  that the tree, header, and initial index memory are preserved.
- The exported expansion is compared against the complete verified program.
- The initial pause shares the first instruction bundle. It does not shift
  absolute dispatch PCs or add a cycle, and preserves the native two-yield
  harness's initial-memory checkpoint.
- Neither `tests/` nor `problem.py` was changed.

## Main transformations

1. Depth-3 dispatch mixes 14 two-group regions with 32 single-group regions.
   These tables occupy 9,216 bundles. Their handlers also prepare both
   depth-4 children. The first 16 groups in the first traversal select the
   child using MADD with a cached difference; the remaining choices use FLOW.
2. Conditional child copies use otherwise available scalar Store slots. A
   following `vload` gathers eight prepared words into a vector. Temporary
   buffers occupy input-value words after those inputs have been loaded, with
   explicit read/write ordering and final-output restoration.
3. Main-program positions replaced by out-of-line handlers are removed. The
   lowerer relocates absolute jumps and table-address constants; the frozen
   verifier checks the resulting PC-to-logical-cycle mapping on every step.
   Removing these 368 unreachable positions leaves 560 main-program bundles.
4. The last four groups use late gathers. The compressed variant uses MADD
   path updates and FLOW shallow selections. Mirrored memory addresses remove
   several later address selections. There is no separate depth-5 dispatch.
5. Small setup constants and addresses are synthesized selectively. Root I/O
   addresses that affect startup retain direct constant loads.
6. Eleven groups cache four depth-5 grandchildren in the existing depth-3
   handlers. Three vector selections later resolve the depth-5 node, replacing
   eight scalar loads per group (88 total), without adding table entries.
   Removing group 6's cache reduced work by 32 ALU instructions and three FLOW
   selections, while adding eight loads; the resulting schedule reached 931.
7. Output addresses are reconstructed near the final stores, using two
   16-group chains. This adds 30 ALU operations while shortening scalar-address
   lifetimes and allowing a two-cycle scheduling improvement.
8. Depths 0–2 share one tree load and one vector XOR. A comparison of initial
   zero scratch creates the constant-one vector without a scalar constant load.
9. The current mirrored heap retains original depth-5 nodes in scratch,
   reconstructs depth 4 from its biased cache, and backs up depths 6–7 in the
   unused index area. Backups are ordered before bias computation, so original
   loaded values can be released promptly. All modified memory is restored.
10. Ninety-eight named binary operations explicitly select ALU or VALU instead
    of relying only on a global fraction. Moves are first placed in measured
    resource holes with all dependencies checked, then rescheduled. Two rounds
    of these changes reduced 931 cycles to 928 without increasing weighted
    arithmetic work. The current payload is 149,597 bytes before source quoting.

All tree values and input values remain runtime data. Dispatch tables encode
scratch operand choices, not precomputed answers.

## Offline tools

`optimize.py` constructs a fresh SSA graph, interprets it against the reference,
groups dense dispatches, binds scratch, lowers real ISA instructions, and
verifies frozen execution. `schedule.cpp` implements repeated forward/backward
resource-constrained scheduling. `lane_allocate.cpp` packs per-lane live ranges
into scratch. NumPy and a C++17 compiler are development dependencies only.

```sh
python3 export_candidate.py results/compact_934 --write
python3 test_lane_allocate.py
python3 perf_takehome.py
python3 tests/submission_tests.py
git diff --exit-code -- tests/ problem.py
```

`solve_schedule.py` uses optional OR-Tools for an independent scheduling attempt.
On the fixed graph derived from the older 932-cycle checkpoint with scalar fraction 0.286,
its 1,200-second run reported a 902-cycle lower bound and a 936-cycle feasible
SSA schedule. That result applies to the fixed operation choices and contiguous
dispatch regions; it is not a lower bound for the entire task. The 936-cycle
schedule failed the tested scratch allocations and was not promoted.

`search_compact.py` screens mixtures of single-group and two-group dispatches,
rebalances scalar arithmetic, and accepts only allocated schedules whose
expanded static size is at most 10,000 bundles. Accepted search results still
require frozen verification by `export_candidate.py` before submission.

Additional verified experimental choices remain available in `optimize.py`:

- `ahead_bits` computes a high-bit branch condition from the input to h5:
  bit 16 of `x * 65537` equals bit 16 of `x` XOR bit 0 of `x`. Folding the
  multiplication into h5 exposes the condition two dependency cycles earlier.
  It adds a MADD per selected branch. The tested placements did not beat 971
  cycles on their source graph; it is not enabled in the current checkpoint.
- `flow_constants` uses `add_imm` with initial zero scratch for selected
  constants. A 941-cycle variant passed frozen verification with its initial
  pause at logical cycle 1. The exporter selects a free FLOW slot before any
  memory write or jump, preserving the initial checkpoint and all PCs.
- `fold_path4_groups` permits partial path folds. `load_children` now keeps
  temporary stores using runtime child values when conditional loads are also
  enabled; it must not store the child-address constants instead.

The present graph contains 10,966 ALU and 5,483 VALU operations, with weighted
work `10966 + 8*5483 = 54,830`. The machine can issue at most 60 such weighted
operations per cycle. Thus this particular graph needs at least 914 cycles
from arithmetic counts alone (also 914 from its fixed VALU count). Reordering
it cannot reach <900. This is not a lower bound for alternative algorithms or
instruction choices; further progress toward the goal requires reducing work.

## Mirrored-heap experiments

`compact_heap` places reversed, XOR-biased depths 4–7 in tree words 16–255.
For a mirrored-to-mirrored transition, the next address is `2*address + bit`,
removing the FLOW selection of an additive constant. Setup protects the exact
original memory blocks each store overlaps. Restoration protects all reading
gathers and reloads before writing back the original tree; the shallow words
overlapped by the shift and the temporary indices are restored as well.

This freed 56 FLOW operations on the initial eight-group prefetch pattern,
allowing more shallow choices to move from MADD to FLOW. The earlier results
used the original whole-vector allocator:

- Initial allocated variants reached 954 cycles.
- Reusing the existing depth-4 biased cache during restoration reached 950
  cycles (`results/heap_reuse/candidate_004`), verified on seeds 0–9 with all
  20,480 checkpoints and unchanged non-output memory.
- Keeping every original level in scratch did not produce a feasible binding.
- Backing originals up in the dead index area reduces weighted arithmetic to
  about 912 cycles, but the tested schedules failed scratch allocation. Both
  bounded restoration live ranges and late output-address reconstruction were
  tried; those failures are not a global impossibility proof.

The proposal in `results/heap_backup_bound/` has a 913-cycle arithmetic bound
and a 993-cycle SSA schedule. The original binder failed at cycle 303 with
1,508 live words and a largest free span of four words. The new per-lane binder
physically allocated the same schedule (`results/heap_lanes_993/`), which passed
10-seed frozen verification at 993 cycles. Its per-lane peak was 1,497 words.

Profiling then found a depth-7 block loaded at cycle 54 but not backed up until
cycle 915. Ordering backup stores before bias computation shortened those live
ranges. Per-round scalar/VALU choices and additional grandchild prefetches
reached 938 cycles with a 1,338-word peak. Retaining depth-5 originals and
sharing shallow setup reached the current verified 934-cycle entry point.

The current per-lane live-value peak is 1,343 words; the allocated address span
remains 1,536 words. Retaining both depths 6 and 7 at the preserved 938-cycle
schedule still failed 84 tested packing strategies, despite a 1,506-word live
peak. That result is a limitation of those strategies, not a proof of an
impossible allocation.

The lane allocator accounts for every declared word read and write. A value
becomes live after its defining bundle and can share a word with a value whose
last read is in that same bundle. Native bitmap placement is independently
audited with sorted intervals, then checked by frozen execution. The submitted
kernel contains only the resulting ordinary ISA instructions.

Explicit dispatch chains can now be selected in scheduled execution order,
including across rounds. They passed semantic checks but did not improve the
entry point in the screened schedules. Precise entry dependencies (only the
first PC word is read by the entry jump) are enabled; handlers retain all their
own data dependencies. A cleanup-phase indexing bug in per-round scalar
settings was fixed before the successful per-round experiments above.

## Warm scheduling and measured engine moves

Serial placement could make a valid warm schedule worse: a negative lag can
hide an early consumer unit behind a parent whose start is later. The native
scheduler now independently checks a supplied integral schedule's dependencies
and resource usage, and retains a valid one as an incumbent. It periodically
restarts from its best schedule instead of indefinitely drifting away. Native
library builds use a temporary file and atomic replacement.

`search_compact.py` assigns new operations priorities derived from their
consumers instead of sending every new operation to priority zero.
`search_neighbors.py` explores single prefetch changes, retained tree levels,
and short dispatch chains. `rebalance_holes.py` moves independent hash-operation
packs between ALU and VALU using actual free slots, audits the preserved
schedule, then searches from it. Saved configurations reproduce every move
through `scalar_overrides`; neither tool is imported by the submitted kernel.

Mirrored-heap conditional loads are now supported with ascending mirrored
addresses, an exact eligible-operation budget, and restoration dependencies
for every depth-4 read. A 32-load probe passed full ten-seed frozen execution
at 944 cycles and 9,792 bundles (`results/heap_child_loads/candidate_001`). It
is not enabled in the current checkpoint: the additional load pressure was
slower than the copied operands. The wider 24-configuration screen did not
beat 934 cycles before the warm-scheduling and neighborhood refinements.

A separate 240-second CP-SAT attempt on the 934-cycle graph retaining depths
5 and 6 preserved its feasible 934-cycle result and reported a 917-cycle
bound (`results/compact_934_cp2`). This applies only to that fixed graph,
not to the current 928-cycle graph or to alternative algorithms.

## Remaining work

- Reduce actual work or change dispatch structure: several superficially
  balanced graphs have stronger load-release/tail bounds above 900.
- Improve scheduling and allocation together. Aggregate live-word demand can
  fit while contiguous vector allocation fails from fragmentation.
- Keep reporting dynamic cycles, static bundles, and static slot operations
  separately. Lower arithmetic counts alone have not predicted the best actual
  schedule in these experiments.
- Preserve the verified checkpoint while investigating further candidates;
  neither an unallocated schedule nor an unverified cycle estimate is a result.
