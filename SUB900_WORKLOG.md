# Sub-900 optimization checkpoint

The current verified entry point is **936 cycles with 9,784 static bundles**,
down from the initial 980 cycles.
The requested **<900-cycle target has not been reached**.

The user revised the static VLIW instruction-bundle limit to **10,000**
(`len(KernelBuilder.instrs)`). Both the exporter and the standalone decoder
enforce this limit. The current program contains **221,604 individual
engine-slot operations**, including the initial pause; those are a different
quantity from bundles.

The preceding 932-cycle checkpoint used 477,092 static bundles. The current
version reduces that count by **97.95%** at a cost of 4 dynamic cycles. Its
verified artifacts remain at `results/portfolio_1/`, and its submitted source
is preserved in commit `e2c302f`; it is no longer the default implementation.
The first compressed checkpoint, 971 cycles and 9,819 bundles, remains in
`results/compact_971/` and commit `8b4ec3a`.
The depth-5 prefetch checkpoint remains in `results/compact_939/` and commit
`5954122` (939 cycles, 9,787 bundles).

## Verified checkpoint

Source configuration and schedule: `results/compact_936/config.json` and
`results/compact_936/best.npz`. Frozen verification and export metadata:
`results/compact_936/verification.json`. Fixed-graph resource counts and
100-cycle utilization windows are in `results/compact_936/analysis.json`.

The submitted `perf_takehome.py` is self-contained: its standard-shape builder
expands an embedded logical schedule and dense dispatch descriptors with Python
standard-library modules. It does not import the optimization tools or read the
saved artifacts. Other shapes retain the previous implementation.

Validation completed:

- The unchanged `python3 tests/submission_tests.py` passes all nine tests at
  936 cycles.
- The three native tests in `python3 perf_takehome.py` also pass.
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
   depth-4 children. The first traversal selects the child using MADD with
   a cached difference; the final traversal uses `vselect`.
2. Conditional child copies use otherwise available scalar Store slots. A
   following `vload` gathers eight prepared words into a vector. Temporary
   buffers occupy input-value words after those inputs have been loaded, with
   explicit read/write ordering and final-output restoration.
3. Main-program positions replaced by out-of-line handlers are removed. The
   lowerer relocates absolute jumps and table-address constants; the frozen
   verifier checks the resulting PC-to-logical-cycle mapping on every step.
   Removing these 368 unreachable positions leaves 568 main-program bundles.
4. The last four groups use late gathers. The compressed variant uses MADD
   path updates and some MADD shallow selections to leave FLOW capacity for
   the larger number of small dispatches. It omits the separate depth-5
   dispatch cache.
5. Small setup constants and addresses are synthesized selectively. Root I/O
   addresses that affect startup retain direct constant loads.
6. Eight groups (0, 4, ..., 28) also cache four depth-5 grandchildren in the
   existing depth-3 handlers. Three vector selections later resolve the
   depth-5 node, replacing eight scalar loads per group. This removes 64
   runtime loads without adding table entries. Per-round shallow MADD choices
   and per-group child MADD choices allow balancing FLOW and arithmetic.
7. Output addresses are reconstructed near the final stores, using two
   16-group chains. This adds 30 ALU operations while shortening scalar-address
   lifetimes and allowing a two-cycle scheduling improvement.
8. Subsequent scalar uses of constants 1, 2, 8, and C6 reuse the first lane
   of an existing constant vector. This candidate reached 936 cycles.
9. The unused original embedded schedule was removed from the submission
   source. The current embedded payload is 151,385 bytes before source quoting.

All tree values and input values remain runtime data. Dispatch tables encode
scratch operand choices, not precomputed answers.

## Offline tools

`optimize.py` constructs a fresh SSA graph, interprets it against the reference,
groups dense dispatches, binds scratch, lowers real ISA instructions, and
verifies frozen execution. `schedule.cpp` implements repeated forward/backward
resource-constrained scheduling. NumPy and a C++17 compiler are development
dependencies only.

```sh
python3 export_candidate.py results/compact_936 --write
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

The present graph contains 11,036 ALU and 5,522 VALU operations, with weighted
work `11036 + 8*5522 = 55,212`. The machine can issue at most 60 such weighted
operations per cycle. Thus this particular graph needs at least 921 cycles
from arithmetic counts alone (and 921 from its fixed VALU count). Reordering
it cannot reach <900. This is not a lower bound for alternative algorithms or
instruction choices; further progress toward the goal requires reducing work.

## Mirrored-heap experiments

`compact_heap` places reversed, XOR-biased depths 4–7 in tree words 16–255.
For a mirrored-to-mirrored transition, the next address is `2*address + bit`,
removing the FLOW selection of an additive constant. Setup protects the exact
original memory blocks each store overlaps. Restoration protects all reading
gathers and reloads before writing back the original tree; the shallow words
overlapped by the shift and the temporary indices are restored as well.

This frees 56 FLOW operations on the present prefetch pattern, allowing more
shallow choices to move from MADD to FLOW. It has not improved the entry point:

- Initial allocated variants reached 954 cycles.
- Reusing the existing depth-4 biased cache during restoration reached 950
  cycles (`results/heap_reuse/candidate_004`), verified on seeds 0–9 with all
  20,480 checkpoints and unchanged non-output memory.
- Keeping every original level in scratch did not produce a feasible binding.
- Backing originals up in the dead index area reduces weighted arithmetic to
  about 912 cycles, but the tested schedules failed scratch allocation. Both
  bounded restoration live ranges and late output-address reconstruction were
  tried; those failures are not a global impossibility proof.

An unallocated proposal is preserved in `results/heap_backup_bound/`: its
specific graph has a 913-cycle arithmetic bound and a 993-cycle SSA schedule,
but the tested binding fails at cycle 303 with 1,508 live words and a largest
free span of four words. The SSA number is not a machine execution result.
Aliasing scalar constants to an existing vector lane was also explored to
reduce duplicate constant storage; it did not make the tested backup schedules
allocatable.

The heap/backup choices remain disabled in the submitted kernel. The output
address reconstruction and constant aliasing were independently beneficial
on the existing layout and were the changes promoted from this experiment
family.

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
