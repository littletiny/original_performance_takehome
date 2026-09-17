# Sub-900 optimization checkpoint

The current verified entry point is **915 cycles with 10,808 static bundles**,
down from the initial 980 cycles.
The requested **<900-cycle target has not been reached**.

On 2026-09-17 the user relaxed the static VLIW instruction-bundle limit from
10,000 to **12,000**
(`len(KernelBuilder.instrs)`). Both the exporter and the standalone decoder
enforce this limit. The current program contains **231,286 individual
engine-slot operations**, including the initial pause; those are a different
quantity from bundles.

The preceding 932-cycle checkpoint used 477,092 static bundles. The current
version reduces that count by **97.73%** and uses 17 fewer dynamic cycles. Its
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
The best checkpoint retained under the previous 10,000-bundle cap is
`results/compact_928/`, commit `5b10cbe` (928 cycles, 9,776 bundles).
The preceding 919-cycle / 10,159-bundle checkpoint remains in
`results/compact_919/` and commit `9c32b1e`.
The 918-cycle / 10,811-bundle checkpoint remains in `results/compact_918/`
and commit `e9da622`.

## Latest verified checkpoint and lower-bound margin

The current source and schedule are recorded in `results/compact_915/`.
Its fixed-graph resource-window lower bound is 912, and its weighted arithmetic
work is 54,577. The latest source/graph hashes still match the promoted 915
checkpoint; the experiments below have not changed the submitted kernel.

`results/sub900_margin_915/` answers the requested engineering margin question.
Bounds of 899, 895 and 890 permit gaps of 0, 4 and 9 cycles respectively at a
899-cycle target. A 890–895 bound is a useful design target, not a guarantee.
The new combined graph with bound 899 has a best frozen-verified schedule of
967 cycles; its 68-cycle gap includes heuristic-search limitations and is not
a proven unavoidable overhead. The same gap cannot be assumed across graphs.

The detailed checkpoint narrative immediately below records the archived 917
version. Later sections document subsequent experiments and the 915 promotion.

## Archived 917 checkpoint

Source configuration and schedule: `results/compact_917/config.json` and
`results/compact_917/best.npz`. Frozen verification and export metadata:
`results/compact_917/verification.json`. Fixed-graph resource counts and
100-cycle utilization windows are in `results/compact_917/analysis.json`.
The release/tail lower bound for this graph is 913 cycles; it is not a global
task bound and is not an achieved schedule.

`python3 plot_utilization.py --output results/compact_917/port_utilization.png`
regenerates the latest PNG, JSON summary, and per-cycle CSV directly from the
submitted program on the frozen machine. The latest measured port occupancy
is ALU 99.35%, VALU 99.33%, LOAD 95.97%, STORE 45.75%, and FLOW 95.53%.
FLOW includes the initial pause and bootstrap jump (876 executed slots,
compared with 874 graph operations). Seeds 0 and 1
take different dispatch paths but have identical per-cycle port occupancy;
both final outputs and all non-output memory pass verification. The plot no
longer treats static bundle addresses as cycles or uses the old 985-cycle
round-activity data.

The submitted `perf_takehome.py` is self-contained: its standard-shape builder
expands an embedded logical schedule and dense dispatch descriptors with Python
standard-library modules. It does not import the optimization tools or read the
saved artifacts. Other shapes retain the previous implementation.

Validation completed:

- The unchanged `python3 tests/submission_tests.py` passes all nine tests at
  917 cycles.
- The three native tests in `python3 perf_takehome.py` also pass.
- Five machine-level allocator tests in `test_lane_allocate.py` pass, covering
  progressive in-place vector updates, later reads, simultaneous writes,
  initial-zero reuse, and rejection of same-cycle RAW dependencies.
- Six scheduler regression tests in `test_schedule.py` pass. They cover
  preserving a feasible warm schedule with negative inter-unit lags and
  rejecting infeasible hints as incumbents, exclusive dispatch execution
  through otherwise free FLOW slots, real no-ops for empty scheduled cycles,
  fixed-address tables with a bootstrap jump and relocated handler returns,
  and aligned vector-store lookups that preserve neighboring rows.
- The memory-order regression in `test_memory_order.py` passes, including a
  negative test that removes a required dependency during partial buffer reuse.
- Seeds 0–9 each pass 20,480 retained hash-stage checkpoints and final output
  comparisons in `tests/frozen_problem.py`.
- Extra seeds 901 and 12345 pass the same checks using the actual submitted
  KernelBuilder. Its full expanded program matches the lowerer's program.
- The verifier checks every executed PC against its logical cycle, and checks
  that the tree, header, and initial index memory are preserved.
- The exported expansion is compared against the complete verified program.
- The initial pause shares logical cycle 1, after the bootstrap jump. It does not shift
  absolute dispatch PCs or add a cycle, and preserves the native two-yield
  harness's initial-memory checkpoint.
- Neither `tests/` nor `problem.py` was changed.

## Main transformations

1. Depth-3 dispatch mixes 15 two-group regions with 30 single-group regions.
   These tables occupy 9,600 bundles. Their handlers also prepare both
   depth-4 children. The first 16 groups in the first traversal select the
   child using MADD with a cached difference; the remaining choices use FLOW.
2. Conditional child copies use otherwise available scalar Store slots. A
   following `vload` gathers eight prepared words into a vector. Temporary
   buffers occupy input-value words after those inputs have been loaded, with
   explicit read/write ordering and final-output restoration.
3. Main-program positions replaced by out-of-line handlers are removed. The
   lowerer relocates absolute jumps and table-address constants; the frozen
   verifier checks the resulting PC-to-logical-cycle mapping on every step.
   Removing these 360 unreachable positions leaves 557 main-program bundles.
   The fixed-address layout also includes 653 unreachable padding bundles.
4. The last four groups use late gathers. The compressed variant uses MADD
   path updates and FLOW shallow selections. Mirrored memory addresses remove
   several later address selections. There is no separate depth-5 dispatch.
5. Small setup constants and addresses are synthesized selectively. One header
   load supplies several existing constants and pointers. Initial zero scratch
   also supplies the zero vector directly. Forty-eight scalar constants now
   use LOAD slots, and the root broadcast reuses the header's runtime root.
6. Eleven groups cache four depth-5 grandchildren in the existing depth-3
   handlers. Three vector selections later resolve the depth-5 node, replacing
   eight scalar loads per group (88 total), without adding table entries.
   Removing group 6's cache reduced work by 32 ALU instructions and three FLOW
   selections, while adding eight loads; the resulting schedule reached 931.
7. Output stores reuse the original input pointers. The preceding checkpoint
   reconstructed them in two 16-group chains; I/O-backed tree backups now keep
   the pointers live already, so those 30 ALU additions are unnecessary.
8. Depths 0–2 share one tree load and one vector XOR. A comparison of initial
   zero scratch creates the constant-one vector without a scalar constant load.
9. The current mirrored heap retains original depth-4 and depth-5 nodes in
   scratch and backs up depths 6–7 in the
   already-read input area after the four 16-word lookup buffers. Input loads
   precede backup writes, and backup reads precede final output stores. This
   removes separate backup pointers and 24 index-clearing stores. Backups still
   precede bias computation, so raw loaded values can be released promptly.
10. Named binary operations explicitly select ALU or VALU instead
    of relying only on a global fraction. Moves are first placed in measured
    resource holes with all dependencies checked, then rescheduled. Two rounds
    of these changes reduced 931 cycles to 928 without increasing weighted
    arithmetic work. Two more contractions helped reach 919 cycles. The current
   payload is 151,870 bytes before source quoting.
11. The first two dispatches in execution order, rounds/groups (3,0) and (3,4),
    share their exit/entry jump. The extra paired lookup at (14,24) and this
    chain remove ten FLOW instructions from the 928-cycle graph.
12. One final depth-3 singleton uses existing temporary-buffer pointers to
    form PC offsets and offsets plus one. A FLOW choice and one MADD replace
    its separate path update and PC addition. This saves seven weighted
    arithmetic slots after the one new scalar constant is counted.

All tree values and input values remain runtime data. Dispatch tables encode
scratch operand choices, not precomputed answers.

## Offline tools

`optimize.py` constructs a fresh SSA graph, interprets it against the reference,
groups dense dispatches, binds scratch, lowers real ISA instructions, and
verifies frozen execution. `schedule.cpp` implements repeated forward/backward
resource-constrained scheduling. `lane_allocate.cpp` packs per-lane live ranges
into scratch. NumPy and a C++17 compiler are development dependencies only.

```sh
python3 export_candidate.py results/compact_917 --write
python3 test_lane_allocate.py
python3 test_schedule.py
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
expanded static size is at most 12,000 bundles. Accepted search results still
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
  memory write, following any bootstrap jump in logical execution order and
  preserving the initial checkpoint and all PCs.
- `fold_path4_groups` permits partial path folds. `load_children` now keeps
  temporary stores using runtime child values when conditional loads are also
  enabled; it must not store the child-address constants instead.

The present graph contains 10,947 ALU and 5,472 VALU operations, with weighted
work `10947 + 8*5472 = 54,723`. The machine can issue at most 60 such weighted
operations per cycle. Thus this particular graph needs at least 913 cycles
from arithmetic counts alone and also from its fixed ALU count. Reordering
or exchanging scalar/vector forms alone cannot reach <900. This is not a
lower bound for alternative algorithms or
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
sharing shallow setup reached the earlier verified 934-cycle checkpoint.

The current per-lane live-value peak is 1,336 words; the allocated address span
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
including across rounds. Early screens found no improvement, but the current
entry point uses the first-pair chain described above. Precise entry dependencies (only the
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
not to the current graph or to alternative algorithms.

## Multi-bundle cases and memory experiments

`dispatch_spans` supports one to four bundles per lookup choice. Additional
STORE phases can transfer the second stream's children or prefetched
grandchildren, followed by vector loads into ordinary SSA values. PC target
scaling, dense-case placement, compaction, and standalone expansion all retain
the per-choice stride. There are no multi-bundle cases in the submitted kernel.

A sixth, offline-only scheduling resource keeps dispatch regions exclusive
while allowing ordinary FLOW selections inside their free phases. It does not
add an engine or change any machine limit. Real self-copy no-ops retain any
otherwise empty scheduled cycle. Merged regions remain limited to 33 cycles.

Full ten-seed frozen execution passed for these development probes:

- One two-bundle case region: `results/span_search/candidate_000`, 930 cycles,
  9,834 bundles.
- Four three-bundle case regions: `results/span_search/candidate_009`,
  948 cycles, 10,244 bundles.
- A three-bundle region chained to a one-bundle region:
  `results/span_chain_probe/candidate_000`, 932 cycles, 9,892 bundles.
- A single I/O lookup buffer reused in saved execution order:
  `results/io_early_ordered/candidate_000`, 943 cycles, 10,183 bundles.
- Early depth-3/depth-4 gathers with I/O backups:
  `results/io_early_ordered/candidate_038`, 933 cycles, 10,061 bundles.

These changes reduced weighted work in some graphs but did not beat 919
cycles. `temp_region_order` can preserve a previously measured reuse order
instead of forcing the builder's group order. An incompatible single-buffer
configuration initially formed a cyclic compound graph; such cases are now
reported as unsupported search candidates without aborting the entire sweep.
They are not treated as proofs of globally impossible schedules.

`balance_resources.py` selects an exact number of scalar/vector hash-pack
migrations. It checks the resulting operation counts instead of relying on
small changes to a global fraction, which can alter many downstream choices.
The named overrides and frozen verification remain the source of truth.

## Fixed-address tables and shared pointer vectors

`pc_address_pools` places the first table at physical PC 14. Four offset
vectors share tree pointer words 14..262 and four share I/O pointer words
2318..2566. Scalar constants occupy lanes of these vectors instead of separate
scratch allocations. Tree and input values remain runtime data.

Compared with the 919-cycle graph, sharing removes seven VALU offset updates
and eight LOAD constants while adding three scalar additions, reducing weighted
arithmetic by 53. Four hash packs then move from VALU to ALU to balance counts.
The resulting graph executes in **918 cycles**, verified on seeds 0–9 plus
additional seeds 901 and 12345, with all hash checkpoints, outputs, PC mappings,
and non-output memory checked.

Cycle zero uses a free FLOW slot to jump past the fixed tables. Logical cycle
one contains the existing initial pause. The remaining main program follows
the tables and ends by falling off the program. No extra dynamic cycle is
introduced. The complete static program has 558 main bundles, 9,600 case
bundles, and 653 padding bundles: **10,811 total**, all counted against 12,000.

The exporter now records main-bundle positions explicitly, so both original
trailing tables and fixed leading tables use the same standalone decoder.
Full expanded programs are compared before export. A frozen-machine regression
executes both choices of all eight small test handlers to check the bootstrap,
fixed table addresses, and relocated return to main code.

`results/pc_address_pools/run.py` reproduces the eight initial scheduling
variants from the preserved 919 checkpoint. Its best result is candidate 001,
now saved as `results/compact_918`. A subsequent 47-configuration neighborhood
screen and six measured-hole rebalance variants did not improve 918. These
screens are allocation-checked search evidence, not additional execution
proofs or global lower bounds. The best submitted program passed the nine
submission tests, three native tests, and ten allocator/scheduler tests.


## Cross-stage XOR audit and further path screens

The fixed hash semantics are preserved. `xor_audit.py` reports that 384 of
512 vector hash groups already defer H6's constant into biased cached nodes.
The remaining 128 groups are in rounds 7, 8, 9, and 15. This is a count of
eliminated hash operations, not net saved cycles; tree biasing and restoration
also perform XORs.

`results/xor_918/audit.json` contains Z3 proofs of the existing H1/H5 MADD
identities, fused H3/H4, H6-to-node bias cancellation, and the 16-bit/19-bit
XOR-shift involutions. It also rejects these specific further templates:

- Absorb H2's constant into a single modified H1 MADD.
- Absorb H6's constant into a single modified H5 MADD.
- Perform either absorption while also allowing any fixed input XOR bias,
  such as an additional bias in cached tree nodes.
- Push H2's constant through the existing fused H4 MADD form.

The rejection equations use necessary low-bit projections. UNSAT proves the
named templates impossible, including arbitrary constant coefficients, but
is not a global claim about all equivalent hash algorithms. The accompanying
README gives the algebra, scope, and reproduction command.

`search_path_choices.py` freezes every existing binary operation's engine
choice before changing traversal modes, avoiding unrelated changes from the
old scalar-fraction counter. It explores FLOW node selection, two-round path
folds, and selected depth-5 dispatch groups. Folded mirrored addresses use
`32 + 4*q3 + 2*b3 + b4`; ordinary-layout addresses use the opposite sign.
`dispatch5_groups` selects individual groups instead of only a suffix.

All 61 screen configurations passed sequential semantic verification before
scheduling. Their best allocated schedule remained 918 cycles. Additional
60-by-800 scheduling attempts on two lower-work graphs also ended at 918 and
919 cycles (`results/path_refine_918/summary.json`). No new configuration was
promoted to the submitted kernel. Ten-seed full frozen execution, with 20,480
stage checks per seed and unchanged non-output memory, passed for:

- `results/path_choices_918/candidate_011`: grandchild path folds, 919 cycles,
  10,812 bundles, weighted arithmetic 54,642.
- `results/path_choices_918/candidate_015`: four prefetch MADDs moved to FLOW,
  918 cycles, 10,811 bundles, weighted arithmetic 54,691. This is retained as
  an alternative search source without changing the submitted program.
- `results/path_choices_918/candidate_024`: folds covering both gathered and
  prefetched depth-5 nodes, 930 cycles, 10,823 bundles.
- `results/path_choices_918/candidate_030`: selected depth-5 dispatch plus
  path folds, 925 cycles, 11,066 bundles.

The existing 918-cycle expanded program remains identical after these tool
changes. The unchanged allocator/scheduler regression suite still passes.


## Cache layouts, restore dependencies, and broader fusion templates

The follow-up in `results/cache_followup_918/README.md` records wide stores of
four-node rows, later depth-4 lookups, exact restore ranges, header-root reuse,
and scalar root mixes. The submitted program remains 918 cycles / 10,811
bundles and is identical after rebuilding the default graph. None of the new
screens improved the score.

Wide stores reduce scalar copying but require later node loads. The complete
48-configuration row screen reached 930 cycles; the 64-configuration depth-4
screen reached 948. Six selected probes, including a paired wide-store case,
passed full ten-seed frozen execution and standalone expansion checks. The
new partial-field dependency regression detects an actual hash mismatch when
a pending old field is overwritten too soon.

`resource_bounds.py` strengthens the current fixed-graph lower bound to 915
cycles by combining resource capacity with earliest releases and required
tails. Its report in `results/resource_windows_918/bounds.json` explains why
lower arithmetic counts alone were insufficient in the row-cache probes.
This bound does not constrain rewritten graphs or alternative algorithms.

`synthesize_xor_pairs.py` extends the XOR audit to two arbitrary affine terms
and to two MADDs separated by XOR. Necessary nine-bit/twelve-bit systems rule
out the named constant-absorption and H3/H4/H5 fusion templates. Nine-bit models
of the latter fail the next projection; no complete 32-bit rewrite is claimed.
See `results/xor_pair_synthesis/README.md` for the exact families and proofs.

## Work-reduction inventory

`results/work_inventory_918/` records the requested enumeration before further
optimization. A per-operation accounting gives 54,723 weighted ALU/VALU slots;
strictly fewer than 900 cycles requires removing at least 783 of those slots
even before startup and tail constraints. The inventory separates existing
component costs from potential savings and covers hash fusion, deferred XOR
bias, path/PC fusion, conditional copies, port transfers, setup/restoration,
and joint dispatch changes. The submitted kernel remains unchanged.

## Measured work-reduction follow-up

`results/work_reduction_followup_918/README.md` records 50 scalar-constant
LOAD variants, 12 preload-priority variants, six measured-hole rebalances,
24 cache/retained-tree exchanges, and 24 shallow-dispatch/address-folding
variants. None improved the submitted 918-cycle result.

The fully verified `results/constant_loads_918/candidate_014` retains 918
cycles and 10,811 bundles while reducing weighted arithmetic by 48 to 54,675.
It is an alternative search source, not a production replacement. Its fixed
resource-window lower bound is 914. Three bounded CP-SAT repairs for 917/916
returned UNKNOWN; a zero-window 918 control reproduced the complete saved
schedule. No timeout is treated as an infeasibility or optimality proof.

Removing all grandchild copies while keeping original tree blocks reduces
weighted work by 512, but introduces a LOAD release/tail bound of 916 and
executes in 967 cycles. The best partial-cache exchange was verified at 942.
Depth-2 dispatch is now available at rounds 2 and 13, with optional two-round
address folding. The best tested shallow-dispatch variant is 920; its folded
comparison saves 40 weighted slots but executes in 923.

Six selected probes passed ten-seed frozen execution with all 20,480 hash
checkpoints per seed, PC mappings, final outputs, non-output memory, and
standalone expansion checked. The 12 allocator/scheduler/memory tests pass.
The rebuilt default graph and expanded actual KernelBuilder program remain
identical; `perf_takehome.py` and the machine/test rules are unchanged.

## PC offset folding and the 917 checkpoint

`pc_bit_pools` uses adjacent scalar addresses to provide both choices for the
last path bit. Its first pool removes one VALU operation, adds one ALU constant
and one FLOW selection, and keeps the table layout unchanged. The default
918 source yielded 919; combining the pool with the 48-constant LOAD variant
retained 918. Header-root reuse and keeping raw depth-4 nodes then produced
the fully verified 917-cycle checkpoint in `results/compact_917/`.

Compared with the former submitted 918 program, weighted arithmetic falls
from 54,723 to 54,652 (71 fewer). The dynamic changes are -15 ALU, -7 VALU,
+48 LOAD, +1 FLOW, and unchanged STORE. This is a measured one-cycle gain;
the static slot count increases despite the one-bundle decrease.

`search_pc_bits.py` records the two 15-configuration screens, and
`results/pc_bits_combine_918/run.py` records 12 combinations. The selected
source is candidate 007; candidate 000 is also verified at 917 with slightly
higher weighted arithmetic. The one-pool and four-pool versions both passed
full ten-seed frozen execution and standalone expansion checks.

`synthesize_product_fusion.py` tests eleven additional nonlinear families for
replacing fused H3/H4 followed by H5 with at most three operations. They
include products with XOR/AND/OR operands and a quadratic superset of products
of two affine terms. Each is UNSAT in a necessary 9-bit or 12-bit projection.
Independent exhaustive Python evaluation checks the projection target against
the original three hash stages and all eight 9-bit models. These are rejection
proofs for the listed families; the fixed hash is unchanged and no global
optimality claim is made. See `results/product_fusion_918/README.md`.

## Remaining work

- The late child-pair proposal is now implemented and validated; its best
  screen is 919. Further copy reductions need a different cost balance.
- Reduce actual work or change dispatch structure: several superficially
  balanced graphs have stronger load-release/tail bounds above 900.
- Improve scheduling and allocation together. Aggregate live-word demand can
  fit while contiguous vector allocation fails from fragmentation.
- Keep reporting dynamic cycles, static bundles, and static slot operations
  separately. Lower arithmetic counts alone have not predicted the best actual
  schedule in these experiments.
- Preserve the verified checkpoint while investigating further candidates;
  neither an unallocated schedule nor an unverified cycle estimate is a result.

## Updated work-reduction inventory at 917

The user's requested enumeration is updated in `results/work_inventory_917/`.
Rebuilding the production graph gives the same saved digest and a fresh
resource-window bound of 913. Arithmetic work is 54,652: a strict sub-900
schedule requires at least 712 fewer weighted arithmetic slots, even before
startup, tails and other ports. The current source still matches the verified
917-cycle / 10,810-bundle checkpoint.

The inventory distinguishes actual category costs from speculative savings,
accounts for the already-used XOR cancellations and PC/root improvements,
and lists eight rewrite directions. It also records the narrow LOAD/FLOW
headroom and the latest local late-pair-layout results, which reduce arithmetic
but do not improve the production cycle count. No new optimization search or
production change is part of this enumeration.

## Memory replication, deep overfetch and the 915 checkpoint

The late-pair-row implementation now has poisoned-padding and delayed-read
regressions, a wider research virtual reference type, and a native allocator
stride parameter. Its 48 layout, 28 chain-order and 16 local-balance screens
do not beat 917; six selected candidates passed full ten-seed frozen execution.
See `results/late_pair_rows_917/README.md`.

`memory_vectors` replaces selected uniform-vector setup operations with eight
STOREs and one VLOAD in a reusable, fully cleared index row. The broad 30-case
screen delayed startup. An 18-case later-use follow-up retained 917 while
reducing weighted arithmetic by 67; six hole rebalances and 15 PC combinations
did not improve cycles. This became the source for a new address representation.

For deep tree addresses A=q+6, VLOAD at q can supply the required node in lane
six while retaining the simpler recurrence q_next=2*q+parity. All eight writes
are allocated and all seven padding lanes are audited as unobserved. The LOAD
instruction count does not increase for overfetch. Thirty configurations and
13 priority/selection follow-ups produced a fully verified **915-cycle /
10,808-bundle** kernel, now embedded in `perf_takehome.py`.

The promoted configuration uses 16 even-numbered groups and transfers four
MADD node selections to the released FLOW slots. Compared with production 917,
it saves 75 weighted arithmetic slots, 28 runtime FLOW slots, two static bundles
and 903 static engine-slot operations. It adds nine LOAD and 73 STORE operations.
The count-only necessary arithmetic cut for strict sub-900 is now 637 W; its
fixed graph has a resource-window lower bound of 912. The goal remains unmet.

The nine submission tests, three native tests and 17 research regressions pass.
Twelve seeds pass all retained hash-stage checkpoints, PC mappings, outputs and
non-output memory. The actual KernelBuilder and standalone decoder both match
the verified program. The updated, visually checked port plot is retained in
`results/compact_915/`; root-level utilization artifacts were left intact.

Further hash synthesis rejects specific four-operation outer-XOR/feedback
families and 18 AND/OR target/family combinations in necessary 9/12-bit systems.
Independent evaluation checks 41,472 original-hash projections and two complete
nine-bit projection models. No 32-bit rewrite or global hash lower bound is
claimed; see `results/affine_outer_917/README.md`.

## Joint reductions and lower-bound margin controls

The builder now permits memory-broadcast rows after disjoint late-child rows.
Optional scalar uniform-operand sharing can remove broadcasts whose consumers
are all scalar; the balancing tools honor complete hash-stage labels. Sixteen
uniform-operand screens did not improve 915. The 240-combination joint inventory
reached a best retained resource-window bound of 902, but no graph <=899.

Explicit cache/constant/chain configurations in `results/cache_constant_trade_915/`
then reached a 899-bound graph. Shorter dispatch chains relax the late LOAD
window, while selected input-pointer LOAD constants improve early arithmetic
readiness. These graph prototypes all rebuild with the saved graph digest and
pass two reference seeds; they are not measured schedules.

`search_bound899.py` screened 60 variants and scheduled the 11 passing the 899
bound. The best allocated result was 967 cycles / 11,900 bundles, now fully
frozen-verified on seeds 0–9, including every retained hash checkpoint, output,
non-output memory, executed-PC mapping and standalone expansion. This provides
a concrete control against treating a lower bound as an achievable schedule.

`early_prefix_groups` forms the depth-5 sibling address after round 3 and adds
the final branch bit after round 4. `prefetch_pair_groups` also issues eight
ordered VLOADs at scratch offsets 0,2,...,14, retaining both possible children
in words 0..15. Real eight-word writes occupy a 22-word virtual span, with all
padding writes allocated. Two FLOW selections later choose the eight nodes.
The load instruction count is unchanged from eight scalar gathers. Relative
to a folded address, this costs eight additional weighted arithmetic slots
per group; the prefix-only variant releases two FLOW selections per group.

The 16-case `search_prefetch_pairs.py` screen does not beat production. Selected
prefix and packed-prefetch candidates are frozen-verified at 915, 917 and 968
cycles. The real-machine microtest checks both branch choices with fixed and
allocated scratch; removing an overlapping-write dependency makes the full
reference negative control fail at the first depth-5 hash checkpoint. All 21
research regressions pass. The production source and graph are unchanged.

For W=54,577, reaching a pure arithmetic capacity target of 895 or 890 requires
at least 877 or 1,177 fewer W respectively. These are optimistic necessary
cuts: stronger per-port windows and scratch still need to fit. The next search
should target actual work reductions together with a small measured scheduling
gap. The <900 goal remains unmet.
