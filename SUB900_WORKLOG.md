# Sub-900 optimization checkpoint

The current verified entry point is **932 cycles**, down from the initial 980.
The requested **<900-cycle target has not been reached**.

The instruction-size limit is being interpreted as the number of static VLIW
instruction bundles (`len(KernelBuilder.instrs)`), consistent with the historical
648,161-bundle comparison. The current program has **477,092 bundles**. These
contain **10,594,362 individual engine-slot operations**, including the initial
pause. Those are different quantities; this checkpoint does not satisfy a limit
of 500,000 individual slot operations.

## Verified checkpoint

Source configuration and schedule: `results/portfolio_1/config.json` and
`results/portfolio_1/best.npz`. Frozen verification and export metadata:
`results/portfolio_1/verification.json`.

The submitted `perf_takehome.py` is self-contained: its standard-shape builder
expands an embedded logical schedule and dense dispatch descriptors with Python
standard-library modules. It does not import the optimization tools or read the
saved artifacts. Other shapes retain the previous implementation.

Validation completed:

- The unchanged `python3 tests/submission_tests.py` passes all nine tests at
  932 cycles.
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

1. Depth-3 dispatch resolves several independent traversals together. Its
   handlers also prepare both depth-4 children, so the next lookup becomes one
   vector selection.
2. Conditional child copies use otherwise available scalar Store slots. A
   following `vload` gathers eight prepared words into a vector. Temporary
   buffers occupy input-value words after those inputs have been loaded, with
   explicit read/write ordering and final-output restoration.
3. Boolean path updates can use two `vselect` instructions instead of a MADD.
   Another fold forms the depth-5 address from two decisions without creating
   the intervening path value. The final depth-3 PC combines prefix and final
   decision directly.
4. A small depth-5 cache reduces the critical block of deep gathers. The last
   four groups use late gathers to avoid the final large dispatch barrier.
5. Small setup constants and addresses are synthesized selectively. Root I/O
   addresses that affect startup retain direct constant loads.

All tree values and input values remain runtime data. Dispatch tables encode
scratch operand choices, not precomputed answers.

## Offline tools

`optimize.py` constructs a fresh SSA graph, interprets it against the reference,
groups dense dispatches, binds scratch, lowers real ISA instructions, and
verifies frozen execution. `schedule.cpp` implements repeated forward/backward
resource-constrained scheduling. NumPy and a C++17 compiler are development
dependencies only.

```sh
python3 export_candidate.py results/portfolio_1 --write
python3 perf_takehome.py
python3 tests/submission_tests.py
git diff --exit-code -- tests/ problem.py
```

`solve_schedule.py` uses optional OR-Tools for an independent scheduling attempt.
On the fixed graph derived from this checkpoint with scalar fraction 0.286,
its 1,200-second run reported a 902-cycle lower bound and a 936-cycle feasible
SSA schedule. That result applies to the fixed operation choices and contiguous
dispatch regions; it is not a lower bound for the entire task. The 936-cycle
schedule failed the tested scratch allocations and was not promoted.

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
