# Late child-pair rows from the 917 checkpoint

The proposed round-14 layout is implemented behind `late_pair_groups`.
Each chosen pair of streams stores adjacent children with one real VSTORE per
stream/handler. Later handlers overwrite padding inside the final payload.
Two VLOADs per stream produce `[L0,R0,L1,R1,...]` blocks; VSELECT consumes the
even conditions and scalar XOR consumes the even results. The normal contiguous
eight-input hash resumes afterward. Hash semantics and the machine are unchanged.

## Physical and memory correctness

- `WideV` gives research virtual references stride 16. The production `pt.V`
  representation retains stride eight. The native allocator accepts a per-value
  stride; real vectors remain eight words and scratch capacity remains 1,536.
- Source spans reserve 14 words and loaded pair blocks reserve nine, so every
  real eight-word vector read is physically in bounds. Every vector write,
  including an unused odd output, receives an allocation lifetime.
- The audit rejects logical consumers or hash checkpoints of odd selection
  outputs. The reference interpreter poisons ignored source/selection padding;
  it does not supply global zero defaults for uninitialized values.
- Reads follow every overlapping payload store. Buffer reuse waits for the
  previous reads, including the previous second block before handler one.
  Same-cycle reads precede writes. Complete 48-word buffers are cleared.
- Three regressions cover poisoned padding, every child choice, merged and
  separate regions, native/fixed allocation, and a delayed-read negative control.
  `WideV.__reduce__` preserves virtual addresses when copying that control graph.

## Measurements

The initial 48 configurations, 28 execution-order chain configurations, and
16 local ALU/VALU rebalance configurations each passed reference semantics on
seeds 0 and 1 before scheduling. None beat the source's 917 cycles.

| Frozen candidate | Cycles | Static bundles | W = ALU + 8*VALU |
| --- | ---: | ---: | ---: |
| `candidate_012`, last two pairs | 919 | 10,812 | 54,638 |
| `candidate_036`, all seven pairs | 927 | 10,820 | 54,558 |
| `candidate_037`, seven pairs with LOAD pointers | 930 | 10,823 | 54,540 |
| `candidate_040`, construction-order chains | 951 | 10,844 | 54,558 |
| `candidate_042`, two buffers | 926 | 10,819 | 54,576 |
| `../late_pair_chain_order_917/candidate_021`, reordered chains | 920 | 10,813 | 54,576 |

Every listed candidate passed seeds 0–9, all 20,480 retained hash checkpoints
per seed, complete PC-to-cycle mapping, final outputs, non-output memory and
standalone decoder expansion. The bounds in `bounds.json` apply to three named
fixed graphs only: the seven-pair graph has arithmetic bound 912 and FLOW bound
910; chaining can reduce its FLOW count but worsens the measured schedule.
Search summaries use 100000 for no retained allocated candidate, not a score.

```sh
python3 search_late_pair_rows.py results/compact_917 results/late_pair_rows_917
PYTHONPATH=. python3 results/late_pair_chain_order_917/run.py
PYTHONPATH=. python3 results/late_pair_local_balance_917/run.py
python3 -m unittest test_late_pair_rows -v
python3 export_candidate.py results/late_pair_rows_917/candidate_036
```
