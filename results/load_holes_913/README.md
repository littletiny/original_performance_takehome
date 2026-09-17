# Eight more scalar constants use existing LOAD holes

Five budgets retain 913 cycles. Eight scalar ALU constants fit existing LOAD
slots without moving other surviving operations in the warm control. All graph
edges, capacities, bootstrap placement and scratch are checked before search.

`candidate_002` is frozen-verified at **913 / 11,049**, W=54,406 and live peak
1,322. Ten seeds pass all retained checkpoints, PC mappings, outputs, non-output
memory and exact standalone expansion. The later packed layout preserves its
operation times and work at 10,537 bundles in `../compact_913_packed/`.

```sh
python3 flow_constant_holes.py results/flow_holes_rebalance_913/candidate_000 results/load_holes_913 --engine load
python3 export_candidate.py results/load_holes_913/candidate_002
```
