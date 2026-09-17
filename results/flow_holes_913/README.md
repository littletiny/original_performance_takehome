# Constant ALU work moved into measured FLOW holes

Five conversion budgets find up to thirteen live scalar constants that can be
materialized with FLOW add_imm in existing free slots. Bootstrap space is
reserved. All other surviving operation times stay fixed in the warm control;
all dependencies, resource capacities and scratch allocation are checked.
Two semantic seeds pass before further scheduling.

All budgets retain 913 cycles. `candidate_004` supplies the following pack
rebalance, with W=54,414. These intermediate schedules are not independently
frozen-verified here; the final descendant is fully verified in
`../compact_913_packed/`.

```sh
python3 flow_constant_holes.py results/prologue_constants_914/candidate_012 results/flow_holes_913
```
